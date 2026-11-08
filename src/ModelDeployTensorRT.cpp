#include "ModelDeployTensorRT.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include<cuda_runtime.h>
#include<fstream>
#include<iostream>

// 引擎缓存文件路径：与模型同目录同名，后缀替换为 .engine（如 model.onnx -> model.engine）
static std::string MakeEngineFilePath(const std::string& strModelPath)
{
	size_t pos = strModelPath.find_last_of('.');
	if(pos == std::string::npos)
	{
		return strModelPath + ".engine";
	}
	return strModelPath.substr(0, pos) + ".engine";
}

void ModelDeployTensorRT::LoadModel(const std::string& strModelPath)
{
	if(bInferenceEngineBuilt)
	{
		return;
	}
	using namespace nvinfer1;
	std::unique_ptr<IRuntime> runtime{createInferRuntime(*m_logger)};

	// 缓存命中：直接从本地 .engine 文件反序列化，跳过约 8 秒的 buildSerializedNetwork
	std::string strEnginePath = MakeEngineFilePath(strModelPath);
	std::ifstream fin(strEnginePath, std::ios::binary);
	if(fin)
	{
		fin.seekg(0, std::ios::end);
		const std::streamsize fileSize = static_cast<std::streamsize>(fin.tellg());
		fin.seekg(0, std::ios::beg);
		m_vecEngineFile.resize(static_cast<size_t>(fileSize));
		fin.read(m_vecEngineFile.data(), fileSize);
		m_engine.reset(runtime->deserializeCudaEngine(m_vecEngineFile.data(), m_vecEngineFile.size()));
		if(m_engine)
		{
			std::cout << "[ModelDeployTensorRT] loaded engine from cache: " << strEnginePath << std::endl;
			if(!InitInferenceResources())
			{
				throw std::runtime_error("Failed to init inference resources");
			}
			bInferenceEngineBuilt = true;
			return;
		}
		// 缓存无效（TensorRT 版本或显卡变化等），回退到重新构建
		std::cout << "[ModelDeployTensorRT] engine cache invalid, rebuilding: " << strEnginePath << std::endl;
		m_vecEngineFile.clear();
		m_engine.reset();
	}

	std::unique_ptr<IBuilder> builder{nvinfer1::createInferBuilder(*m_logger)};

	size_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
	std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(flag)};

	std::unique_ptr<nvonnxparser::IParser> parser {nvonnxparser::createParser(*network, *m_logger)};

	if(!parser->parseFromFile(strModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
	{
		throw std::logic_error("Failed to parse ONNX file");
	}

	std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
	config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GiB

	// 固定输入 shape 1x3x224x224：MIN=OPT=MAX 相同，TRT 可为该 shape 选最优 tactic，
	// 避免 MIN=1/OPT=1/MAX=2048 的大范围导致 conv 回退 generic 核（推理慢约 50 倍）且构建耗时 8 秒
	IOptimizationProfile* profile = builder->createOptimizationProfile();
	const Dims fixedDims{4, {1, 3, 224, 224}};
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, fixedDims);
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, fixedDims);
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, fixedDims);
	config->addOptimizationProfile(profile);

	std::unique_ptr<IHostMemory> serialized{builder->buildSerializedNetwork(*network, *config)};
	if(!serialized)
	{
		throw std::logic_error("Failed to build serialized network");
	}
	m_vecEngineFile.assign(static_cast<const char*>(serialized->data()), static_cast<const char*>(serialized->data()) + serialized->size());

	// 首次构建结果落盘，下次启动直接加载
	std::ofstream fout(strEnginePath, std::ios::binary);
	if(fout && fout.write(m_vecEngineFile.data(), static_cast<std::streamsize>(m_vecEngineFile.size())))
	{
		std::cout << "[ModelDeployTensorRT] engine saved to: " << strEnginePath << std::endl;
	}
	else
	{
		std::cout << "[ModelDeployTensorRT] warning: failed to write engine file: " << strEnginePath << std::endl;
	}

	m_engine.reset(runtime->deserializeCudaEngine(m_vecEngineFile.data(), m_vecEngineFile.size()));
	if(!InitInferenceResources())
	{
		throw std::runtime_error("Failed to init inference resources");
	}
	bInferenceEngineBuilt = true;
}

ModelDeployTensorRT::~ModelDeployTensorRT()
{
	if(m_pInput)
	{
		cudaFreeHost(m_pInput);
	}
	if(m_pOutput)
	{
		cudaFreeHost(m_pOutput);
	}
	if(m_dInput)
	{
		cudaFree(m_dInput);
	}
	if(m_dOutput)
	{
		cudaFree(m_dOutput);
	}
	if(m_stream)
	{
		cudaStreamDestroy(m_stream);
	}
}

// 一次性初始化推理资源：context、stream、固定 shape 的输入输出缓冲（设备端 + pinned）
// 这些资源全程复用；cudaMalloc/cudaFree 会隐式同步 GPU，不能放在每次推理调用里
bool ModelDeployTensorRT::InitInferenceResources()
{
	m_context.reset(m_engine->createExecutionContext());
	if(!m_context)
	{
		return false;
	}

	char const* const inputName = m_engine->getIOTensorName(0);
	char const* const outputName = m_engine->getIOTensorName(1);
	const nvinfer1::Dims inputDims = m_context->getTensorShape(inputName);
	const nvinfer1::Dims outputDims = m_context->getTensorShape(outputName);
	if(inputDims.nbDims <= 0 || outputDims.nbDims <= 0)
	{
		return false;
	}
	m_inputCount = 1;
	for(int32_t i = 0; i < inputDims.nbDims; ++i)
	{
		m_inputCount *= inputDims.d[i];
	}
	m_outputCount = 1;
	for(int32_t i = 0; i < outputDims.nbDims; ++i)
	{
		m_outputCount *= outputDims.d[i];
	}

	if(cudaStreamCreate(&m_stream) != cudaSuccess)
	{
		return false;
	}
	if(cudaMalloc(&m_dInput, m_inputCount * sizeof(float)) != cudaSuccess)
	{
		return false;
	}
	if(cudaMalloc(&m_dOutput, m_outputCount * sizeof(float)) != cudaSuccess)
	{
		return false;
	}
	if(cudaMallocHost(reinterpret_cast<void**>(&m_pInput), m_inputCount * sizeof(float)) != cudaSuccess)
	{
		return false;
	}
	if(cudaMallocHost(reinterpret_cast<void**>(&m_pOutput), m_outputCount * sizeof(float)) != cudaSuccess)
	{
		return false;
	}

	m_context->setTensorAddress(inputName, m_dInput);
	m_context->setTensorAddress(outputName, m_dOutput);
	return true;
}
	
std::vector<float> ModelDeployTensorRT::Inference(const std::vector<float>& vecFloatInputImage,
	const nvinfer1::Dims& inputDims)
{
	// 固定 shape 引擎：输入 shape 已在 InitInferenceResources 固化，运行时仅校验元素数
	(void)inputDims;
	if(!bInferenceEngineBuilt || !m_context)
	{
		throw std::runtime_error("Inference engine is not built yet");
	}
	if(vecFloatInputImage.size() != m_inputCount)
	{
		throw std::runtime_error("input size mismatch: expect " + std::to_string(m_inputCount));
	}

	// 拷入 pinned 缓冲后全程异步：H2D -> enqueueV3 -> D2H 依次排在同一条流上
	std::copy(vecFloatInputImage.begin(), vecFloatInputImage.end(), m_pInput);
	cudaMemcpyAsync(m_dInput, m_pInput, m_inputCount * sizeof(float), cudaMemcpyHostToDevice, m_stream);

	if(!m_context->enqueueV3(m_stream))
	{
		throw std::runtime_error("enqueueV3 failed");
	}

	cudaMemcpyAsync(m_pOutput, m_dOutput, m_outputCount * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	return std::vector<float>(m_pOutput, m_pOutput + m_outputCount);
}
