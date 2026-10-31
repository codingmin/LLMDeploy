#include "ModelDeployTensorRT.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include<cuda_runtime.h>

void ModelDeployTensorRT::LoadModel(const std::string& strModelPath)
{
	if(bInferenceEngineBuilt)
	{
		return;
	}
	using namespace nvinfer1;
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

	IOptimizationProfile* profile = builder->createOptimizationProfile();

	for(int32_t i=0;i<network->getNbInputs();++i)
	{
		ITensor* inputTensor = network->getInput(i);
		Dims dims = inputTensor->getDimensions();
		bool bDynamic = false;
		Dims smallest = dims;
		Dims largest = dims;
		for(int32_t j=0;j<dims.nbDims;++j)
		{
			if(dims.d[j] == -1)
			{
				bDynamic = true;
				smallest.d[j] = 1;
				largest.d[j] = 2048;
			}
		}
		if(bDynamic)
		{
			profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, smallest);
			profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, smallest);
			profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, largest);
		}
	}
	config->addOptimizationProfile(profile);

	std::unique_ptr<IHostMemory> serialized{builder->buildSerializedNetwork(*network, *config)};
	if(!serialized)
	{
		throw std::logic_error("Failed to build serialized network");
	}
	m_vecEngineFile.assign(static_cast<const char*>(serialized->data()), static_cast<const char*>(serialized->data()) + serialized->size());

	std::unique_ptr<IRuntime> runtime{createInferRuntime(*m_logger)};
	m_engine.reset(runtime->deserializeCudaEngine(m_vecEngineFile.data(), m_vecEngineFile.size()));
	bInferenceEngineBuilt = true;
}
	
std::vector<float> ModelDeployTensorRT::Inference(const std::vector<float>& vecFloatInputImage,
	const nvinfer1::Dims& inputDims)  
{
	if(bInferenceEngineBuilt)
	{
		using namespace nvinfer1;
		std::unique_ptr<IExecutionContext> context{m_engine->createExecutionContext()};

		char const* const inputName = m_engine->getIOTensorName(0);
		char const* const outputName = m_engine->getIOTensorName(1);
		context->setInputShape(inputName, inputDims);
		
		nvinfer1::Dims outputDims = context->getTensorShape(outputName);
		size_t outputCount = 1;
		for (int32_t i = 0; i < outputDims.nbDims; ++i)
		{
			outputCount *= outputDims.d[i];
		}

		cudaStream_t stream;
		cudaStreamCreate(&stream);
		void* dInput{nullptr};
		void* dOutput{nullptr};
		cudaMalloc(&dInput, vecFloatInputImage.size() * sizeof(float));
		cudaMalloc(&dOutput, outputCount * sizeof(float));
		cudaMemcpyAsync(dInput, vecFloatInputImage.data(), vecFloatInputImage.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
		context->setTensorAddress(inputName,dInput);
		context->setTensorAddress(outputName, dOutput);

		if(!context->enqueueV3(stream))
		{
			throw std::runtime_error("enqueueV3 failed");
		}

		std::vector<float> vecFloatOutput(outputCount);
		cudaMemcpyAsync(vecFloatOutput.data(), dOutput, vecFloatOutput.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		cudaFree(dInput);
		cudaFree(dOutput);	
		cudaStreamDestroy(stream);
		return vecFloatOutput;
	}
	else
	{
		throw std::runtime_error("Inference engine is not built yet");	
	}
}
