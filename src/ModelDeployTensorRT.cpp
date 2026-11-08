#include "ModelDeployTensorRT.h"

#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "NvInfer.h"
#include "NvOnnxParser.h"

static std::string MakeEngineFilePath(const std::string& model_path) {
  size_t dot_pos = model_path.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return model_path + ".engine";
  }
  return model_path.substr(0, dot_pos) + ".engine";
}

void ModelDeployTensorRT::LoadModel(const std::string& model_path) {
  if (is_engine_built) {
    return;
  }
  using namespace nvinfer1;
  std::unique_ptr<IRuntime> runtime{createInferRuntime(*m_logger)};

  std::string engine_path = MakeEngineFilePath(model_path);
  std::ifstream fin(engine_path, std::ios::binary);
  if (fin) {
    fin.seekg(0, std::ios::end);
    const std::streamsize file_size = static_cast<std::streamsize>(fin.tellg());
    fin.seekg(0, std::ios::beg);
    m_engine_file.resize(static_cast<size_t>(file_size));
    fin.read(m_engine_file.data(), file_size);
    m_engine.reset(runtime->deserializeCudaEngine(m_engine_file.data(), m_engine_file.size()));
    if (m_engine) {
      std::cout << "[ModelDeployTensorRT] loaded engine from cache: " << engine_path << std::endl;
      if (!InitInferenceResources()) {
        throw std::runtime_error("Failed to init inference resources");
      }
      is_engine_built = true;
      Warmup();
      return;
    }
    std::cout << "[ModelDeployTensorRT] engine cache invalid, rebuilding: " << engine_path
              << std::endl;
    m_engine_file.clear();
    m_engine.reset();
  }

  std::unique_ptr<IBuilder> builder{nvinfer1::createInferBuilder(*m_logger)};

  size_t flags =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(flags)};

  std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, *m_logger)};

  if (!parser->parseFromFile(model_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::logic_error("Failed to parse ONNX file");
  }

  std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

  IOptimizationProfile* profile = builder->createOptimizationProfile();
  const Dims fixed_dims{4, {1, 3, 224, 224}};
  profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, fixed_dims);
  profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, fixed_dims);
  profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, fixed_dims);
  config->addOptimizationProfile(profile);

  std::unique_ptr<IHostMemory> serialized{builder->buildSerializedNetwork(*network, *config)};
  if (!serialized) {
    throw std::logic_error("Failed to build serialized network");
  }
  m_engine_file.assign(static_cast<const char*>(serialized->data()),
                       static_cast<const char*>(serialized->data()) + serialized->size());

  std::ofstream fout(engine_path, std::ios::binary);
  if (fout &&
      fout.write(m_engine_file.data(), static_cast<std::streamsize>(m_engine_file.size()))) {
    std::cout << "[ModelDeployTensorRT] engine saved to: " << engine_path << std::endl;
  } else {
    std::cout << "[ModelDeployTensorRT] warning: failed to write engine file: " << engine_path
              << std::endl;
  }

  m_engine.reset(runtime->deserializeCudaEngine(m_engine_file.data(), m_engine_file.size()));
  if (!InitInferenceResources()) {
    throw std::runtime_error("Failed to init inference resources");
  }
  is_engine_built = true;
  Warmup();
}

ModelDeployTensorRT::~ModelDeployTensorRT() {
  if (m_p_input) {
    cudaFreeHost(m_p_input);
  }
  if (m_p_output) {
    cudaFreeHost(m_p_output);
  }
  if (m_d_input) {
    cudaFree(m_d_input);
  }
  if (m_d_output) {
    cudaFree(m_d_output);
  }
  if (m_stream) {
    cudaStreamDestroy(m_stream);
  }
}

bool ModelDeployTensorRT::InitInferenceResources() {
  m_context.reset(m_engine->createExecutionContext());
  if (!m_context) {
    return false;
  }

  char const* const input_name = m_engine->getIOTensorName(0);
  char const* const output_name = m_engine->getIOTensorName(1);
  const nvinfer1::Dims input_dims = m_context->getTensorShape(input_name);
  const nvinfer1::Dims output_dims = m_context->getTensorShape(output_name);
  if (input_dims.nbDims <= 0 || output_dims.nbDims <= 0) {
    return false;
  }
  m_input_count = 1;
  for (int32_t i = 0; i < input_dims.nbDims; ++i) {
    m_input_count *= input_dims.d[i];
  }
  m_output_count = 1;
  for (int32_t i = 0; i < output_dims.nbDims; ++i) {
    m_output_count *= output_dims.d[i];
  }

  if (cudaStreamCreate(&m_stream) != cudaSuccess) {
    return false;
  }
  if (cudaMalloc(&m_d_input, m_input_count * sizeof(float)) != cudaSuccess) {
    return false;
  }
  if (cudaMalloc(&m_d_output, m_output_count * sizeof(float)) != cudaSuccess) {
    return false;
  }
  if (cudaMallocHost(reinterpret_cast<void**>(&m_p_input), m_input_count * sizeof(float)) !=
      cudaSuccess) {
    return false;
  }
  if (cudaMallocHost(reinterpret_cast<void**>(&m_p_output), m_output_count * sizeof(float)) !=
      cudaSuccess) {
    return false;
  }

  m_context->setTensorAddress(input_name, m_d_input);
  m_context->setTensorAddress(output_name, m_d_output);
  return true;
}

void ModelDeployTensorRT::Warmup() {
  constexpr int kWarmupRuns = 3;
  const nvinfer1::Dims fixed_dims{4, {1, 3, 224, 224}};
  const std::vector<float> warmup_input(m_input_count, 0.0f);

  const auto start_time = std::chrono::steady_clock::now();
  for (int i = 0; i < kWarmupRuns; ++i) {
    Inference(warmup_input, fixed_dims);
  }
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time)
          .count();
  std::cout << "[ModelDeployTensorRT] warmup: " << kWarmupRuns << " runs in " << elapsed_ms
            << " ms" << std::endl;
}

std::vector<float> ModelDeployTensorRT::Inference(const std::vector<float>& input_image,
                                                  const nvinfer1::Dims& input_dims) {
  (void)input_dims;
  if (!is_engine_built || !m_context) {
    throw std::runtime_error("Inference engine is not built yet");
  }
  if (input_image.size() != m_input_count) {
    throw std::runtime_error("input size mismatch: expect " + std::to_string(m_input_count));
  }

  std::copy(input_image.begin(), input_image.end(), m_p_input);
  cudaMemcpyAsync(m_d_input, m_p_input, m_input_count * sizeof(float), cudaMemcpyHostToDevice,
                  m_stream);

  if (!m_context->enqueueV3(m_stream)) {
    throw std::runtime_error("enqueueV3 failed");
  }

  cudaMemcpyAsync(m_p_output, m_d_output, m_output_count * sizeof(float), cudaMemcpyDeviceToHost,
                  m_stream);
  cudaStreamSynchronize(m_stream);
  return std::vector<float>(m_p_output, m_p_output + m_output_count);
}
