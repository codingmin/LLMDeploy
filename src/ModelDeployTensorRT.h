#ifndef LLMDEPLOY_MODEL_DEPLOY_TENSORRT_H_
#define LLMDEPLOY_MODEL_DEPLOY_TENSORRT_H_

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "IModelDeploy.h"
#include "Logger.h"

class ModelDeployTensorRT : public IModelDeploy {
 public:
  ModelDeployTensorRT() : m_logger(std::make_shared<MyLogger>()) {}

  ~ModelDeployTensorRT();

  ModelDeployTensorRT(const ModelDeployTensorRT& other) = delete;
  ModelDeployTensorRT& operator=(const ModelDeployTensorRT& other) = delete;

  ModelDeployTensorRT(const ModelDeployTensorRT&& other) = delete;
  ModelDeployTensorRT& operator=(const ModelDeployTensorRT&& other) = delete;

  virtual void LoadModel(const std::string& model_path) override;

  virtual std::vector<float> Inference(const std::vector<float>& input_image,
                                       const nvinfer1::Dims& input_dims) override;

 private:
  bool InitInferenceResources();
  void Warmup();

  std::shared_ptr<nvinfer1::ILogger> m_logger;
  std::vector<char> m_engine_file;
  bool is_engine_built = false;
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;
  cudaStream_t m_stream = nullptr;
  void* m_d_input = nullptr;
  void* m_d_output = nullptr;
  float* m_p_input = nullptr;
  float* m_p_output = nullptr;
  size_t m_input_count = 0;
  size_t m_output_count = 0;
};

#endif
