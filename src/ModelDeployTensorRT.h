#ifndef _MODEL_DEPLOY_TENSORRT_H_
#define _MODEL_DEPLOY_TENSORRT_H_

#include <cuda_runtime.h>

#include <memory>

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

  virtual void LoadModel(const std::string& strModelPath) override;

  virtual std::vector<float> Inference(const std::vector<float>& vecFloatInputImage,
                                       const nvinfer1::Dims& dim) override;

 private:
  bool InitInferenceResources();

  std::shared_ptr<nvinfer1::ILogger> m_logger;
  std::vector<char> m_vecEngineFile;
  bool bInferenceEngineBuilt = false;
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
  // 以下资源初始化一次、全程复用（cudaFree/cudaFreeHost 会隐式同步 GPU，禁止每次调用创建销毁）
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;
  cudaStream_t m_stream = nullptr;
  void* m_dInput = nullptr;    // 设备端输入缓冲（固定 shape）
  void* m_dOutput = nullptr;   // 设备端输出缓冲
  float* m_pInput = nullptr;   // host 端 pinned 输入缓冲（支持异步 DMA）
  float* m_pOutput = nullptr;  // host 端 pinned 输出缓冲
  size_t m_inputCount = 0;
  size_t m_outputCount = 0;
};

#endif  // _MODEL_DEPLOY_TENSORRT_H_