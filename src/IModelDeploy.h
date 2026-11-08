#ifndef LLMDEPLOY_I_MODEL_DEPLOY_H_
#define LLMDEPLOY_I_MODEL_DEPLOY_H_

#include <string>
#include <vector>

#include "NvInfer.h"

class IModelDeploy {
 public:
  IModelDeploy() = default;
  virtual ~IModelDeploy() = default;

  virtual void LoadModel(const std::string& model_path) = 0;

  virtual std::vector<float> Inference(const std::vector<float>& input_image,
                                       const nvinfer1::Dims& input_dims) = 0;
};
#endif
