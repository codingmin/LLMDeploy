#ifndef _IMODEL_DEPLOY_H_
#define _IMODEL_DEPLOY_H_

#include <iostream>
#include <string>
#include <vector>

#include "NvInfer.h"

class IModelDeploy {
 public:
  IModelDeploy() = default;
  virtual ~IModelDeploy() = default;

  virtual void LoadModel(const std::string& strModelPath) = 0;

  virtual std::vector<float> Inference(const std::vector<float>& vecFloatInputImage,
                                       const nvinfer1::Dims& dim) = 0;
};
#endif
