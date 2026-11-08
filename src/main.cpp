#include <chrono>
#include <memory>
#include <string>

#include "IModelDeploy.h"
#include "LabelString.h"
#include "ModelDeployHelper.h"
#include "ModelDeployTensorRT.h"

void OutputResult(const std::vector<float>& vecFloatResultResNet) {
  const int iIndex = ModelDeployHelper::GetMaxValueIndex(vecFloatResultResNet);
  std::cout << "Predicted class index: " << iIndex;
  std::cout << ";predicted class name: " << imagenet_zh_labels[iIndex] << std::endl;
}

void TestLocalImage(const std::string& ImagePath,
                    const std::shared_ptr<IModelDeploy>& pModelDeploy) {
  const std::string strImagePath = "fishOne.jpg";
  std::vector<float> vecFloatPreprocessedImage =
      ModelDeployHelper::PreprocessImage(strImagePath, 224, 224);
  std::vector<float> vecFloatResultResNet =
      pModelDeploy->Inference(vecFloatPreprocessedImage, nvinfer1::Dims{4, {1, 3, 224, 224}});
  std::cout << "vecFloatResultResNet.size(): " << vecFloatResultResNet.size()
            << "; ImagePath: " << ImagePath << ": ";
  OutputResult(vecFloatResultResNet);
}

void TestVideoDevice(const std::string& strVideoDevicePath,
                     const std::shared_ptr<IModelDeploy>& pModelDeploy) {
  cv::Mat img = ModelDeployHelper::ReadImageFromVideoDevice(strVideoDevicePath);
  imwrite("captured_frame.jpg", img);
  std::vector<float> vecFloatPreprocessedImage = ModelDeployHelper::PreprocessImage(img, 224, 224);
  std::vector<float> vecFloatResultResNet =
      pModelDeploy->Inference(vecFloatPreprocessedImage, nvinfer1::Dims{4, {1, 3, 224, 224}});
  std::cout << strVideoDevicePath << ": ";
  OutputResult(vecFloatResultResNet);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <path_to_onnx_model_file>" << std::endl;
    return 1;
  }
  std::string strModelPath = argv[1];
  std::cout << "Loading model from: " << strModelPath << std::endl;

  std::shared_ptr<IModelDeploy> pModelDeploy = std::make_shared<ModelDeployTensorRT>();
  pModelDeploy->LoadModel(strModelPath);

  const int iRepeatCount = 5;
  const std::string strImagePath = "fishOne.jpg";
  double totalTimeMs = 0.0;
  for (int i = 0; i < iRepeatCount; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    TestLocalImage(strImagePath, pModelDeploy);
    const double elapsedMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    totalTimeMs += elapsedMs;
    std::cout << "[Timing] TestLocalImage #" << i << ": " << elapsedMs << " ms" << std::endl;
  }
  if (iRepeatCount > 0) {
    std::cout << "[Timing] average: " << totalTimeMs / iRepeatCount << " ms over " << iRepeatCount
              << " runs" << std::endl;
  }

  // const std::string strDevicePath = "/dev/video0";
  // TestVideoDevice(strDevicePath,pModelDeploy);

  return 0;
}