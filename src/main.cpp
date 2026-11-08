#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "IModelDeploy.h"
#include "LabelString.h"
#include "ModelDeployHelper.h"
#include "ModelDeployTensorRT.h"

void OutputResult(const std::vector<float>& result) {
  const int max_index = ModelDeployHelper::GetMaxValueIndex(result);
  std::cout << "Predicted class index: " << max_index;
  std::cout << ";predicted class name: " << imagenet_zh_labels[max_index] << std::endl;
}

void TestLocalImage(const std::string& image_path,
                    const std::shared_ptr<IModelDeploy>& model_deploy) {
  const std::string local_image_path = "fishOne.jpg";
  std::vector<float> preprocessed_image =
      ModelDeployHelper::PreprocessImage(local_image_path, 224, 224);
  std::vector<float> result =
      model_deploy->Inference(preprocessed_image, nvinfer1::Dims{4, {1, 3, 224, 224}});
  std::cout << "vecFloatResultResNet.size(): " << result.size() << "; ImagePath: " << image_path
            << ": ";
  OutputResult(result);
}

void TestVideoDevice(const std::string& video_device_path,
                     const std::shared_ptr<IModelDeploy>& model_deploy) {
  cv::Mat img = ModelDeployHelper::ReadImageFromVideoDevice(video_device_path);
  imwrite("captured_frame.jpg", img);
  std::vector<float> preprocessed_image = ModelDeployHelper::PreprocessImage(img, 224, 224);
  std::vector<float> result =
      model_deploy->Inference(preprocessed_image, nvinfer1::Dims{4, {1, 3, 224, 224}});
  std::cout << video_device_path << ": ";
  OutputResult(result);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <path_to_onnx_model_file>" << std::endl;
    return 1;
  }
  std::string model_path = argv[1];
  std::cout << "Loading model from: " << model_path << std::endl;

  std::shared_ptr<IModelDeploy> model_deploy = std::make_shared<ModelDeployTensorRT>();
  model_deploy->LoadModel(model_path);

  constexpr int kRepeatCount = 5;
  const std::string image_path = "fishOne.jpg";
  double total_time_ms = 0.0;
  for (int i = 0; i < kRepeatCount; ++i) {
    const auto start_time = std::chrono::steady_clock::now();
    TestLocalImage(image_path, model_deploy);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time)
            .count();
    total_time_ms += elapsed_ms;
    std::cout << "[Timing] TestLocalImage #" << i << ": " << elapsed_ms << " ms" << std::endl;
  }
  if (kRepeatCount > 0) {
    std::cout << "[Timing] average: " << total_time_ms / kRepeatCount << " ms over "
              << kRepeatCount << " runs" << std::endl;
  }

  return 0;
}
