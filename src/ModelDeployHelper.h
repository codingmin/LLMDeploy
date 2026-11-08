#ifndef LLMDEPLOY_MODEL_DEPLOY_HELPER_H_
#define LLMDEPLOY_MODEL_DEPLOY_HELPER_H_

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class ModelDeployHelper {
 public:
  ModelDeployHelper() = delete;

  static cv::Mat ReadImageFromVideoDevice(const std::string& video_device_path);

  static std::vector<float> PreprocessImage(const std::string& image_path, int target_h,
                                            int target_w);

  static std::vector<float> PreprocessImage(cv::Mat& input, int target_h, int target_w,
                                            bool normalize = true);

  static int GetMaxValueIndex(const std::vector<float>& prediction);
};
#endif
