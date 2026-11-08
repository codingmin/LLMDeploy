#include "ModelDeployHelper.h"

#include <stdexcept>

cv::Mat ModelDeployHelper::ReadImageFromVideoDevice(const std::string& video_device_path) {
  cv::VideoCapture cap(video_device_path, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    throw std::runtime_error("Failed to open video device: " + video_device_path);
  }
  cv::Mat frame;
  cap >> frame;
  if (frame.empty()) {
    throw std::runtime_error("Failed to capture frame from video device: " + video_device_path);
  }

  cap.release();
  cv::destroyAllWindows();
  return frame;
}

std::vector<float> ModelDeployHelper::PreprocessImage(const std::string& image_path, int target_h,
                                                      int target_w) {
  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }
  return PreprocessImage(img, target_h, target_w);
}

std::vector<float> ModelDeployHelper::PreprocessImage(cv::Mat& input, int target_h, int target_w,
                                                      bool normalize) {
  cv::resize(input, input, cv::Size(target_w, target_h));

  cv::Mat rgb;
  cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);

  cv::Mat f32;
  rgb.convertTo(f32, CV_32FC3);

  if (normalize) {
    f32 /= 255.0f;
    const cv::Scalar mean{0.485f, 0.456f, 0.406f};
    const cv::Scalar std_dev{0.229f, 0.224f, 0.225f};
    f32 = (f32 - mean) / std_dev;
  }

  std::vector<float> output(3 * target_h * target_w, 0.0f);
  float* r_plane = output.data() + 0 * target_h * target_w;
  float* g_plane = output.data() + 1 * target_h * target_w;
  float* b_plane = output.data() + 2 * target_h * target_w;

  for (int y = 0; y < target_h; ++y) {
    const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
    for (int x = 0; x < target_w; ++x) {
      const cv::Vec3f& value = row[x];
      r_plane[y * target_w + x] = value[0];
      g_plane[y * target_w + x] = value[1];
      b_plane[y * target_w + x] = value[2];
    }
  }
  return output;
}

int ModelDeployHelper::GetMaxValueIndex(const std::vector<float>& prediction) {
  int max_index = 0;
  for (size_t i = 1; i < prediction.size(); ++i) {
    if (prediction[i] > prediction[max_index]) {
      max_index = static_cast<int>(i);
    }
  }
  return max_index;
}
