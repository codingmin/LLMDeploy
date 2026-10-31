#ifndef _MODEL_DEPLOY_HELPER_HPP_
#define _MODEL_DEPLOY_HELPER_HPP_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class ModelDeployHelper
{
public:
    ModelDeployHelper()=delete;

    static cv::Mat ReadImageFromVideoDevice(const std::string& strVideoDevicePath);

    static std::vector<float> PreprocessImage(cv::Mat& input,int targetH, int targetW,bool normalize=true);

    static std::vector<float> PreprocessImage(const std::string& strImagePath, int targetH, int targetW);

    static int GetMaxValueIndex(const std::vector<float>& vecFloatPrediction);
};
#endif // _MODEL_DEPLOY_HELPER_HPP_