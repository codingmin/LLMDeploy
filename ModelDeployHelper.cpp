#include "ModelDeployHelper.h"

cv::Mat ModelDeployHelper::ReadImageFromVideoDevice(const std::string& strVideoDevicePath)
{
    cv::VideoCapture cap(strVideoDevicePath, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        throw std::runtime_error("Failed to open video device: " + strVideoDevicePath);
    }
    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
    {
        throw std::runtime_error("Failed to capture frame from video device: " + strVideoDevicePath);
    }
	
    cap.release();
    cv::destroyAllWindows();
    return frame;

}

std::vector<float> ModelDeployHelper::PreprocessImage(const std::string& strImagePath, int targetH, int targetW)
{
    cv::Mat img = cv::imread(strImagePath);
    if (img.empty())
    {
        throw std::runtime_error("Failed to read image: " + strImagePath);
    }
    return PreprocessImage(img, targetH, targetW);
}

 std::vector<float> ModelDeployHelper::PreprocessImage(cv::Mat& img,int targetH, int targetW,bool normalize)
 {
    // resize
    cv::resize(img, img, cv::Size(targetW, targetH));

    // BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    // uint8 -> float32
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3);

    if (normalize)
    {
        // 先 /255 到 [0,1]，再减均值除方差（ImageNet）
        f32 /= 255.0f;
        const cv::Scalar mean{0.485f, 0.456f, 0.406f};
        const cv::Scalar std{0.229f, 0.224f, 0.225f};
        f32 = (f32 - mean) / std;
    }

    std::vector<float> out(3 * targetH * targetW, 0.0f);
    float* pR = out.data() + 0 * targetH * targetW;
    float* pG = out.data() + 1 * targetH * targetW;
    float* pB = out.data() + 2 * targetH * targetW;

    for (int y = 0; y < targetH; ++y)
    {
        const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < targetW; ++x)
        {
            auto& v = row[x];
            pR[y * targetW + x] = v[0];
            pG[y * targetW + x] = v[1];
            pB[y * targetW + x] = v[2];
        }
    }
    return out;
 }

int ModelDeployHelper::GetMaxValueIndex(const std::vector<float>& vecFloatPrediction)
{
    int iIndex = 0;
    for(int i = 1; i < vecFloatPrediction.size(); ++i)
    {
        if (vecFloatPrediction[i] > vecFloatPrediction[iIndex])
        {
            iIndex = i;
        }
    }
    return iIndex;
}
