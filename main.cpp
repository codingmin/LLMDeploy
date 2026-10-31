#include"IModelDeploy.h"
#include"LabelString.h"
#include"ModelDeployHelper.h"
#include"ModelDeployTensorRT.h"
#include<string>
#include<memory>


void OutputResult(const std::vector<float>& vecFloatResultResNet)
{
    const int iIndex = ModelDeployHelper::GetMaxValueIndex(vecFloatResultResNet);
    std::cout << "Predicted class index: " << iIndex;
    std::cout << ";predicted class name: " << imagenet_zh_labels[iIndex] << std::endl;
}

void TestLocalImage(const std::string&ImagePath,const std::shared_ptr<IModelDeploy>& pModelDeploy)
{
    const std::string strImagePath = "fishOne.jpg";
    std::vector<float> vecFloatPreprocessedImage = ModelDeployHelper::PreprocessImage(strImagePath,224,224);
    std::vector<float> vecFloatResultResNet = pModelDeploy->Inference(vecFloatPreprocessedImage, nvinfer1::Dims{4, {1, 3, 224, 224}});
    std::cout << ImagePath << ": ";
    OutputResult(vecFloatResultResNet);
}

void TestVideoDevice(const std::string& strVideoDevicePath,const std::shared_ptr<IModelDeploy>& pModelDeploy)
{
    cv::Mat img = ModelDeployHelper::ReadImageFromVideoDevice(strVideoDevicePath);
    imwrite("captured_frame.jpg", img);
    std::vector<float> vecFloatPreprocessedImage = ModelDeployHelper::PreprocessImage(img,224,224);
    std::vector<float> vecFloatResultResNet = pModelDeploy->Inference(vecFloatPreprocessedImage, nvinfer1::Dims{4, {1, 3, 224, 224}});
    std::cout << strVideoDevicePath << ": ";
    OutputResult(vecFloatResultResNet);
}

int main(int argc,char* argv[])
{
    if(argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <path_to_onnx_model_file>" << std::endl;
        return 1;
    }
    std::string strModelPath = argv[1];
    std::cout << "Loading model from: " << strModelPath << std::endl;
    
    std::shared_ptr<IModelDeploy> pModelDeploy = std::make_shared<ModelDeployTensorRT>();
    pModelDeploy->LoadModel(strModelPath);

    const std::string strImagePath = "fishOne.jpg";
    TestLocalImage(strImagePath,pModelDeploy);

    const std::string strDevicePath = "/dev/video0";
    TestVideoDevice(strDevicePath,pModelDeploy);

    return 0;
}