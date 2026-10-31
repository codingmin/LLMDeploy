#ifndef _MODEL_DEPLOY_TENSORRT_H_
#define _MODEL_DEPLOY_TENSORRT_H_

#include "IModelDeploy.h"
#include "Logger.h"
#include <memory>

class  ModelDeployTensorRT:public IModelDeploy
{
public:
	ModelDeployTensorRT():m_logger(std::make_shared<MyLogger>())
	{

	}
	
	~ModelDeployTensorRT()=default;
	
	ModelDeployTensorRT(const ModelDeployTensorRT& other)=delete;
	ModelDeployTensorRT& operator=(const ModelDeployTensorRT& other)=delete;
	
	ModelDeployTensorRT(const ModelDeployTensorRT&& other)=delete;	
	ModelDeployTensorRT& operator=(const ModelDeployTensorRT&& other)=delete;
	
	virtual void LoadModel(const std::string& strModelPath) override;
	
	virtual std::vector<float> Inference(const std::vector<float>& vecFloatInputImage,const nvinfer1::Dims& dim) override;

private:
	std::shared_ptr<nvinfer1::ILogger> m_logger;
	std::vector<char> m_vecEngineFile;
	bool bInferenceEngineBuilt = false;
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
};	

#endif // _MODEL_DEPLOY_TENSORRT_H_