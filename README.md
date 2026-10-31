# 简介

通过TensorRT C++ API部署RestNet50模型，可以通过本地图片和本地视频设备，输入图片进行分类输出结果。

# 快速运行

1、安装依赖相关依赖

- Cuda：Driver，CUDAToolkit，
- TensorRT
- OpenCV
- CMake

2、下载代码

``` git clone https://github.com/codingmin/LLMDeploy.git
```



3、从huggingface上下载模型

```git
git clone https://huggingface.co/onnx-community/resnet-50-ONNX
```



4、进入代码目录之后，修改main函数中代码，之后编译和运行

```bash
mkdir build 
cd build
cmake ..
make
```

运行进行推理，正常输出如下：

```bash
Loading model from: model.onnx
fishOne.jpg: Predicted class index: 392;predicted class name:  rock beauty, Holocanthus tricolor
/dev/video0: Predicted class index: 593;predicted class name:  harmonica, mouth organ, harp
```





# 参考

https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html



