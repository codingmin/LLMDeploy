# 简介

通过TensorRT C++ API部署ResNet50模型，可通过本地图片和本地视频设备输入图片。

# 快速运行

1、环境及相关依赖

- Ubuntu 22.04.5 LTS

- Cuda Driver 580.178.04
- CUDA Toolkit 12.8
- TensorRT 10.16.1.11-1
- OpenCV 4.5.4
- CMake 3.22.1

2、下载代码

```git
gitgit clone https://github.com/codingmin/LLMDeploy.git
```

3、huggingface上下载模型

```git
git clone https://huggingface.co/onnx-community/resnet-50-ONNX
```

4、进入代码目录之后，**修改main函数中代码为本地相关路径或者设备**，编译和运行

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



