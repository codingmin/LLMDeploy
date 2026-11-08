CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

project(ModelDeployTensorRT CXX)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-g)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(ModelDeployTensorRT
    src/main.cpp
    src/IModelDeploy.h
    src/ModelDeployTensorRT.cpp
    src/ModelDeployTensorRT.h
    src/ModelDeployHelper.cpp
    src/ModelDeployHelper.h
    src/Logger.h
    src/LabelString.h
)

# 测试图片拷贝到 build 目录，保持相对路径可运行
file(COPY ${CMAKE_SOURCE_DIR}/assets/fishOne.jpg DESTINATION ${CMAKE_BINARY_DIR})

target_link_libraries(ModelDeployTensorRT 
${OpenCV_LIBS} 
${CUDA_LIBRARIES}
nvinfer 
nvonnxparser
)
