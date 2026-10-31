CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

project(ModelDeployTensorRT CXX)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-g)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(ModelDeployTensorRT 
    ModelDeployTensorRT.cpp
    ModelDeployHelper.cpp
    main.cpp
)

target_link_libraries(ModelDeployTensorRT 
${OpenCV_LIBS} 
${CUDA_LIBRARIES}
nvinfer 
nvonnxparser
)
