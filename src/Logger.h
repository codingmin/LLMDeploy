#ifndef LLMDEPLOY_LOGGER_H_
#define LLMDEPLOY_LOGGER_H_

#include <iostream>

#include "NvInferRuntime.h"

class MyLogger : public nvinfer1::ILogger {
 public:
  virtual void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[MyLogger] " << msg << std::endl;
    }
  }
};

#endif
