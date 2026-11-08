#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <iostream>

#include "NvInferRuntime.h"

class MyLogger : public nvinfer1::ILogger {
  virtual void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[MyLogger] ";
      std::cout << msg << std::endl;
    }
  }
};

#endif