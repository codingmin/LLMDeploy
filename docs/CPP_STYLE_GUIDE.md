# C++ 编码规范（初版）

> 参考：[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)、[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
>
> 适用范围：本项目全部 C++/CUDA 代码。规范基于 Google 风格裁剪，结合本项目（TensorRT 推理部署）的实际做了明确偏离，偏离项见第 1 节。
>
> 状态：初版（v0.1），欢迎评审提意见；对规范本身的修改需先改本文件再改代码。

## 1. 与 Google 规范的偏离项（本项目明确不采纳的部分）

| # | Google 规范 | 本项目约定 | 原因 |
|---|---|---|---|
| D1 | 禁止异常 | **允许异常**，不可恢复错误用 `throw` | 存量代码已大量使用 `std::runtime_error`，全面改返回码代价过高 |
| D2 | 行宽 80 列 | **行宽上限 100 列** | 现代显示器与 CUDA 长签名下 80 过于局促 |
| D3 | 成员变量尾下划线 `name_` | **`m_` 前缀**（如 `m_engine`） | 与存量代码一致，且一眼区分成员/局部 |
| D4 | 文件名 snake_case（`my_class.cc`） | **文件名与主类同名 PascalCase**（`ModelDeployTensorRT.cpp`） | 沿用现状，文件即类 |
| D5 | 局部变量 snake_case，禁匈牙利前缀 | **新代码 snake_case，不使用匈牙利前缀**；存量文件保持原样、触碰到时渐进迁移 | 避免无意义的大面积 diff |
| D6 | 缩进 2 空格 | **新代码 2 空格、禁止 Tab**；存量 tab 文件迁移期保持原样 | 同 D5 |

除上表外，一律遵循 Google 规范。

## 2. 总原则

1. **可读性第一**：代码首先给人读，其次给编译器。禁止炫技式写法。
2. **一致性高于个人偏好**：与存量文件风格冲突时，改该文件内的代码遵循存量风格；新文件遵循本规范。
3. **RAII 管理一切资源**：内存、CUDA 资源、句柄一律用对象生命周期管理，禁止裸 `new`/`delete`/`malloc`/`free` 管理所有权。
4. **先测量后优化**：性能相关修改必须附 profile 数据（nsys/ncu），禁止凭感觉优化；优化后必须复测验证。

## 3. 命名规范

| 元素 | 规则 | 正例 | 反例 |
|---|---|---|---|
| 类型/类/结构体/枚举 | PascalCase | `ModelDeployTensorRT`, `GpuPreprocessor` | `model_deploy` |
| 函数/方法 | PascalCase | `LoadModel()`, `Inference()`, `GetMaxValueIndex()` | `load_model()` |
| 局部变量/参数 | snake_case，不用匈牙利前缀 | `input_dims`, `output_count` | `strModelPath`, `vecFloatOutput` |
| 成员变量 | `m_` + snake_case（偏离项 D3） | `m_engine`, `m_input_count` | `engine_`, `pEngine` |
| 常量（含 enum 值） | `k` + PascalCase | `kMaxBatchSize`, `kInvalid` | `MAX_BATCH_SIZE` |
| 宏 | 尽量不用；必须时全大写+项目前缀 | `LLMDEPLOY_CUDA_CHECK` | `CHECK`（易冲突） |
| 命名空间 | snake_case | `namespace deploy {}` | `namespace Deploy {}` |
| 布尔量 | 语义化 is/has/can 前缀 | `is_initialized`, `has_cached_engine` | `flag1`, `bResult` |

```cpp
// ✅ 新代码
size_t output_count = 1;
for (int32_t i = 0; i < dims.nbDims; ++i) {
  output_count *= dims.d[i];
}

// ❌ 新代码禁止（匈牙利前缀 + 驼峰）
size_t szOutputCount = 1;
```

## 4. 头文件

### 4.1 #define 保护宏

格式 `<项目>_<文件名大写>_H_`，**禁止下划线开头**（`_XXX_H_` 中的前导下划线是保留标识符）：

```cpp
#ifndef LLMDEPLOY_MODEL_DEPLOY_TENSORRT_H_   // ✅
#define LLMDEPLOY_MODEL_DEPLOY_TENSORRT_H_
...
#endif

#ifndef _MODEL_DEPLOY_TENSORRT_H_            // ❌ 前导下划线是保留的
```

### 4.2 include 顺序

本文件对应的头文件 → C 系统头 → C++ 标准库 → 其他库（OpenCV/TensorRT/CUDA）→ 本项目头，各组间空行：

```cpp
#include "ModelDeployTensorRT.h"   // ① 对应头文件（验证自包含）
#include <cuda_runtime.h>          // ② C 系统
#include <memory>
#include <string>
#include <NvInfer.h>               // ③ 第三方库
#include <opencv2/opencv.hpp>
#include "GpuPreprocess.h"         // ④ 本项目
```

### 4.3 其他规则

- 头文件必须自包含（单独编译它应通过）。
- 禁止在**头文件**中 `using namespace`；.cpp 中尽量避免，函数内 using 可接受。
- 内联函数只在小型（≤10 行）且热点场景使用。
- 前向声明可用但不要滥用（能用 include 就 include，除非明确为了减编译依赖）。

## 5. 类与结构体

```cpp
class ModelDeployTensorRT : public IModelDeploy {
public:
  ModelDeployTensorRT();
  ~ModelDeployTensorRT();                      // 有资源释放就不能 =default

  // 不需要拷贝的类显式删除（存量做法，保持）
  ModelDeployTensorRT(const ModelDeployTensorRT&) = delete;
  ModelDeployTensorRT& operator=(const ModelDeployTensorRT&) = delete;

  virtual void LoadModel(const std::string& model_path) override;

private:
  bool InitInferenceResources();               // 私有方法在前
  // ... 成员函数 ...

  std::shared_ptr<ILogger> m_logger;           // 数据成员最后
};
```

- 单参数构造函数必须 `explicit`。
- 类内声明顺序：`public` → `protected` → `private`；每组内：类型别名 → 常量 → 构造/析构 → 方法 → 数据成员。
- 初始化列表顺序必须与成员声明顺序一致（编译器 `-Wreorder` 会查）。
- 需要析构/拷贝/移动中任意一个时，五者全部显式定义（或 delete）。
- 继承默认 `public`，虚函数一律 `override` 关键字标注。

## 6. 函数

- 参数顺序：**输入在前，输出在后**。
- 大对象传 `const&`；小对象（≤16 字节或标准容器视图）按值传；需要修改的输出用指针或返回值。
- **优先返回值而非输出参数**（C++17 RVO/move 语义下返回更高效且清晰）：

```cpp
// ✅ 返回值
std::vector<float> Inference(const cv::Mat& img, int target_h, int target_w);

// ❌ 输出参数（除非高频热路径实测证明必要）
void Inference(const cv::Mat& img, std::vector<float>* output);
```

- 单个函数不超过 ~60 行；嵌套不超过 4 层，超过就拆分。
- 虚函数禁止默认参数。
- 谨慎使用函数重载，避免仅参数为引用/值之别的重载。

## 7. 现代 C++（C++17）使用规则

### 7.1 智能指针

```cpp
// ✅ 默认 unique_ptr；确需共享所有权才 shared_ptr；用 make_ 系列
auto context = std::make_unique<IExecutionContext>(...);
m_engine = std::shared_ptr<ICudaEngine>(...);

// ❌ 禁止
ICudaEngine* p = new ICudaEngine(...);   // 裸指针持有所有权
std::auto_ptr<T> p;                       // 已删除的类型
```

- 裸指针只表示"不拥有"（观察者/出参），禁止 `delete` 别人给你的指针。
- 函数间传递所有权用 `unique_ptr` 按值移动。

### 7.2 其他

- `auto`：类型显然（迭代器、`make_unique` 结果）时用；接口边界和数值类型慎用（`auto x = foo() / 2` 掩盖类型错误）。
- `const`/`constexpr` 能加则加；编译期常量用 `constexpr` 不用 `#define`。
- 遍历用 range-for；需要下标参与逻辑时才用下标循环。
- 强制转换只用 `static_cast`/`reinterpret_cast`（尽量不用）/`dynamic_cast`（仅确需 RTTI 时），**禁止 C 风格 `(T)x`**。

## 8. 错误处理（偏离 Google：允许异常）

- **不可恢复/违反契约**：抛异常，消息带上下文（函数名、期望值、实际值）：

```cpp
if (input_count != m_inputCount) {
  throw std::runtime_error("input size mismatch: expect " + std::to_string(m_inputCount));
}
```

- **可恢复/可预期的失败**（如用户传错路径）：也可抛异常，由调用方决定如何处理——本项目统一以异常为主。
- 析构函数、`noexcept` 函数内**禁止抛异常**。
- 不吞异常：catch 后必须处理、记录或重新抛出，禁止空 catch。

## 9. CUDA / TensorRT 专项规范（本项目核心）

### 9.1 CUDA 错误检查

每次 CUDA Runtime API 调用必须检查返回值，统一用检查宏：

```cpp
#define LLMDEPLOY_CUDA_CHECK(call)                                          \
  do {                                                                      \
    cudaError_t err__ = (call);                                             \
    if (err__ != cudaSuccess) {                                             \
      throw std::runtime_error(std::string(#call) + " failed: " +           \
                               cudaGetErrorString(err__));                  \
    }                                                                       \
  } while (0)

LLMDEPLOY_CUDA_CHECK(cudaMalloc(&m_d_input, bytes));
LLMDEPLOY_CUDA_CHECK(cudaStreamSynchronize(m_stream));
```

### 9.2 GPU 资源生命周期（血的教训，见 profile 分析）

- 设备内存、流、事件、ExecutionContext 等**初始化一次、全程复用**，或用 RAII 类封装。
- **禁止在推理热路径里** `cudaMalloc`/`cudaFree`/`cudaStreamCreate`/`cudaStreamDestroy`（隐式同步 + 毫秒级开销）。
- `cudaFreeHost`/`cudaFree` 同理，只在析构和确需扩容时调用。

```cpp
// ❌ 每次调用创建销毁
void Inference() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaMalloc(&d_in, bytes);
  ...
  cudaFree(d_in);
  cudaStreamDestroy(stream);
}

// ✅ 成员资源初始化一次
bool ModelDeployTensorRT::InitInferenceResources() { /* 一次性创建 */ }
```

### 9.3 主机-设备传输

- 高频路径一律 **pinned 内存 + `cudaMemcpyAsync`**；禁止高频同步 `cudaMemcpy`（它走 legacy 默认流，会隐式同步全部阻塞流——曾导致"4KB 拷贝耗时 85ms"）。
- 需要等待结果时用 `cudaStreamSynchronize` 显式表达，不要依赖隐式语义。

### 9.4 Kernel

- 命名：`<动作>_<含义>Kernel`，如 `PreprocessBgrKernel`。
- 入口参数加 `__restrict__`，边界检查必须存在。
- 优先融合（fuse）：一次数据加载完成 resize+格式转换+归一化（参考 `GpuPreprocess.cu`），减少中间缓冲。

### 9.5 TensorRT

- 引擎必须**落盘缓存**（`deserialize` 命中则跳过 build），缓存失效（TRT 版本/显卡/ONNX 变更）时自动重建覆盖。
- 优化 profile 的 shape 范围收窄到真实业务范围；禁止全维度 1→2048 这类范围（会触发 conv 回退 generic 核，慢约 50 倍，且 builder 无告警）。
- 所有 TRT 日志走统一的 `MyLogger`。

### 9.6 性能修改流程

1. `nsys profile` 获取基线 → 2. 单变量修改 → 3. 复测对比（同一指标）→ 4. 数据贴进 commit/文档。
- 看内核名判断异常：名字含 `generic`、sm 前缀比显卡老（sm50 跑在 sm86 卡上）、`_ffma_` 无 Tensor Core，都是回退/低效信号。

## 10. 注释

- 统一 `//` 风格；统一用英文。
- 注释解释**为什么**，不解释是什么（代码能看懂的不注释）。
- 每个类/非平凡函数说明：用途、参数约束、异常行为。
- 待办用 `TODO(名字): 描述`。
- 文件头 1~3 行说明职责即可，不要复制粘贴许可证模板（私有项目）。

## 11. 格式

- 缩进：新代码 **2 空格**，禁止 Tab（偏离项 D6）。
- 行宽 100 列。
- 所有 `if/for/while` **必须带大括号**（即使单行）——比 Google 略严，避免悬挂歧义：

```cpp
// ✅
if (img.empty()) {
  throw std::runtime_error("empty image");
}
// ❌
if (img.empty()) throw std::runtime_error("empty image");
```

- 函数定义：返回类型与函数名同行，`{` 换行置行首；控制语句 `{` 与语句同行；`else` 换行：
```cpp
bool InitInferenceResources()
{
  if (!m_context) {
    return false;
  }
  else {
    ...
  }
}
```
- switch 的每个 case 必须有 `break` 或显式 `FALLTHROUGH` 注释；default 必须存在。
- 指针/引用贴合类型还是变量不强求（`float* p` 或 `float *p`），**同一文件内保持一致**。

## 12. 工具链

- **clang-format**：仓库提供 `.clang-format`（Google 基础 + 行宽 100），新代码提交前格式化；对存量 tab 文件运行会产生大 diff，择机统一。
- **clang-tidy / cpplint**：建议后续接入 CI，初版不强制。
- 构建：CMake 目标名与主类一致；新源文件加进 `add_executable` 列表；CUDA 源文件（.cu）需 `project(... CXX CUDA)`。

## 13. Code Review 自检清单

- [ ] 资源（CUDA/TRT/内存）是否初始化一次复用，热路径有无 malloc/free/create/destroy？
- [ ] 每个 CUDA 调用是否检查了错误？
- [ ] 高频传输是否 pinned + async？同步语义是否显式？
- [ ] 异常消息是否带上下文？析构/noexcept 内是否无抛出？
- [ ] 命名是否符合第 3 节？头文件保护宏是否无前导下划线？
- [ ] 性能修改是否附了 profile 数据？
- [ ] 新代码是否 2 空格、100 列内、大括号齐全？

## 14. 参考

- Google C++ Style Guide: https://google.github.io/styleguide/cppguide.html
- C++ Core Guidelines: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
