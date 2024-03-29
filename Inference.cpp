#include <assert.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "Inference.h"


/// class declarations
struct TrtDestructor {
  template <class T>
  void operator()(T* obj) const {
    if (obj) obj->destroy();
  }
};

template <class T>
// wrap with std::unique_ptr to call `destroy()` while destructing
using TrtUniquePtr = std::unique_ptr<T, TrtDestructor>;

typedef struct {
  float scale_factor_;
  cv::Scalar norm_mean_;
  cv::Scalar norm_std_;
} Transform;

/**
 * @brief   predictor class for patches
 */
class InferencerImpl {
 public:
  InferencerImpl(const std::string& engine_path);
  InferencerImpl(const std::string& model_path, int max_batch_size);
  InferencerImpl(const InferencerImpl&) = delete;
  Inferencer& operator=(const Inferencer&) = delete;
  virtual ~InferencerImpl();
  void DoInference(std::vector<cv::Mat>& images,
                   std::vector<std::vector<float>>& probs);
  void SerializeEngine(const std::string& model_path);
  void set_scale_factor(float factor);
  void set_norm_mean(cv::Scalar);
  void set_norm_std(cv::Scalar);
  size_t get_output_size() const;

 protected:
  virtual void ProcessOutput(std::vector<cv::Mat>& images,
                             std::vector<std::vector<float>>& probs,
                             std::vector<float>& logits) = 0;

  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  TrtUniquePtr<nvinfer1::IExecutionContext> context_;
  Transform transform_;
  int batch_size_;
  std::vector<nvinfer1::Dims> input_dims_;
  std::vector<nvinfer1::Dims> output_dims_;
  int out_dim_;  // 每个sample对应输出的size
  std::vector<size_t> binding_sizes_;

  std::vector<void*> buffers_;

 private:
  void Init();
};

class ClsInferencerImpl : public InferencerImpl {
 public:
  ClsInferencerImpl(const std::string& model_path, int max_batch_size);
  ClsInferencerImpl(const std::string& engine_path);

 private:
  void ProcessOutput(std::vector<cv::Mat>& images,
                     std::vector<std::vector<float>>& probs,
                     std::vector<float>& logits) override;
};

class SegInferencerImpl : public InferencerImpl {
 public:
  SegInferencerImpl(const std::string& engine_path);

 private:
  void ProcessOutput(std::vector<cv::Mat>& images,
                     std::vector<std::vector<float>>& probs,
                     std::vector<float>& logits) override;

  nvinfer1::Dims output_dim_;
};


/**
 * @ brief  calculate size of tensor
 *
 * @ param dims     dimension of tensor
 * @ return size_t  total size of tensor
 */
static size_t get_size_by_dim(const nvinfer1::Dims& dims) {
  size_t size = 1;
  for (size_t i = 0; i < dims.nbDims; ++i) size *= dims.d[i];
  return size;
}

/**
 * @ brief  preprocess image (resize, normalize and upload to gpu)
 *
 * @ param image        image to be processed
 * @ param gpu_input    pointer to allocated space for network input on gpu
 * @ param dims         dimensions of network inputs
 * @ param transform    resize and normalization factors
 */
static void preprocess_image(const cv::Mat& image, float* gpu_input,
                             const nvinfer1::Dims& dims,
                             const Transform& transform,
                             cudaStream_t& stream) {
  // read input image
  size_t nbdims = dims.nbDims;
  auto input_width = dims.d[nbdims - 1];
  auto input_height = dims.d[nbdims - 2];
  auto channels = dims.d[nbdims - 3];
  auto input_size = cv::Size(input_width, input_height);

  // resize
  cv::Mat resized;
  cv::resize(image, resized, input_size, 0, 0, cv::INTER_LINEAR);

  // normalize
  cv::Mat flt_image;
  resized.convertTo(flt_image, CV_32FC3, transform.scale_factor_);
  cv::subtract(flt_image, transform.norm_mean_, flt_image);
  cv::divide(flt_image, transform.norm_std_, flt_image);

  // convert to tensor and upload to gpu
  int image_size = channels * input_width * input_height;
  float* cpu_input = (float*)malloc(image_size * sizeof(float));
  std::vector<cv::Mat> image_channels;
  for (size_t i = 0; i < channels; ++i) {
    image_channels.emplace_back(cv::Mat(
        input_size, CV_32FC1, cpu_input + i * input_width * input_height));
  }
  cv::split(flt_image, image_channels);
  cudaMemcpyAsync(gpu_input, cpu_input, image_size * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  free(cpu_input);
  cpu_input = nullptr;
}

/**
 * @brief   postprocess network output results
 *
 * @param gpu_output    pointer to allocated space for network output on gpu
 * @param dims          dimensions of network outputs
 * @param batch_size    input batch_size
 * @return std::vector<float>   outputs logits for all classes
 */
static std::vector<float> postprocess_results(float* gpu_output,
                                              const nvinfer1::Dims& dims,
                                              int batch_size,
                                              cudaStream_t& stream) {
  // copy results from GPU to CPU
  std::vector<float> cpu_output(get_size_by_dim(dims));
  cudaMemcpyAsync(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  return cpu_output;
}

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if ((severity == Severity::kERROR) ||
        (severity == Severity::kINTERNAL_ERROR)) {
      std::cerr << msg << "\n";
    }
  }
} gLogger;

/**
 * @brief   parse onnx model and convert to TensorRT engine
 *
 * @param model_path    path to onnx model file
 * @param engine        pointer to TensorRT `nvinfer1::ICudaEngine` object
 * @param context       pointer to execution context
 * @param maxBatchSize  max batch size of inputs
 */
static void parse_onnx_model(const std::string& model_path,
                             TrtUniquePtr<nvinfer1::ICudaEngine>& engine,
                             TrtUniquePtr<nvinfer1::IExecutionContext>& context,
                             int max_batch_size) {
  TrtUniquePtr<nvinfer1::IBuilder> builder{
      nvinfer1::createInferBuilder(gLogger)};
  const auto explicitBatch =
      1 << static_cast<int>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TrtUniquePtr<nvinfer1::INetworkDefinition> network{
      builder->createNetworkV2(explicitBatch)};
  TrtUniquePtr<nvonnxparser::IParser> parser{
      nvonnxparser::createParser(*network, gLogger)};
  TrtUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
  // disable cublas_lt to avoid unknown error for full connected layers in some
  // cases
  config->setTacticSources(
      1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS));
  // parse ONNX
  if (!parser->parseFromFile(
          model_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
    std::cerr << "ERROR: could not parse the model.\n";
    return;
  }
  // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
  config->setMaxWorkspaceSize(1ULL << 30);
  // use FP16 mode if possible
  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  builder->setMaxBatchSize(max_batch_size);
  // generate TensorRT engine optimized for the target platform
  engine.reset(builder->buildEngineWithConfig(*network, *config));
  context.reset(engine->createExecutionContext());
}

/**
 * @brief   parse onnx model and convert to TensorRT engine
 *
 * @param model_path    path to serialized trt engine file
 * @param engine        pointer to TensorRT `nvinfer1::ICudaEngine` object
 * @param context       pointer to execution context
 * @param maxBatchSize  max batch size of inputs
 */
static void parse_trt_engine(const std::string& model_path,
                             TrtUniquePtr<nvinfer1::ICudaEngine>& engine,
                             TrtUniquePtr<nvinfer1::IExecutionContext>& context)
{
  std::ifstream engine_file(model_path, std::ios::binary);
  if (!engine_file.is_open()) {
    std::cerr << "Engine file [" << model_path << "] does not exist!"
              << std::endl;
    exit(-1);
  }
  std::stringstream ss;
  ss << engine_file.rdbuf();
  std::string plan = ss.str();
  TrtUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(gLogger)
  };
  // generate TensorRT engine optimized for the target platform
  engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
  std::cout << "maxBatchsize = " << engine->getMaxBatchSize() << std::endl;
  context.reset(engine->createExecutionContext());
}

InferencerImpl::InferencerImpl(const std::string& engine_path) {
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>{nullptr};
  context_ = TrtUniquePtr<nvinfer1::IExecutionContext>{nullptr};
  std::cout << "Parsing engine file: " << engine_path << std::endl;
  parse_trt_engine(engine_path, engine_, context_);
  std::cout << "Parsed engine file: " << engine_path << std::endl;
  Init();
}

InferencerImpl::InferencerImpl(const std::string& model_path, int max_batch_size) {
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>{nullptr};
  context_ = TrtUniquePtr<nvinfer1::IExecutionContext>{nullptr};
  std::cout << "Parsing onnx model: " << model_path << std::endl;
  parse_onnx_model(model_path, engine_, context_, max_batch_size);
  std::cout << "Parsed onnx model: " << model_path << std::endl;
  Init();
}

InferencerImpl::~InferencerImpl() {
  for (size_t i = 0; i < engine_->getNbBindings(); ++i) cudaFree(buffers_[i]);
}

void InferencerImpl::Init() {
  // init configs
  transform_ = Transform{1.f / 255.f, cv::Scalar(0.485f, 0.456f, 0.406f),
                         cv::Scalar(0.229f, 0.224f, 0.225f)};

  std::vector<void*> buffers(
      engine_->getNbBindings());  // buffers for input and output data

  for (size_t i = 0; i < engine_->getNbBindings(); ++i) {
    auto dim = engine_->getBindingDimensions(i);
    auto binding_size = get_size_by_dim(dim) * sizeof(float);
    binding_sizes_.emplace_back(binding_size);
    if (engine_->bindingIsInput(i)) {
      input_dims_.emplace_back(dim);
    } else {
      batch_size_ = dim.d[0];
      out_dim_ = 1;
      for (size_t k = 1; k < dim.nbDims; ++k) out_dim_ *= dim.d[k];
      output_dims_.emplace_back(dim);
    }
  }
  if (input_dims_.empty() || output_dims_.empty()) {
    std::cerr << "Expect at least one input and one output for network"
              << std::endl;
    exit(-1);
  }

  buffers_.assign(engine_->getNbBindings(),
                  nullptr);  // buffers for input and output data
  for (size_t i = 0; i < engine_->getNbBindings(); ++i) {
    cudaMalloc(&buffers_[i], binding_sizes_[i]);
  }

  std::cout << "Inferencer initialized." << std::endl;
}

void Inferencer::SerializeEngine(const std::string& model_path) {
  impl_->SerializeEngine(model_path);
}

void InferencerImpl::SerializeEngine(const std::string& model_path) {
  nvinfer1::IHostMemory* serialized_model = engine_->serialize();
  std::ofstream engine_file(model_path, std::ios::out | std::ios::binary);
  engine_file.write((char*)(serialized_model->data()), serialized_model->size());
  engine_file.close();
}

void InferencerImpl::set_scale_factor(float factor) {
  transform_.scale_factor_ = factor;
}

void InferencerImpl::set_norm_mean(cv::Scalar mean) {
  transform_.norm_mean_ = std::move(mean);
}

void InferencerImpl::set_norm_std(cv::Scalar std) {
  transform_.norm_std_ = std::move(std);
}

size_t InferencerImpl::get_output_size() const {
  return get_size_by_dim(output_dims_[0]) * batch_size_;
}

void Inferencer::DoInference(std::vector<cv::Mat>& images,
                             std::vector<std::vector<float>>& probs) {
  impl_->DoInference(images, probs);
}

void InferencerImpl::DoInference(std::vector<cv::Mat>& images,
                             std::vector<std::vector<float>>& probs) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // preprocess input data
  for (size_t i = 0; i < images.size(); ++i) {
    preprocess_image(
        images[i],
        reinterpret_cast<float*>((char*)buffers_[0] +
                                 binding_sizes_[0] / batch_size_ * i),
        input_dims_[0], transform_, stream);
  }
  // inference
  context_->enqueue(images.size(), buffers_.data(), stream, nullptr);
  // postprocess results
  std::vector<float> logits = postprocess_results(
      (float*)buffers_[1], output_dims_[0],
                             images.size(), stream);
  cudaStreamSynchronize(stream);

  ProcessOutput(images, probs, logits);
}

ClsInferencer::ClsInferencer(const std::string& model_path,
                             int max_batch_size) {
  impl_ = new ClsInferencerImpl(model_path, max_batch_size);
}

ClsInferencerImpl::ClsInferencerImpl(const std::string& model_path, int max_batch_size)
    : InferencerImpl(model_path, max_batch_size) {}

ClsInferencer::ClsInferencer(const std::string& engine_path) {
  impl_ = new ClsInferencerImpl(engine_path);
}

ClsInferencerImpl::ClsInferencerImpl(const std::string& engine_path)
    : InferencerImpl(engine_path) {}

void ClsInferencerImpl::ProcessOutput(std::vector<cv::Mat>& images,
                                  std::vector<std::vector<float>>& probs,
                                  std::vector<float>& logits) {
  probs.clear();
  for (size_t i = 0; i < images.size(); ++i) {
    // calculate softmax
    std::vector<float> results(logits.begin() + i * out_dim_,
                               logits.begin() + (i + 1) * out_dim_);
    std::transform(results.begin(), results.end(), results.begin(),
                   [](float val) { return std::exp(val); });
    auto sum = std::accumulate(results.begin(), results.end(), 0.0);
    std::transform(results.begin(), results.end(), results.begin(),
                   [sum](float val) { return val / sum; });
    probs.emplace_back(results);
  }
}

SegInferencer::SegInferencer(const std::string& engine_path) {
  impl_ = new SegInferencerImpl(engine_path);
}

SegInferencerImpl::SegInferencerImpl(const std::string& engine_path)
    : InferencerImpl(engine_path) {
  output_dim_ = output_dims_[0];
}

void SegInferencerImpl::ProcessOutput(
  std::vector<cv::Mat>& images,
  std::vector<std::vector<float>>& probs,
  std::vector<float>& logits)
{
  std::vector<cv::Mat> masks;

  for (size_t i = 0; i < images.size(); ++i) {
    std::vector<float> results(logits.begin() + i * out_dim_,
                               logits.begin() + (i + 1) * out_dim_);
    cv::Mat mask = cv::Mat(output_dim_.d[output_dim_.nbDims - 2],
                           output_dim_.d[output_dim_.nbDims - 1], CV_32F,
                           logits.data() + i * out_dim_);
    masks.emplace_back(mask.clone());
  }
  images.swap(masks);
}
