#ifndef INFERENCER_H
#define INFERENCER_H

#include <NvInfer.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

struct TrtDestructor {
    template <class T>
    void operator()(T* obj) const {
        if (obj) obj->destroy();
    }
};

template <class T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestructor>;

typedef struct {
    float scale_factor_;
    cv::Scalar norm_mean_;
    cv::Scalar norm_std_;
} Transform;


/**
 * @brief   predictor class for single cell patch classification
 */
class Inferencer {
public:
    Inferencer(const std::string& model_path, int max_batch_size);
    Inferencer(const Inferencer&) = delete;
    Inferencer& operator=(const Inferencer&) = delete;
    virtual ~Inferencer();
    virtual void Inference(const std::vector<cv::Mat>& images,
                           std::vector<std::vector<float>>& probs) = 0;
    void set_scale_factor(float factor);
    void set_norm_mean(cv::Scalar);
    void set_norm_std(cv::Scalar);
    size_t get_output_size() const;

protected:
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;
    Transform transform_;
    int batch_size_;
    std::vector<nvinfer1::Dims> input_dims_;
    std::vector<nvinfer1::Dims> output_dims_;
    int out_dim_;
    std::vector<size_t> binding_sizes_;

    std::vector<void*> buffers_;
};

class ClsInferencer : public Inferencer {
public:
    ClsInferencer(const std::string& model_path, int max_batch_size);
    void Inference(const std::vector<cv::Mat>& images,
                   std::vector<std::vector<float>>& probs) override;

private:
    virtual std::vector<float> Predict(const std::vector<cv::Mat>& images);
};

#endif
