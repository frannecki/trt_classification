#ifndef INFERENCE_H
#define INFERENCE_H

#define TRT_RESNET_API __declspec(dllexport)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

class InferencerImpl;

/**
 * @brief   predictor class for patches
 */
class TRT_RESNET_API Inferencer {
 public:
  Inferencer() = default;
  Inferencer(const Inferencer&) = delete;
  Inferencer& operator=(const Inferencer&) = delete;
  virtual ~Inferencer() = default;
  void DoInference(std::vector<cv::Mat>& images,
                   std::vector<std::vector<float>>& probs);
  void SerializeEngine(const std::string& model_path);

 protected:
  InferencerImpl* impl_{nullptr};
};

class TRT_RESNET_API ClsInferencer : public Inferencer {
 public:
  ClsInferencer(const std::string& model_path, int max_batch_size);
  ClsInferencer(const std::string& engine_path);
};

/**
 * @brief   segmentor class for thumbnails
 */
class TRT_RESNET_API SegInferencer : public Inferencer {
 public:
  SegInferencer(const std::string& engine_path);
};

#endif
