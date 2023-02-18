#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Inference.h"

int main(int argc, char** argv) {
  
  std::vector<std::string> image_paths = {"husky.jpg", "siam_cat.jfif",
                                          "elephant.jpg", "kangaroo.jfif"};
  std::vector<cv::Mat> images;  
  for (auto& filename : image_paths) {
    cv::Mat image = cv::imread("data/" + filename);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    images.emplace_back(image);
  }

  // Inferencer* inferencer = new ClsInferencer("data/resnet34.onnx", 32);
  // inferencer->SerializeEngine("data/resnet34.engine");
  Inferencer* inferencer = new ClsInferencer("data/resnet34.engine");

  std::vector<std::vector<float>> probs;

  auto current = clock();
  inferencer->DoInference(images, probs);
  std::cout << "Time elapsed inferencing (in ms): " << (clock() - current) / 1000.0 << std::endl;

  // probs: N * C
  for (auto &prob : probs) {
    float max_prob = 0.;
    int argmax = 0;
    for (size_t i = 0; i < prob.size(); ++i)
      if (max_prob < prob[i]) {
        max_prob = prob[i];
        argmax = i;
      }
    std::cout << "Image is of class " << argmax << std::endl;
  }
  delete inferencer;
  return 0;
}
