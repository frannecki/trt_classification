#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Inference.h"

int main(int argc, char** argv) {
  cv::Mat image = cv::imread("D:/Documents/python/bird.jpg");
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  std::vector<cv::Mat> images;
  images.push_back(image);
  images.push_back(image);
  Inferencer* inferencer = new ClsInferencer("D:/Documents/python/resnet34.engine.1");
  std::vector<std::vector<float>> probs;
  inferencer->DoInference(images, probs);
  for (auto &prob : probs) {
    for (float val : prob) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
