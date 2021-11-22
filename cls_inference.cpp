// cls_inference.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ClsInferencer.h"

int main(int argc, char** argv) {
    Inferencer* inferencer = new ClsInferencer("D:/documents/pyproj/pyLib/hqm/model.onnx", 256);
    
    std::vector<cv::Mat> images;
    cv::Mat image = cv::imread("D:/documents/pyproj/data/0/43_HE2IHC_0_25856_185088_1_128.png");
    for (size_t i = 0; i < 256; ++i)
        images.push_back(image);

    std::vector<std::vector<float>> probs;
    inferencer->Inference(images, probs);

    std::cout << "probs.size() = " << probs.size() << std::endl;

    for (size_t i = 0; i < probs.size(); ++i) {
        for (size_t j = 0; j < probs[i].size(); ++j) {
            std::cout << std::setprecision(4) << probs[i][j] << " ";
        }
        std::cout << std::endl;
    }

    delete inferencer;
    return 0;
}
