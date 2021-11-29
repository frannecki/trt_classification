# TensorRT example for image classification / single-channel segmentation
## Prerequisites
* cuda
* cudnn
* TensorRT
* OpenCV

## Usage
An example main function is as follows (suppose `husky.jpg` and `resnet34.engine` is placed in `data` folder).

```cpp
int main(int argc, char** argv) {
  cv::Mat image = cv::imread("data/husky.jpg");
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  Inferencer* inferencer = new ClsInferencer("data/resnet34.engine");
  // -------------------------------------
  // // To use onnx model
  // Inferencer* inferencer = new ClsInferencer("data/resnet34.onnx", max_batch_size);
  // 
  // // To export TensorRT engine from onnx
  // inferencer->SerialzeModel("data/resnet34_dest.engine");
  // 
  // -------------------------------------
  std::vector<cv::Mat> images;
  for (size_t i = 0; i < 32; ++i) {
    images.push_back(image);
  }
  std::vector<std::vector<float>> probs;

  auto current = clock();
  inferencer->DoInference(images, probs);
  std::cout << "Time elapsed inferencing (in ms): " << (clock() - current) / 1000.0 << std::endl;

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
```
