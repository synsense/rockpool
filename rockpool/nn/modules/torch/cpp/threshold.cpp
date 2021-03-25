#include <torch/extension.h>
#include <iostream>

using namespace torch::autograd;

class ThresholdSubtract: public Function<ThresholdSubtract> {

 public:
  static torch::Tensor forward(AutogradContext *ctx, 
                               torch::Tensor data, 
                               double threshold, 
                               double window) {

    ctx->save_for_backward({data});
    ctx->saved_data["threshold"] = threshold;
    ctx->saved_data["window"] = window;

    data = (data > 0) * torch::floor(data / threshold);
    return data;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto data = saved[0];
    double threshold = ctx->saved_data["threshold"].toDouble();
    double window = ctx->saved_data["window"].toDouble();

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output * (data >= (threshold - window)) / threshold;

    return {grad_input, torch::Tensor(), torch::Tensor()};
  }
};

