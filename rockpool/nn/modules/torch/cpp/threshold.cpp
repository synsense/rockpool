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
    auto membranePotential = saved[0];
    double threshold = ctx->saved_data["threshold"].toDouble();
    double window = ctx->saved_data["window"].toDouble();

    auto vmem_shifted = membranePotential - threshold / 2;
    auto vmem_periodic = vmem_shifted % threshold;
    auto vmem_below = vmem_shifted * (membranePotential < threshold);
    auto vmem_above = vmem_periodic * (membranePotential >= threshold);
    auto vmem_new = vmem_above + vmem_below;
    auto spikePdf = torch::exp(-torch::abs(vmem_new - threshold / 2) / window) / threshold;

    return {grad_outputs[0] * spikePdf, torch::Tensor(), torch::Tensor()};
  }
};

