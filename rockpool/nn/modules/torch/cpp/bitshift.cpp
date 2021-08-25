#include <torch/extension.h>
#include <iostream>

using namespace torch::autograd;

class BitshiftDecay: public Function<BitshiftDecay> {

 public:
  static torch::Tensor forward(AutogradContext *ctx, 
                               torch::Tensor data, 
                               torch::Tensor dash,
                               torch::Tensor tau) {

    int scale = 10000;
    torch::Tensor data_ = data * scale;
    // right shift is difficult to use here
    // as it only works with scalars
    // torch::Tensor dv = (data_.__rshift__(dash));
    // simulate right shift by division with 2 ** dash
    torch::Tensor dv = torch::floor(data_ / dash);
    dv = torch::floor(torch::where(dv == 0, data_.sign(), dv));
    torch::Tensor v = (data_ - dv) / scale;

    ctx->save_for_backward({data, v, tau});
    return v;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto data = saved[0];
    auto v = saved[1];
    auto tau = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output * tau;
    //grad_input = torch::where(torch::isnan(grad_input), torch::zeros_like(grad_output), grad_input);

    return {grad_input, torch::Tensor(), torch::Tensor()};
  }
};

