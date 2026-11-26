#include "curvelets/udct.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

struct CurveletDenoiserImpl : public torch::nn::Module {
    explicit CurveletDenoiserImpl(const std::string& plan_path)
        : udct(register_module("udct", curvelets::UDCT(plan_path))),
          threshold(register_parameter(
              "threshold", torch::full({udct->ncoef()}, 0.1, torch::kFloat32))) {}

    torch::Tensor forward(const torch::Tensor& x) {
        const auto coeffs = udct->forward(x);
        const auto mag = coeffs.abs();
        const auto scale = torch::relu(mag - torch::relu(threshold)) / (mag + 1e-8);
        return udct->backward(coeffs * scale);
    }

    curvelets::UDCT udct{nullptr};
    torch::Tensor threshold;
};
TORCH_MODULE(CurveletDenoiser);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: train_model <udct_plan.pt>\n";
        return 1;
    }

    CurveletDenoiser model(argv[1]);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-2));

    const auto& shape = model->udct->shape();
    std::vector<int64_t> input_shape = {4};
    input_shape.insert(input_shape.end(), shape.begin(), shape.end());

    const torch::Tensor clean = torch::randn(input_shape, torch::kFloat32);
    const torch::Tensor noisy = clean + 0.25 * torch::randn_like(clean);

    for (int step = 0; step < 10; ++step) {
        optimizer.zero_grad();
        const torch::Tensor denoised = model->forward(noisy);
        const torch::Tensor loss = torch::mse_loss(denoised, clean);
        loss.backward();
        optimizer.step();
        std::cout << "step " << step << " loss=" << loss.item<double>() << "\n";
    }
    return 0;
}
