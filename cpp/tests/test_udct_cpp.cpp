#include "curvelets/udct.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/torch.h>

namespace fs = std::filesystem;

namespace {

c10::Dict<c10::IValue, c10::IValue> load_dict(const fs::path& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(ifs.good(), "Cannot open ", path.string());
    const auto sz = ifs.tellg();
    std::vector<char> buf(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    ifs.read(buf.data(), sz);
    return torch::jit::pickle_load(buf).toGenericDict();
}

torch::Tensor get_t(const c10::Dict<c10::IValue, c10::IValue>& d, const char* key) {
    return d.find(c10::IValue(key))->value().toTensor();
}

void assert_close(
    const torch::Tensor& a,
    const torch::Tensor& b,
    double atol,
    double rtol,
    const std::string& label) {
    TORCH_CHECK(a.sizes() == b.sizes(), label, ": shape mismatch ", a.sizes(), " vs ", b.sizes());
    TORCH_CHECK(a.dtype() == b.dtype(), label, ": dtype mismatch ", a.dtype(), " vs ", b.dtype());
    if (!torch::allclose(a, b, rtol, atol)) {
        const double max_diff = (a - b).abs().max().item<double>();
        std::ostringstream oss;
        oss << label << ": allclose failed (max_diff=" << max_diff
            << ", atol=" << atol << ", rtol=" << rtol << ")";
        throw std::runtime_error(oss.str());
    }
}

void test_plan_parity(const fs::path& dir, const std::string& name, double atol, double rtol) {
    const fs::path plan_path = dir / (name + "_plan.pt");
    const fs::path ref_path = dir / (name + "_ref.pt");
    curvelets::UDCT udct(plan_path.string());
    const auto ref = load_dict(ref_path);

    const torch::Tensor x_ref = get_t(ref, "x");
    const torch::Tensor y_ref = get_t(ref, "y");
    const torch::Tensor recon_ref = get_t(ref, "recon");
    const torch::Tensor grad_ref = get_t(ref, "grad");

    torch::Tensor x = x_ref.clone().detach().requires_grad_(true);
    const torch::Tensor y = udct->forward(x);
    assert_close(y, y_ref, atol, rtol, name + ".forward");

    const torch::Tensor recon = udct->backward(y);
    assert_close(recon, recon_ref, atol, rtol, name + ".backward");

    const torch::Tensor loss = y.abs().pow(2).sum() + recon.abs().sum();
    loss.backward();
    TORCH_CHECK(x.grad().defined(), name, ": missing gradient");
    assert_close(x.grad(), grad_ref, atol, rtol, name + ".grad");
    std::cout << "[PASS] plan parity: " << name << std::endl;
}

void test_edge_cases(const fs::path& dir) {
    curvelets::UDCT real_udct((dir / "real_2d_f32_plan.pt").string());
    const torch::Tensor empty_x = torch::empty({0, 32, 32}, torch::kFloat32);
    const torch::Tensor empty_y = real_udct->forward(empty_x);
    TORCH_CHECK(empty_y.sizes() == c10::IntArrayRef({0, real_udct->ncoef()}));
    const torch::Tensor empty_recon = real_udct->backward(empty_y);
    TORCH_CHECK(empty_recon.sizes() == c10::IntArrayRef({0, 32, 32}));

    // Real input into complex transform and real coefficients into backward transform.
    curvelets::UDCT cplx_udct((dir / "complex_2d_f64_plan.pt").string());
    torch::manual_seed(3);
    const torch::Tensor xr = torch::randn({2, 24, 36}, torch::kFloat64);
    const torch::Tensor yc = cplx_udct->forward(xr);
    TORCH_CHECK(yc.is_complex() && yc.size(0) == 2 && yc.size(1) == cplx_udct->ncoef());
    const torch::Tensor recon_from_real = cplx_udct->backward(torch::real(yc));
    TORCH_CHECK(recon_from_real.is_complex() && recon_from_real.sizes() == xr.sizes());
    std::cout << "[PASS] edge cases (empty batch, real-in-complex, real-coeffs-bwd)"
              << std::endl;
}

struct CurveletFilterNetImpl : public torch::nn::Module {
    CurveletFilterNetImpl(const std::string& plan_path, int64_t ncoef)
        : udct(register_module("udct", curvelets::UDCT(plan_path))),
          conv(register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 3).padding(1)))),
          gain(register_parameter("gain", torch::ones({ncoef}, torch::kFloat32))) {}

    torch::Tensor forward(const torch::Tensor& x) {
        const auto h = conv->forward(x.unsqueeze(1)).squeeze(1);
        const auto c = udct->forward(h);
        return udct->backward(c * gain);
    }

    curvelets::UDCT udct{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::Tensor gain;
};
TORCH_MODULE(CurveletFilterNet);

void test_native_training(const fs::path& dir) {
    const fs::path plan_path = dir / "real_2d_f32_plan.pt";
    curvelets::UDCT probe(plan_path.string());
    CurveletFilterNet model(plan_path.string(), probe->ncoef());
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(5e-2));

    torch::manual_seed(0);
    const torch::Tensor x = torch::randn({4, 32, 32}, torch::kFloat32);
    const torch::Tensor target = torch::sin(x);

    double first_loss = 0.0;
    double last_loss = 0.0;
    for (int step = 0; step < 20; ++step) {
        opt.zero_grad();
        const torch::Tensor pred = model->forward(x);
        const torch::Tensor loss = torch::mse_loss(pred, target);
        loss.backward();
        for (const auto& p : model->parameters()) {
            TORCH_CHECK(p.grad().defined(), "Missing parameter grad in native training");
            TORCH_CHECK(torch::isfinite(p.grad()).all().item<bool>(), "Non-finite grad");
        }
        opt.step();
        const double l = loss.item<double>();
        if (step == 0) {
            first_loss = l;
        }
        last_loss = l;
    }
    TORCH_CHECK(last_loss < 0.5 * first_loss,
                "Expected native C++ training loss to drop significantly: ",
                first_loss, " -> ", last_loss);

    // Verify precision conversion in C++ (weights were originally float32, so
    // tight-frame reconstruction holds to float32 precision ~1e-6).
    probe->to(torch::kFloat64);
    const torch::Tensor x64 = torch::randn({2, 32, 32}, torch::kFloat64);
    const torch::Tensor recon64 = probe->backward(probe->forward(x64));
    assert_close(recon64, x64, 1e-5, 1e-5, "cast_f64_roundtrip");

    // Monogenic precision conversion preserves Riesz imaginary components.
    curvelets::UDCT mono_probe((dir / "monogenic_2d_f64_plan.pt").string());
    mono_probe->to(torch::kFloat32);
    const torch::Tensor xm32 = torch::randn({32, 32}, torch::kFloat32);
    const torch::Tensor ym32 = mono_probe->forward(xm32);
    const torch::Tensor rm32 = mono_probe->backward(ym32);
    assert_close(rm32[0], xm32, 1e-4, 1e-4, "mono_cast_f32_roundtrip");

    std::cout << "[PASS] native C++ training (" << first_loss << " -> " << last_loss << ")"
              << std::endl;
}

void test_gradients(const fs::path& dir) {
    curvelets::UDCT udct((dir / "real_2d_f64_plan.pt").string());
    torch::manual_seed(7);
    const torch::Tensor x = torch::randn({32, 32}, torch::kFloat64);
    const torch::Tensor w_fwd = torch::randn({udct->ncoef()}, torch::kComplexDouble);

    // Finite-difference check on forward
    torch::Tensor xg = x.clone().requires_grad_(true);
    torch::real((udct->forward(xg) * w_fwd).sum()).backward();
    const double eps = 1e-6;
    for (int i = 0; i < 4; ++i) {
        const int r = (i * 7) % 32;
        const int c = (i * 11) % 32;
        torch::Tensor xp = x.clone();
        torch::Tensor xm = x.clone();
        xp[r][c] += eps;
        xm[r][c] -= eps;
        const double lp = torch::real((udct->forward(xp) * w_fwd).sum()).item<double>();
        const double lm = torch::real((udct->forward(xm) * w_fwd).sum()).item<double>();
        const double fd = (lp - lm) / (2.0 * eps);
        const double ad = xg.grad()[r][c].item<double>();
        TORCH_CHECK(std::abs(fd - ad) < 1e-6, "Forward finite diff mismatch: ", fd, " vs ", ad);
    }

    // Finite-difference check on backward (real coefficients -> real output)
    const torch::Tensor v = torch::randn({udct->ncoef()}, torch::kFloat64);
    const torch::Tensor w_bwd = torch::randn({32, 32}, torch::kFloat64);
    torch::Tensor vg = v.clone().requires_grad_(true);
    (udct->backward(vg) * w_bwd).sum().backward();
    for (int i = 0; i < 4; ++i) {
        const int k = (i * 97) % udct->ncoef();
        torch::Tensor vp = v.clone();
        torch::Tensor vm = v.clone();
        vp[k] += eps;
        vm[k] -= eps;
        const double lp = (udct->backward(vp) * w_bwd).sum().item<double>();
        const double lm = (udct->backward(vm) * w_bwd).sum().item<double>();
        const double fd = (lp - lm) / (2.0 * eps);
        const double ad = vg.grad()[k].item<double>();
        TORCH_CHECK(std::abs(fd - ad) < 1e-6, "Backward finite diff mismatch: ", fd, " vs ", ad);
    }

    // Second-order gradients via create_graph=true (tight frame: grad^2(0.5 * ||C x||^2) == 1)
    torch::Tensor x2 = x.clone().requires_grad_(true);
    const torch::Tensor e_fwd = udct->forward(x2).abs().pow(2).sum();
    const auto g1 = torch::autograd::grad({e_fwd}, {x2}, {}, /*retain_graph=*/true, /*create_graph=*/true);
    TORCH_CHECK(g1[0].requires_grad(), "Expected 1st-order grad to retain graph");
    const auto g2 = torch::autograd::grad({g1[0].pow(2).sum()}, {x2});
    assert_close(g2[0], 8.0 * x, 1e-8, 1e-8, "second_order_grad");

    std::cout << "[PASS] C++ 1st-order finite-diff & 2nd-order autograd checks" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_udct_cpp <export_dir>\n";
        return 2;
    }
    const fs::path dir(argv[1]);
    test_plan_parity(dir, "real_2d_f32", 1e-4, 1e-4);
    test_plan_parity(dir, "real_2d_f64", 1e-10, 1e-10);
    test_plan_parity(dir, "real_wavelet_2d_f32", 1e-4, 1e-4);
    test_plan_parity(dir, "complex_2d_f64", 1e-10, 1e-10);
    test_plan_parity(dir, "monogenic_2d_f64", 1e-10, 1e-10);
    test_plan_parity(dir, "real_3d_f32", 1e-4, 1e-4);
    test_plan_parity(dir, "real_odd_2d_f64", 1e-10, 1e-10);
    test_edge_cases(dir);
    test_native_training(dir);
    test_gradients(dir);
    std::cout << "All C++ libtorch tests passed.\n";
    return 0;
}
