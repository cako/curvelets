#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace curvelets {

/// Native libtorch module for running and training with an exported UDCT plan.
class UDCTImpl : public torch::nn::Module {
public:
    /// Load a UDCT plan exported from Python via ``UDCT.save_plan``.
    explicit UDCTImpl(const std::string& plan_path);

    /// Forward transform of ``(*batch, *shape)`` to flat ``(*batch, ncoef)`` coefficients.
    torch::Tensor forward(const torch::Tensor& image);

    /// Backward transform of flat ``(*batch, ncoef)`` coefficients.
    /// Returns ``(*batch, *shape)`` (or ``(ndim + 1, *batch, *shape)`` for monogenic).
    torch::Tensor backward(const torch::Tensor& vector);

    using torch::nn::Module::to;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t ncoef() const { return total_ncoef_; }
    const std::string& transform_kind() const { return transform_kind_; }
    bool complex_output() const { return complex_output_; }

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> group_offsets_;
    std::vector<int64_t> group_counts_;
    std::vector<int64_t> group_shapes_;
    std::vector<int64_t> even_axes_;
    int64_t plan_ncoef_ = 0;
    int64_t total_ncoef_ = 0;
    bool complex_output_ = false;
    std::string transform_kind_;

    torch::Tensor fwd_row_ptr_;
    torch::Tensor fwd_col_;
    torch::Tensor fwd_weights_;
    torch::Tensor bwd_row_ptr_;
    torch::Tensor bwd_col_;
    torch::Tensor bwd_weights_;
    torch::Tensor riesz_filters_;
    torch::Tensor monogenic_perm_;
    torch::Tensor monogenic_inverse_;
};

TORCH_MODULE(UDCT);

} // namespace curvelets
