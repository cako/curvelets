#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#if defined(_WIN32)
#if defined(CURVELETS_BUILD_DLL)
#define CURVELETS_API __declspec(dllexport)
#else
#define CURVELETS_API __declspec(dllimport)
#endif
#else
#define CURVELETS_API
#endif

namespace curvelets {

/// Nested curvelet coefficients indexed as ``coeffs[scale][direction][wedge]``.
using UDCTCoefficients = std::vector<std::vector<std::vector<torch::Tensor>>>;

/// Nested wedge shapes indexed as ``shapes[scale][direction][wedge]``.
using UDCTCoefficientShapes =
    std::vector<std::vector<std::vector<std::vector<int64_t>>>>;

/// Native libtorch module for running and training with an exported UDCT plan.
class CURVELETS_API UDCTImpl : public torch::nn::Module {
public:
    /// Load a UDCT plan exported from Python via ``UDCT.save_plan``.
    explicit UDCTImpl(const std::string& plan_path);

    /// Forward transform of ``(*batch, *shape)`` to flat ``(*batch, ncoef)`` coefficients.
    torch::Tensor forward(const torch::Tensor& image);

    /// Forward transform returning nested ``[scale][direction][wedge]`` coefficient views.
    UDCTCoefficients forward_structured(const torch::Tensor& image);

    /// Backward transform of flat ``(*batch, ncoef)`` coefficients.
    /// Returns ``(*batch, *shape)`` (or ``(ndim + 1, *batch, *shape)`` for monogenic).
    torch::Tensor backward(const torch::Tensor& vector);

    /// Backward transform of nested ``[scale][direction][wedge]`` coefficients.
    torch::Tensor backward(const UDCTCoefficients& coefficients);

    /// Restructure a flat ``(*batch, ncoef)`` vector into nested coefficient views.
    UDCTCoefficients structure(const torch::Tensor& vector) const;

    /// Flatten nested ``[scale][direction][wedge]`` coefficients into ``(*batch, ncoef)``.
    torch::Tensor vect(const UDCTCoefficients& coefficients) const;

    /// Return the expected trailing shape of every coefficient wedge.
    UDCTCoefficientShapes coefficient_shapes() const;

    using torch::nn::Module::to;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;

    const std::vector<int64_t>& shape() const noexcept { return shape_; }
    int64_t ndim() const noexcept { return static_cast<int64_t>(shape_.size()); }
    int64_t num_scales() const noexcept { return static_cast<int64_t>(num_directions_.size()); }
    int64_t ncoef() const noexcept { return total_ncoef_; }
    const std::string& transform_kind() const noexcept { return transform_kind_; }
    bool complex_output() const noexcept { return complex_output_; }

private:
    struct LayoutEntry {
        int64_t scale;
        int64_t direction;
        int64_t wedge;
    };

    struct WedgeSpan {
        int64_t start;
        std::vector<int64_t> shape;
    };

    std::vector<int64_t> shape_;
    std::vector<int64_t> group_offsets_;
    std::vector<int64_t> group_counts_;
    std::vector<int64_t> group_shapes_;
    std::vector<int64_t> num_directions_;
    std::vector<LayoutEntry> layout_;
    std::vector<std::vector<int64_t>> entry_shapes_;
    std::vector<std::vector<int64_t>> wedge_counts_;
    std::vector<std::vector<std::vector<WedgeSpan>>> mono_wedges_;
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
