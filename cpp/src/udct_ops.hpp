#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#include <torch/torch.h>

namespace curvelets::detail {

inline int64_t checked_add(int64_t a, int64_t b, const char* what = "addition") {
    TORCH_CHECK(a >= 0 && b >= 0, "Expected non-negative operands in ", what);
    TORCH_CHECK(a <= std::numeric_limits<int64_t>::max() - b, "Integer overflow in ", what);
    return a + b;
}

inline int64_t checked_mul(int64_t a, int64_t b, const char* what = "multiplication") {
    TORCH_CHECK(a >= 0 && b >= 0, "Expected non-negative operands in ", what);
    if (a == 0 || b == 0) {
        return 0;
    }
    TORCH_CHECK(a <= std::numeric_limits<int64_t>::max() / b, "Integer overflow in ", what);
    return a * b;
}

inline int64_t product(std::span<const int64_t> values) {
    return std::accumulate(
        values.begin(),
        values.end(),
        int64_t{1},
        [](int64_t a, int64_t b) { return checked_mul(a, b, "shape product"); });
}

inline std::vector<int64_t> fft_dims(int64_t ndim) {
    TORCH_CHECK(ndim >= 0, "ndim must be non-negative, got ", ndim);
    std::vector<int64_t> dims(static_cast<size_t>(ndim));
    std::iota(dims.begin(), dims.end(), -ndim);
    return dims;
}

inline void check_trailing_shape(
    const at::Tensor& tensor,
    std::span<const int64_t> expected,
    const char* what) {
    const auto n = static_cast<int64_t>(expected.size());
    TORCH_CHECK(
        tensor.dim() >= n,
        "Expected ", what, " with at least ", n, " dims, got ", tensor.dim());
    const auto trailing = tensor.sizes().slice(static_cast<size_t>(tensor.dim() - n));
    TORCH_CHECK(
        std::ranges::equal(trailing, expected),
        "Trailing shape mismatch in ", what);
}

inline std::vector<int64_t> spectrum_shape_for(
    std::span<const int64_t> shape,
    bool complex_output) {
    std::vector<int64_t> out(shape.begin(), shape.end());
    if (!complex_output && !out.empty()) {
        out.back() = out.back() / 2 + 1;
    }
    return out;
}

void validate_csr_plan(
    const at::Tensor& row_ptr,
    const at::Tensor& col,
    const at::Tensor& weights,
    int64_t num_sources,
    int64_t num_targets,
    const char* label);

at::Tensor udct_forward(
    const at::Tensor& image,
    const at::Tensor& fwd_row_ptr,
    const at::Tensor& fwd_col,
    const at::Tensor& fwd_weights,
    const at::Tensor& bwd_row_ptr,
    const at::Tensor& bwd_col,
    const at::Tensor& bwd_weights,
    c10::IntArrayRef shape,
    c10::IntArrayRef group_offsets,
    c10::IntArrayRef group_counts,
    c10::IntArrayRef group_shapes,
    int64_t ncoef,
    bool complex_output);

at::Tensor udct_backward(
    const at::Tensor& vector,
    const at::Tensor& fwd_row_ptr,
    const at::Tensor& fwd_col,
    const at::Tensor& fwd_weights,
    const at::Tensor& bwd_row_ptr,
    const at::Tensor& bwd_col,
    const at::Tensor& bwd_weights,
    c10::IntArrayRef shape,
    c10::IntArrayRef group_offsets,
    c10::IntArrayRef group_counts,
    c10::IntArrayRef group_shapes,
    int64_t ncoef,
    bool complex_output);

} // namespace curvelets::detail
