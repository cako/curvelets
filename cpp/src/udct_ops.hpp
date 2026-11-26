#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

#include <torch/torch.h>

namespace curvelets::detail {

inline int64_t product(std::span<const int64_t> values) {
    return std::reduce(values.begin(), values.end(), int64_t{1}, std::multiplies<>{});
}

inline std::vector<int64_t> fft_dims(int64_t ndim) {
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
