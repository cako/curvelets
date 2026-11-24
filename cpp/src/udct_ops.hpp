#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace curvelets::detail {

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
