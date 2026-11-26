#include "udct_ops.hpp"

#include <functional>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <ATen/Parallel.h>

namespace curvelets::detail {

namespace {

std::vector<int64_t> spectrum_shape_for(c10::IntArrayRef shape, bool complex_output) {
    std::vector<int64_t> out(shape.begin(), shape.end());
    if (!complex_output && !out.empty()) {
        out.back() = out.back() / 2 + 1;
    }
    return out;
}

template <typename scalar_t, typename row_t, typename col_t>
void csr_fold_cpu_kernel(
    const at::Tensor& src,
    const at::Tensor& row_ptr,
    const at::Tensor& col,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_targets) {
    const int64_t rows = src.size(0);
    const int64_t num_sources = src.size(1);
    const bool pair = weights.dim() == 2;

    const auto* src_ptr = src.const_data_ptr<c10::complex<scalar_t>>();
    auto* out_ptr = out.data_ptr<c10::complex<scalar_t>>();
    const auto* rp_ptr = row_ptr.const_data_ptr<row_t>();
    const auto* col_ptr = col.const_data_ptr<col_t>();
    const auto* w_ptr = weights.const_data_ptr<scalar_t>();

    at::parallel_for(0, rows * num_targets, 1024, [&](int64_t begin, int64_t end) {
        if (pair) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t r = idx / num_targets;
                const int64_t t = idx % num_targets;
                const int64_t start = static_cast<int64_t>(rp_ptr[t]);
                const int64_t stop = static_cast<int64_t>(rp_ptr[t + 1]);
                const auto* row_src = src_ptr + r * num_sources;

                scalar_t acc_re = 0;
                scalar_t acc_im = 0;
                for (int64_t j = start; j < stop; ++j) {
                    const int64_t s = static_cast<int64_t>(col_ptr[j]);
                    const c10::complex<scalar_t> v = row_src[s];
                    const scalar_t w_re = w_ptr[2 * j];
                    const scalar_t w_im = w_ptr[2 * j + 1];
                    acc_re += w_re * v.real();
                    acc_im += w_im * v.imag();
                }
                out_ptr[idx] = c10::complex<scalar_t>(acc_re, acc_im);
            }
        } else {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t r = idx / num_targets;
                const int64_t t = idx % num_targets;
                const int64_t start = static_cast<int64_t>(rp_ptr[t]);
                const int64_t stop = static_cast<int64_t>(rp_ptr[t + 1]);
                const auto* row_src = src_ptr + r * num_sources;

                out_ptr[idx] = std::transform_reduce(
                    col_ptr + start,
                    col_ptr + stop,
                    w_ptr + start,
                    c10::complex<scalar_t>(0, 0),
                    std::plus<>{},
                    [row_src](col_t c, scalar_t w) {
                        const c10::complex<scalar_t> v = row_src[static_cast<int64_t>(c)];
                        return c10::complex<scalar_t>(w * v.real(), w * v.imag());
                    });
            }
        }
    });
}

template <typename scalar_t, typename row_t>
void csr_fold_cpu_dispatch_col(
    const at::Tensor& src,
    const at::Tensor& row_ptr,
    const at::Tensor& col,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_targets) {
    if (col.scalar_type() == at::kInt) {
        csr_fold_cpu_kernel<scalar_t, row_t, int32_t>(src, row_ptr, col, weights, out, num_targets);
    } else if (col.scalar_type() == at::kLong) {
        csr_fold_cpu_kernel<scalar_t, row_t, int64_t>(src, row_ptr, col, weights, out, num_targets);
    } else {
        TORCH_CHECK(false, "csr_fold col must be int32 or int64, got ", col.scalar_type());
    }
}

template <typename scalar_t>
void csr_fold_cpu_dispatch_idx(
    const at::Tensor& src,
    const at::Tensor& row_ptr,
    const at::Tensor& col,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_targets) {
    if (row_ptr.scalar_type() == at::kInt) {
        csr_fold_cpu_dispatch_col<scalar_t, int32_t>(src, row_ptr, col, weights, out, num_targets);
    } else if (row_ptr.scalar_type() == at::kLong) {
        csr_fold_cpu_dispatch_col<scalar_t, int64_t>(src, row_ptr, col, weights, out, num_targets);
    } else {
        TORCH_CHECK(false, "csr_fold row_ptr must be int32 or int64, got ", row_ptr.scalar_type());
    }
}

at::Tensor csr_fold_cpu(
    const at::Tensor& src_in,
    const at::Tensor& row_ptr_in,
    const at::Tensor& col_in,
    const at::Tensor& weights_in,
    int64_t num_targets) {
    const at::Tensor src = src_in.contiguous();
    const int64_t rows = src.size(0);
    at::Tensor out = at::empty({rows, num_targets}, src.options());
    if (rows == 0 || num_targets == 0) {
        return out;
    }
    const at::Tensor row_ptr = row_ptr_in.contiguous();
    const at::Tensor col = col_in.contiguous();
    const at::Tensor weights = weights_in.contiguous();

    if (weights.scalar_type() == at::kFloat) {
        TORCH_CHECK(src.scalar_type() == at::kComplexFloat, "Expected complex64 src for float32 weights");
        csr_fold_cpu_dispatch_idx<float>(src, row_ptr, col, weights, out, num_targets);
    } else if (weights.scalar_type() == at::kDouble) {
        TORCH_CHECK(src.scalar_type() == at::kComplexDouble, "Expected complex128 src for float64 weights");
        csr_fold_cpu_dispatch_idx<double>(src, row_ptr, col, weights, out, num_targets);
    } else {
        TORCH_CHECK(false, "csr_fold weights must be float32 or float64, got ", weights.scalar_type());
    }
    return out;
}

at::Tensor csr_fold_aten(
    const at::Tensor& src,
    const at::Tensor& row_ptr,
    const at::Tensor& col,
    const at::Tensor& weights,
    int64_t num_targets) {
    const int64_t rows = src.size(0);
    if (rows == 0 || num_targets == 0) {
        return at::empty({rows, num_targets}, src.options());
    }
    const at::Tensor parts = at::view_as_real(src);
    const at::Tensor w = weights.dim() == 2 ? weights : weights.unsqueeze(-1);
    const at::Tensor terms = parts.index_select(1, col.to(at::kLong)) * w;
    const at::Tensor rp64 = row_ptr.to(at::kLong);
    const at::Tensor counts = rp64.slice(0, 1, rp64.size(0)) - rp64.slice(0, 0, rp64.size(0) - 1);
    const at::Tensor targets = at::repeat_interleave(
        at::arange(num_targets, rp64.options()), counts);
    at::Tensor out = at::zeros({rows, num_targets, 2}, parts.options());
    out.index_add_(1, targets, terms);
    return at::view_as_complex(out);
}

at::Tensor csr_fold_device(
    const at::Tensor& src,
    const at::Tensor& row_ptr,
    const at::Tensor& col,
    const at::Tensor& weights,
    int64_t num_targets) {
    if (src.is_cpu()) {
        return csr_fold_cpu(src, row_ptr, col, weights, num_targets);
    }
    return csr_fold_aten(src, row_ptr, col, weights, num_targets);
}

void validate_group_metadata(
    int64_t ndim,
    c10::IntArrayRef group_offsets,
    c10::IntArrayRef group_counts,
    c10::IntArrayRef group_shapes) {
    const int64_t num_groups = group_offsets.size();
    TORCH_CHECK(group_counts.size() == static_cast<size_t>(num_groups),
                "group_counts length must match group_offsets");
    TORCH_CHECK(group_shapes.size() == static_cast<size_t>(num_groups * ndim),
                "group_shapes length must equal num_groups * ndim");
}

void check_precision(const at::Tensor& tensor, const at::Tensor& weights, const char* what) {
    const auto st = tensor.scalar_type();
    TORCH_CHECK(
        st == at::kFloat || st == at::kDouble || st == at::kComplexFloat || st == at::kComplexDouble,
        "Unsupported ", what, " dtype ", st);
    const auto real_st = c10::toRealValueType(st);
    TORCH_CHECK(
        real_st == weights.scalar_type(),
        "UDCT was built for ", weights.scalar_type(), " but got ", st, " ", what);
}

at::Tensor udct_forward_impl(
    const at::Tensor& image,
    const at::Tensor& fwd_row_ptr,
    const at::Tensor& fwd_col,
    const at::Tensor& fwd_weights,
    c10::IntArrayRef shape,
    c10::IntArrayRef group_offsets,
    c10::IntArrayRef group_counts,
    c10::IntArrayRef group_shapes,
    int64_t ncoef,
    bool complex_output) {
    const int64_t ndim = static_cast<int64_t>(shape.size());
    TORCH_CHECK(ndim >= 1, "shape must be non-empty");
    validate_group_metadata(ndim, group_offsets, group_counts, group_shapes);

    check_trailing_shape(image, shape, "input");
    if (!complex_output) {
        TORCH_CHECK(!image.is_complex(), "Real UDCT requires real-valued input");
    }
    check_precision(image, fwd_weights, "input");
    TORCH_CHECK(image.device() == fwd_weights.device(),
                "Input device (", image.device(), ") does not match UDCT device (", fwd_weights.device(), ")");

    std::vector<int64_t> batch_dims(image.sizes().begin(), image.sizes().end() - ndim);
    std::vector<int64_t> out_shape = batch_dims;
    out_shape.push_back(ncoef);
    const auto cdtype = c10::toComplexType(fwd_weights.scalar_type());

    if (image.numel() == 0) {
        return at::empty(out_shape, image.options().dtype(cdtype));
    }

    std::vector<int64_t> flat_image_shape = {-1};
    flat_image_shape.insert(flat_image_shape.end(), shape.begin(), shape.end());
    const at::Tensor images = image.reshape(flat_image_shape);
    const int64_t rows = images.size(0);
    const auto dims = fft_dims(ndim);

    const at::Tensor spectrum = complex_output
        ? at::fft_fftn(images, std::nullopt, dims)
        : at::fft_rfftn(images, std::nullopt, dims);
    const auto spec_shape = spectrum_shape_for(shape, complex_output);
    const int64_t num_spectrum = product(spec_shape);
    const at::Tensor folded = csr_fold_device(
        spectrum.reshape({rows, num_spectrum}),
        fwd_row_ptr,
        fwd_col,
        fwd_weights,
        ncoef);

    at::Tensor out = at::empty({rows, ncoef}, folded.options());
    const int64_t num_groups = static_cast<int64_t>(group_offsets.size());
    for (int64_t g = 0; g < num_groups; ++g) {
        const int64_t offset = group_offsets[static_cast<size_t>(g)];
        const int64_t count = group_counts[static_cast<size_t>(g)];
        const auto g_shape = std::span(group_shapes).subspan(
            static_cast<size_t>(g * ndim), static_cast<size_t>(ndim));
        std::vector<int64_t> chunk_shape;
        chunk_shape.reserve(static_cast<size_t>(ndim + 2));
        chunk_shape.push_back(rows);
        chunk_shape.push_back(count);
        chunk_shape.insert(chunk_shape.end(), g_shape.begin(), g_shape.end());
        const int64_t g_len = count * product(g_shape);
        const at::Tensor chunk = folded.narrow(1, offset, g_len).view(chunk_shape);
        // 1/g_size normalization is folded into fwd_weights.
        const at::Tensor ifft = at::fft_ifftn(chunk, std::nullopt, dims, "forward");
        out.narrow(1, offset, g_len).copy_(ifft.reshape({rows, g_len}));
    }
    return out.reshape(out_shape);
}

at::Tensor udct_backward_impl(
    const at::Tensor& vector,
    const at::Tensor& bwd_row_ptr,
    const at::Tensor& bwd_col,
    const at::Tensor& bwd_weights,
    c10::IntArrayRef shape,
    c10::IntArrayRef group_offsets,
    c10::IntArrayRef group_counts,
    c10::IntArrayRef group_shapes,
    int64_t ncoef,
    bool complex_output) {
    const int64_t ndim = static_cast<int64_t>(shape.size());
    TORCH_CHECK(ndim >= 1, "shape must be non-empty");
    validate_group_metadata(ndim, group_offsets, group_counts, group_shapes);

    TORCH_CHECK(vector.dim() >= 1 && vector.size(-1) == ncoef,
                "Expected coefficients with trailing dimension ", ncoef, ", got ", vector.sizes());
    check_precision(vector, bwd_weights, "coefficients");
    TORCH_CHECK(vector.device() == bwd_weights.device(),
                "Coefficients device (", vector.device(), ") does not match UDCT device (", bwd_weights.device(), ")");

    std::vector<int64_t> batch_dims(vector.sizes().begin(), vector.sizes().end() - 1);
    std::vector<int64_t> out_shape = batch_dims;
    out_shape.insert(out_shape.end(), shape.begin(), shape.end());
    const auto cdtype = c10::toComplexType(bwd_weights.scalar_type());
    const auto out_dtype = complex_output ? cdtype : bwd_weights.scalar_type();

    if (vector.numel() == 0) {
        return at::empty(out_shape, vector.options().dtype(out_dtype));
    }

    const at::Tensor flat = vector.reshape({-1, ncoef});
    const int64_t rows = flat.size(0);
    const auto dims = fft_dims(ndim);

    at::Tensor small = at::empty({rows, ncoef}, flat.options().dtype(cdtype));
    const int64_t num_groups = static_cast<int64_t>(group_offsets.size());
    for (int64_t g = 0; g < num_groups; ++g) {
        const int64_t offset = group_offsets[static_cast<size_t>(g)];
        const int64_t count = group_counts[static_cast<size_t>(g)];
        const auto g_shape = std::span(group_shapes).subspan(
            static_cast<size_t>(g * ndim), static_cast<size_t>(ndim));
        std::vector<int64_t> chunk_shape;
        chunk_shape.reserve(static_cast<size_t>(ndim + 2));
        chunk_shape.push_back(rows);
        chunk_shape.push_back(count);
        chunk_shape.insert(chunk_shape.end(), g_shape.begin(), g_shape.end());
        const int64_t g_len = count * product(g_shape);
        const at::Tensor chunk = flat.narrow(1, offset, g_len).view(chunk_shape);
        const at::Tensor fft = at::fft_fftn(chunk, std::nullopt, dims);
        small.narrow(1, offset, g_len).copy_(fft.reshape({rows, g_len}));
    }

    const auto spec_shape = spectrum_shape_for(shape, complex_output);
    const int64_t num_spectrum = product(spec_shape);
    const at::Tensor spectrum_flat = csr_fold_device(
        small,
        bwd_row_ptr,
        bwd_col,
        bwd_weights,
        num_spectrum);

    std::vector<int64_t> spec_batched_shape = {rows};
    spec_batched_shape.insert(spec_batched_shape.end(), spec_shape.begin(), spec_shape.end());
    const at::Tensor spectrum = spectrum_flat.reshape(spec_batched_shape);

    const at::Tensor recon = complex_output
        ? at::fft_ifftn(spectrum, std::nullopt, dims)
        : at::fft_irfftn(spectrum, shape, dims);
    return recon.reshape(out_shape);
}

class UdctForwardAutograd : public torch::autograd::Function<UdctForwardAutograd> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
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
        bool complex_output) {
        ctx->save_for_backward(
            {fwd_row_ptr, fwd_col, fwd_weights, bwd_row_ptr, bwd_col, bwd_weights});
        ctx->saved_data["shape"] = shape.vec();
        ctx->saved_data["group_offsets"] = group_offsets.vec();
        ctx->saved_data["group_counts"] = group_counts.vec();
        ctx->saved_data["group_shapes"] = group_shapes.vec();
        ctx->saved_data["ncoef"] = ncoef;
        ctx->saved_data["complex_output"] = complex_output;
        ctx->saved_data["real_input"] = !image.is_complex();

        torch::NoGradGuard no_grad;
        return udct_forward_impl(
            image,
            fwd_row_ptr,
            fwd_col,
            fwd_weights,
            shape,
            group_offsets,
            group_counts,
            group_shapes,
            ncoef,
            complex_output);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        const auto shape = ctx->saved_data["shape"].toIntVector();
        const auto group_offsets = ctx->saved_data["group_offsets"].toIntVector();
        const auto group_counts = ctx->saved_data["group_counts"].toIntVector();
        const auto group_shapes = ctx->saved_data["group_shapes"].toIntVector();
        const int64_t ncoef = ctx->saved_data["ncoef"].toInt();
        const bool complex_output = ctx->saved_data["complex_output"].toBool();
        const bool real_input = ctx->saved_data["real_input"].toBool();

        at::Tensor grad = udct_backward(
            grad_outputs[0],
            saved[0],
            saved[1],
            saved[2],
            saved[3],
            saved[4],
            saved[5],
            shape,
            group_offsets,
            group_counts,
            group_shapes,
            ncoef,
            complex_output);
        if (real_input && grad.is_complex()) {
            grad = at::real(grad);
        }
        torch::autograd::variable_list grads(13);
        grads[0] = std::move(grad);
        return grads;
    }
};

class UdctBackwardAutograd : public torch::autograd::Function<UdctBackwardAutograd> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
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
        bool complex_output) {
        ctx->save_for_backward(
            {fwd_row_ptr, fwd_col, fwd_weights, bwd_row_ptr, bwd_col, bwd_weights});
        ctx->saved_data["shape"] = shape.vec();
        ctx->saved_data["group_offsets"] = group_offsets.vec();
        ctx->saved_data["group_counts"] = group_counts.vec();
        ctx->saved_data["group_shapes"] = group_shapes.vec();
        ctx->saved_data["ncoef"] = ncoef;
        ctx->saved_data["complex_output"] = complex_output;
        ctx->saved_data["real_input"] = !vector.is_complex();

        torch::NoGradGuard no_grad;
        return udct_backward_impl(
            vector,
            bwd_row_ptr,
            bwd_col,
            bwd_weights,
            shape,
            group_offsets,
            group_counts,
            group_shapes,
            ncoef,
            complex_output);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        const auto shape = ctx->saved_data["shape"].toIntVector();
        const auto group_offsets = ctx->saved_data["group_offsets"].toIntVector();
        const auto group_counts = ctx->saved_data["group_counts"].toIntVector();
        const auto group_shapes = ctx->saved_data["group_shapes"].toIntVector();
        const int64_t ncoef = ctx->saved_data["ncoef"].toInt();
        const bool complex_output = ctx->saved_data["complex_output"].toBool();
        const bool real_input = ctx->saved_data["real_input"].toBool();

        at::Tensor grad = udct_forward(
            grad_outputs[0],
            saved[0],
            saved[1],
            saved[2],
            saved[3],
            saved[4],
            saved[5],
            shape,
            group_offsets,
            group_counts,
            group_shapes,
            ncoef,
            complex_output);
        if (real_input && grad.is_complex()) {
            grad = at::real(grad);
        }
        torch::autograd::variable_list grads(13);
        grads[0] = std::move(grad);
        return grads;
    }
};

} // namespace

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
    bool complex_output) {
    return UdctForwardAutograd::apply(
        image,
        fwd_row_ptr,
        fwd_col,
        fwd_weights,
        bwd_row_ptr,
        bwd_col,
        bwd_weights,
        shape,
        group_offsets,
        group_counts,
        group_shapes,
        ncoef,
        complex_output);
}

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
    bool complex_output) {
    return UdctBackwardAutograd::apply(
        vector,
        fwd_row_ptr,
        fwd_col,
        fwd_weights,
        bwd_row_ptr,
        bwd_col,
        bwd_weights,
        shape,
        group_offsets,
        group_counts,
        group_shapes,
        ncoef,
        complex_output);
}

} // namespace curvelets::detail
