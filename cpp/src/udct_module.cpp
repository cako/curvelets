#include "curvelets/udct.hpp"
#include "udct_ops.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/csrc/jit/serialization/pickle.h>

namespace curvelets {

namespace {

std::vector<char> read_file_bytes(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(ifs.good(), "Failed to open UDCT plan file: ", path);
    const auto size = ifs.tellg();
    TORCH_CHECK(size >= 0, "Failed to determine size of UDCT plan file: ", path);
    std::vector<char> bytes(static_cast<size_t>(size));
    ifs.seekg(0, std::ios::beg);
    ifs.read(bytes.data(), size);
    TORCH_CHECK(ifs.good(), "Failed to read UDCT plan file: ", path);
    return bytes;
}

at::Tensor get_tensor_field(
    const c10::Dict<c10::IValue, c10::IValue>& dict,
    const char* key) {
    auto it = dict.find(c10::IValue(key));
    TORCH_CHECK(it != dict.end(), "UDCT plan missing required key: ", key);
    return it->value().toTensor();
}

std::vector<int64_t> to_int64_vec(const at::Tensor& t) {
    const at::Tensor cpu = t.to(at::kCPU, at::kLong).contiguous();
    const auto* ptr = cpu.const_data_ptr<int64_t>();
    return std::vector<int64_t>(ptr, ptr + cpu.numel());
}

std::string kind_from_code(int64_t code) {
    switch (code) {
        case 0:
            return "real";
        case 1:
            return "complex";
        case 2:
            return "monogenic";
        default:
            TORCH_CHECK(false, "Unknown UDCT transform_kind code in plan: ", code);
    }
}

} // namespace

UDCTImpl::UDCTImpl(const std::string& plan_path) {
    const auto bytes = read_file_bytes(plan_path);
    const c10::IValue loaded = torch::jit::pickle_load(bytes);
    TORCH_CHECK(loaded.isGenericDict(), "UDCT plan must be a dictionary of tensors");
    const auto dict = loaded.toGenericDict();

    const auto meta = to_int64_vec(get_tensor_field(dict, "meta"));
    TORCH_CHECK(meta.size() == 4, "Invalid UDCT plan meta tensor length");
    TORCH_CHECK(meta[0] == 1, "Unsupported UDCT plan format version: ", meta[0]);

    plan_ncoef_ = meta[1];
    complex_output_ = meta[2] != 0;
    transform_kind_ = kind_from_code(meta[3]);

    shape_ = to_int64_vec(get_tensor_field(dict, "shape"));
    group_offsets_ = to_int64_vec(get_tensor_field(dict, "group_offsets"));
    group_counts_ = to_int64_vec(get_tensor_field(dict, "group_counts"));
    group_shapes_ = to_int64_vec(get_tensor_field(dict, "group_shapes"));
    even_axes_ = to_int64_vec(get_tensor_field(dict, "even_axes"));

    fwd_row_ptr_ = register_buffer("fwd_row_ptr", get_tensor_field(dict, "fwd_row_ptr"));
    fwd_col_ = register_buffer("fwd_col", get_tensor_field(dict, "fwd_col"));
    fwd_weights_ = register_buffer("fwd_weights", get_tensor_field(dict, "fwd_weights"));
    bwd_row_ptr_ = register_buffer("bwd_row_ptr", get_tensor_field(dict, "bwd_row_ptr"));
    bwd_col_ = register_buffer("bwd_col", get_tensor_field(dict, "bwd_col"));
    bwd_weights_ = register_buffer("bwd_weights", get_tensor_field(dict, "bwd_weights"));

    riesz_filters_ = register_buffer("riesz_filters", get_tensor_field(dict, "riesz_filters"));
    monogenic_perm_ = register_buffer("monogenic_perm", get_tensor_field(dict, "monogenic_perm"));
    monogenic_inverse_ = register_buffer("monogenic_inverse", get_tensor_field(dict, "monogenic_inverse"));

    total_ncoef_ = (transform_kind_ == "monogenic")
        ? monogenic_perm_.numel()
        : plan_ncoef_;
}

torch::Tensor UDCTImpl::forward(const torch::Tensor& image) {
    if (transform_kind_ != "monogenic") {
        return detail::udct_forward(
            image,
            fwd_row_ptr_,
            fwd_col_,
            fwd_weights_,
            bwd_row_ptr_,
            bwd_col_,
            bwd_weights_,
            shape_,
            group_offsets_,
            group_counts_,
            group_shapes_,
            plan_ncoef_,
            complex_output_);
    }

    TORCH_CHECK(!image.is_complex(), "Monogenic transform requires real-valued input");
    const int64_t n = ndim();
    TORCH_CHECK(image.dim() >= n, "Expected input with at least ", n, " dims");
    for (int64_t i = 0; i < n; ++i) {
        TORCH_CHECK(
            image.size(image.dim() - n + i) == shape_[static_cast<size_t>(i)],
            "Input trailing shape does not match UDCT shape");
    }

    std::vector<int64_t> dims(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        dims[static_cast<size_t>(i)] = -n + i;
    }

    const torch::Tensor filters = torch::view_as_complex(riesz_filters_);
    const torch::Tensor spectrum = torch::fft::fftn(image, std::nullopt, dims);
    const torch::Tensor riesz = torch::fft::ifftn(
        spectrum.unsqueeze(-n - 1) * filters, std::nullopt, dims);

    std::vector<torch::Tensor> parts = {image.unsqueeze(-n - 1), torch::real(riesz)};
    if (!even_axes_.empty()) {
        const torch::Tensor even_idx = torch::tensor(
            even_axes_, torch::TensorOptions().dtype(torch::kLong).device(image.device()));
        parts.push_back(torch::imag(riesz).index_select(-n - 1, even_idx));
    }
    const torch::Tensor images = torch::cat(parts, -n - 1);

    const torch::Tensor coeffs = detail::udct_forward(
        images,
        fwd_row_ptr_,
        fwd_col_,
        fwd_weights_,
        bwd_row_ptr_,
        bwd_col_,
        bwd_weights_,
        shape_,
        group_offsets_,
        group_counts_,
        group_shapes_,
        plan_ncoef_,
        complex_output_);

    const torch::Tensor scalar = coeffs.select(-2, 0);
    std::vector<torch::Tensor> channels = {torch::real(scalar), torch::imag(scalar)};
    for (int64_t k = 0; k < n; ++k) {
        torch::Tensor riesz_k = torch::real(coeffs.select(-2, 1 + k));
        int64_t even_pos = -1;
        for (size_t idx = 0; idx < even_axes_.size(); ++idx) {
            if (even_axes_[idx] == k) {
                even_pos = static_cast<int64_t>(idx);
                break;
            }
        }
        if (even_pos >= 0) {
            riesz_k = riesz_k - torch::imag(coeffs.select(-2, 1 + n + even_pos));
        }
        channels.push_back(riesz_k);
    }
    const torch::Tensor stacked = torch::stack(channels, -2);
    return stacked.flatten(-2).index_select(-1, monogenic_perm_);
}

torch::Tensor UDCTImpl::backward(const torch::Tensor& vector) {
    if (transform_kind_ != "monogenic") {
        return detail::udct_backward(
            vector,
            fwd_row_ptr_,
            fwd_col_,
            fwd_weights_,
            bwd_row_ptr_,
            bwd_col_,
            bwd_weights_,
            shape_,
            group_offsets_,
            group_counts_,
            group_shapes_,
            plan_ncoef_,
            complex_output_);
    }

    TORCH_CHECK(!vector.is_complex(), "Monogenic coefficients must be real-valued");
    TORCH_CHECK(
        vector.dim() >= 1 && vector.size(-1) == total_ncoef_,
        "Expected monogenic coefficients with trailing dim ", total_ncoef_,
        ", got ", vector.sizes());

    const int64_t n = ndim();
    const torch::Tensor channels = vector
        .index_select(-1, monogenic_inverse_)
        .unflatten(-1, {n + 2, plan_ncoef_});
    const torch::Tensor scalar = torch::complex(
        channels.select(-2, 0), channels.select(-2, 1));
    const torch::Tensor riesz = channels.narrow(-2, 2, n);
    const torch::Tensor real_in = torch::cat(
        {scalar.unsqueeze(-2), torch::complex(riesz, torch::zeros_like(riesz))}, -2);

    torch::Tensor images = detail::udct_backward(
        real_in,
        fwd_row_ptr_,
        fwd_col_,
        fwd_weights_,
        bwd_row_ptr_,
        bwd_col_,
        bwd_weights_,
        shape_,
        group_offsets_,
        group_counts_,
        group_shapes_,
        plan_ncoef_,
        complex_output_);

    images = images.movedim(-n - 1, 0);
    return torch::cat({images.narrow(0, 0, 1), -images.narrow(0, 1, n)}, 0);
}

void UDCTImpl::to(torch::Dtype dtype, bool non_blocking) {
    torch::NoGradGuard no_grad;
    for (auto& entry : named_buffers(/*recurse=*/false)) {
        if (entry.value().is_floating_point()) {
            entry.value().set_data(entry.value().to(dtype, non_blocking));
        }
    }
}

void UDCTImpl::to(torch::Device device, torch::Dtype dtype, bool non_blocking) {
    torch::nn::Module::to(device, non_blocking);
    to(dtype, non_blocking);
}

} // namespace curvelets
