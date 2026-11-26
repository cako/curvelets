#include "curvelets/udct.hpp"
#include "udct_ops.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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
    num_directions_ = to_int64_vec(get_tensor_field(dict, "num_directions"));
    even_axes_ = to_int64_vec(get_tensor_field(dict, "even_axes"));

    const at::Tensor layout_t = get_tensor_field(dict, "layout").to(at::kCPU, at::kLong).contiguous();
    TORCH_CHECK(layout_t.dim() == 2 && layout_t.size(1) == 3, "Invalid UDCT plan layout shape");
    const int64_t num_entries = layout_t.size(0);
    const auto* layout_ptr = layout_t.const_data_ptr<int64_t>();
    layout_.reserve(static_cast<size_t>(num_entries));
    std::ranges::transform(
        std::views::iota(int64_t{0}, num_entries),
        std::back_inserter(layout_),
        [layout_ptr](int64_t i) {
            return LayoutEntry{
                layout_ptr[3 * i], layout_ptr[3 * i + 1], layout_ptr[3 * i + 2]};
        });

    const int64_t n = ndim();
    TORCH_CHECK(
        group_counts_.size() == group_offsets_.size() &&
            group_shapes_.size() == group_offsets_.size() * static_cast<size_t>(n),
        "Invalid UDCT plan group metadata");

    wedge_counts_.reserve(num_directions_.size());
    std::ranges::transform(
        num_directions_,
        std::back_inserter(wedge_counts_),
        [](int64_t count) { return std::vector<int64_t>(static_cast<size_t>(count), 0); });

    struct RawSpan {
        int64_t start;
        int64_t size;
        std::vector<int64_t> shape;
    };
    std::vector<std::vector<std::vector<RawSpan>>> spans;
    spans.reserve(num_directions_.size());
    std::ranges::transform(
        num_directions_,
        std::back_inserter(spans),
        [](int64_t count) {
            return std::vector<std::vector<RawSpan>>(static_cast<size_t>(count));
        });

    entry_shapes_.reserve(static_cast<size_t>(num_entries));
    int64_t entry_idx = 0;
    for (size_t g = 0; g < group_offsets_.size(); ++g) {
        const auto g_shape_span = std::span(group_shapes_).subspan(
            g * static_cast<size_t>(n), static_cast<size_t>(n));
        std::vector<int64_t> g_shape(g_shape_span.begin(), g_shape_span.end());
        const int64_t g_size = detail::product(g_shape_span);
        const int64_t count = group_counts_[g];
        for (int64_t i = 0; i < count; ++i) {
            TORCH_CHECK(entry_idx < num_entries, "UDCT group count exceeds layout entries");
            const LayoutEntry& e = layout_[static_cast<size_t>(entry_idx)];
            TORCH_CHECK(
                e.scale >= 0 && static_cast<size_t>(e.scale) < num_directions_.size() &&
                    e.direction >= 0 && e.direction < num_directions_[static_cast<size_t>(e.scale)],
                "UDCT layout entry out of bounds");
            wedge_counts_[static_cast<size_t>(e.scale)][static_cast<size_t>(e.direction)] += 1;
            entry_shapes_.push_back(g_shape);
            spans[static_cast<size_t>(e.scale)][static_cast<size_t>(e.direction)].push_back(
                RawSpan{group_offsets_[g] + i * g_size, g_size, g_shape});
            ++entry_idx;
        }
    }
    TORCH_CHECK(entry_idx == num_entries, "UDCT layout entries do not match group counts");

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

    if (transform_kind_ == "monogenic") {
        const int64_t num_channels = n + 2;
        mono_wedges_.reserve(spans.size());
        int64_t position = 0;
        for (const auto& scale_spans : spans) {
            auto& scale_wedges = mono_wedges_.emplace_back();
            scale_wedges.reserve(scale_spans.size());
            for (const auto& dir_spans : scale_spans) {
                auto& dir_wedges = scale_wedges.emplace_back();
                dir_wedges.reserve(dir_spans.size());
                std::ranges::transform(
                    dir_spans,
                    std::back_inserter(dir_wedges),
                    [&position, num_channels](const RawSpan& sp) {
                        const WedgeSpan wedge{position, sp.shape};
                        position += num_channels * sp.size;
                        return wedge;
                    });
            }
        }
        TORCH_CHECK(position == total_ncoef_, "Monogenic layout size mismatch in plan");
    }
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
    detail::check_trailing_shape(image, shape_, "input");

    const auto dims = detail::fft_dims(n);

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
    channels.reserve(static_cast<size_t>(n + 2));
    std::ranges::transform(
        std::views::iota(int64_t{0}, n),
        std::back_inserter(channels),
        [this, &coeffs, n](int64_t k) {
            torch::Tensor riesz_k = torch::real(coeffs.select(-2, 1 + k));
            if (const auto it = std::ranges::find(even_axes_, k); it != even_axes_.end()) {
                const auto even_pos = std::distance(even_axes_.begin(), it);
                riesz_k = riesz_k - torch::imag(coeffs.select(-2, 1 + n + even_pos));
            }
            return riesz_k;
        });
    const torch::Tensor stacked = torch::stack(channels, -2);
    return stacked.flatten(-2).index_select(-1, monogenic_perm_);
}

UDCTCoefficients UDCTImpl::forward_structured(const torch::Tensor& image) {
    return structure(forward(image));
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

torch::Tensor UDCTImpl::backward(const UDCTCoefficients& coefficients) {
    return backward(vect(coefficients));
}

UDCTCoefficients UDCTImpl::structure(const torch::Tensor& vector) const {
    TORCH_CHECK(
        vector.dim() >= 1 && vector.size(-1) == total_ncoef_,
        "Expected flat coefficients with trailing dim ", total_ncoef_,
        ", got ", vector.sizes());

    UDCTCoefficients coeffs;
    coeffs.reserve(num_directions_.size());
    std::ranges::transform(
        num_directions_,
        std::back_inserter(coeffs),
        [](int64_t count) {
            return std::vector<std::vector<torch::Tensor>>(static_cast<size_t>(count));
        });

    if (transform_kind_ == "monogenic") {
        const int64_t batch_ndim = vector.dim() - 1;
        const std::vector<int64_t> batch_dims(
            vector.sizes().begin(), vector.sizes().begin() + batch_ndim);
        const int64_t num_channels = ndim() + 2;
        for (size_t s = 0; s < mono_wedges_.size(); ++s) {
            for (size_t d = 0; d < mono_wedges_[s].size(); ++d) {
                coeffs[s][d].reserve(mono_wedges_[s][d].size());
                std::ranges::transform(
                    mono_wedges_[s][d],
                    std::back_inserter(coeffs[s][d]),
                    [&vector, &batch_dims, batch_ndim, num_channels](const WedgeSpan& w) {
                        const int64_t len = num_channels * detail::product(w.shape);
                        std::vector<int64_t> reshape_dims = batch_dims;
                        reshape_dims.push_back(num_channels);
                        reshape_dims.insert(reshape_dims.end(), w.shape.begin(), w.shape.end());
                        const torch::Tensor block =
                            vector.narrow(-1, w.start, len).reshape(reshape_dims);
                        return block.movedim(batch_ndim, -1);
                    });
            }
        }
        return coeffs;
    }

    const int64_t n = ndim();
    size_t entry_idx = 0;
    for (size_t g = 0; g < group_offsets_.size(); ++g) {
        const auto g_shape = std::span(group_shapes_).subspan(
            g * static_cast<size_t>(n), static_cast<size_t>(n));
        const int64_t g_size = detail::product(g_shape);
        const c10::IntArrayRef g_shape_ref(g_shape.data(), g_shape.size());
        const int64_t count = group_counts_[g];
        for (int64_t i = 0; i < count; ++i) {
            const int64_t start = group_offsets_[g] + i * g_size;
            const LayoutEntry& e = layout_[entry_idx++];
            coeffs[static_cast<size_t>(e.scale)][static_cast<size_t>(e.direction)].push_back(
                vector.narrow(-1, start, g_size).unflatten(-1, g_shape_ref));
        }
    }
    return coeffs;
}

torch::Tensor UDCTImpl::vect(const UDCTCoefficients& coefficients) const {
    TORCH_CHECK(
        coefficients.size() == num_directions_.size(),
        "Expected ", num_directions_.size(), " scales, got ", coefficients.size());
    for (size_t s = 0; s < num_directions_.size(); ++s) {
        TORCH_CHECK(
            coefficients[s].size() == static_cast<size_t>(num_directions_[s]),
            "Expected ", num_directions_[s], " directions at scale ", s,
            ", got ", coefficients[s].size());
        for (size_t d = 0; d < coefficients[s].size(); ++d) {
            TORCH_CHECK(
                static_cast<int64_t>(coefficients[s][d].size()) == wedge_counts_[s][d],
                "Expected ", wedge_counts_[s][d], " wedges at scale ", s,
                " direction ", d, ", got ", coefficients[s][d].size());
        }
    }

    std::vector<torch::Tensor> parts;
    std::optional<std::vector<int64_t>> expected_batch;

    if (transform_kind_ == "monogenic") {
        const int64_t num_channels = ndim() + 2;
        const int64_t trailing_ndim = ndim() + 1;
        for (size_t s = 0; s < mono_wedges_.size(); ++s) {
            for (size_t d = 0; d < mono_wedges_[s].size(); ++d) {
                for (size_t w = 0; w < mono_wedges_[s][d].size(); ++w) {
                    const torch::Tensor& coeff = coefficients[s][d][w];
                    std::vector<int64_t> expected_shape = mono_wedges_[s][d][w].shape;
                    expected_shape.push_back(num_channels);
                    detail::check_trailing_shape(coeff, expected_shape, "monogenic coefficients");
                    const int64_t batch_ndim = coeff.dim() - trailing_ndim;
                    const std::vector<int64_t> batch_dims(
                        coeff.sizes().begin(), coeff.sizes().begin() + batch_ndim);
                    if (!expected_batch) {
                        expected_batch = batch_dims;
                    } else {
                        TORCH_CHECK(
                            batch_dims == *expected_batch,
                            "Inconsistent batch dimensions across coefficient wedges");
                    }
                    std::vector<int64_t> flat_shape = batch_dims;
                    flat_shape.push_back(detail::product(expected_shape));
                    parts.push_back(coeff.movedim(-1, batch_ndim).reshape(flat_shape));
                }
            }
        }
        return torch::cat(parts, -1);
    }

    const int64_t n = ndim();
    parts.reserve(layout_.size());
    for (size_t i = 0; i < layout_.size(); ++i) {
        const LayoutEntry& e = layout_[i];
        const torch::Tensor& coeff = coefficients[static_cast<size_t>(e.scale)]
                                                 [static_cast<size_t>(e.direction)]
                                                 [static_cast<size_t>(e.wedge)];
        detail::check_trailing_shape(coeff, entry_shapes_[i], "coefficients");
        const int64_t batch_ndim = coeff.dim() - n;
        const std::vector<int64_t> batch_dims(
            coeff.sizes().begin(), coeff.sizes().begin() + batch_ndim);
        if (!expected_batch) {
            expected_batch = batch_dims;
        } else {
            TORCH_CHECK(
                batch_dims == *expected_batch,
                "Inconsistent batch dimensions across coefficient wedges");
        }
        std::vector<int64_t> flat_shape = batch_dims;
        flat_shape.push_back(detail::product(entry_shapes_[i]));
        parts.push_back(coeff.reshape(flat_shape));
    }
    return torch::cat(parts, -1);
}

UDCTCoefficientShapes UDCTImpl::coefficient_shapes() const {
    if (transform_kind_ == "monogenic") {
        const int64_t num_channels = ndim() + 2;
        UDCTCoefficientShapes shapes;
        shapes.reserve(mono_wedges_.size());
        for (const auto& scale_wedges : mono_wedges_) {
            auto& scale_shapes = shapes.emplace_back();
            scale_shapes.reserve(scale_wedges.size());
            for (const auto& dir_wedges : scale_wedges) {
                auto& dir_shapes = scale_shapes.emplace_back();
                dir_shapes.reserve(dir_wedges.size());
                std::ranges::transform(
                    dir_wedges,
                    std::back_inserter(dir_shapes),
                    [num_channels](const WedgeSpan& w) {
                        std::vector<int64_t> ws = w.shape;
                        ws.push_back(num_channels);
                        return ws;
                    });
            }
        }
        return shapes;
    }

    UDCTCoefficientShapes shapes;
    shapes.reserve(wedge_counts_.size());
    for (const auto& dir_counts : wedge_counts_) {
        auto& scale_shapes = shapes.emplace_back();
        scale_shapes.reserve(dir_counts.size());
        std::ranges::transform(
            dir_counts,
            std::back_inserter(scale_shapes),
            [](int64_t count) {
                return std::vector<std::vector<int64_t>>(static_cast<size_t>(count));
            });
    }

    for (size_t i = 0; i < layout_.size(); ++i) {
        const LayoutEntry& e = layout_[i];
        shapes[static_cast<size_t>(e.scale)]
              [static_cast<size_t>(e.direction)]
              [static_cast<size_t>(e.wedge)] = entry_shapes_[i];
    }
    return shapes;
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
