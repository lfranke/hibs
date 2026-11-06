// Derived from Faster-GS (Apache-2.0); see README.md.
// Launch orchestration + torch wrappers for the fused-Adam beta rasterizer.
#include "config.h"
#include "buffer_utils.h"
#include "kernels_forward.cuh"
#include "kernels_backward.cuh"
#include "torch_utils.h"
#include "utils.h"
#include "helper_math.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <functional>
#include <tuple>

namespace faster_beta {

// Per-primitive size of the sh_rest buffer in FLOATS, taken from the tensor
// itself: 45 for the deg-3 SH storage [N, 15, 3] and 6K for a tight lobe
// buffer [N, K, 6]. Also checks the layout against the colour model.
static int check_sh_rest(const torch::Tensor& sh_rest, const int active_sh_bases) {
    TORCH_CHECK(sh_rest.dim() == 3, "sh_rest must be [N, rows, cols], got ", sh_rest.sizes());
    const int rows = sh_rest.size(1), cols = sh_rest.size(2);
    if (active_sh_bases >= 0) {
        TORCH_CHECK(rows == config::n_sh_bases_rest && cols == 3,
                    "SH colour needs sh_rest [N, 15, 3], got ", sh_rest.sizes());
    } else {
        const int n_lobes = -active_sh_bases;
        if (cols == 6) {
            TORCH_CHECK(rows == n_lobes, "a tight lobe buffer must be [N, ", n_lobes,
                        ", 6] for active_sh_bases = ", active_sh_bases, ", got ", sh_rest.sizes());
        } else {
            TORCH_CHECK(rows * cols >= 6 * n_lobes,
                        "sh_rest holds ", rows * cols, " floats per primitive, too few for ",
                        n_lobes, " lobes of 6");
        }
    }
    return rows * cols;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           int, int, int>
forward_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& raw_betas,
    const torch::Tensor& sh_coefficients_0,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c,           // [3, 4] row-major world-to-camera
    const torch::Tensor& cam_position,  // [3]
    const torch::Tensor& bg_color,      // [3, H, W]
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool want_invdepth,
    const bool want_main_ids)
{
    CHECK_INPUT(config::debug, means, "means");
    CHECK_INPUT(config::debug, scales, "scales");
    CHECK_INPUT(config::debug, rotations, "rotations");
    CHECK_INPUT(config::debug, opacities, "opacities");
    CHECK_INPUT(config::debug, raw_betas, "raw_betas");
    CHECK_INPUT(config::debug, sh_coefficients_0, "sh_coefficients_0");
    CHECK_INPUT(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

    const int n_primitives = means.size(0);
    const int sh_rest_stride = check_sh_rest(sh_coefficients_rest, active_sh_bases);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions int_options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({3, height, width}, float_options);
    torch::Tensor invdepth = want_invdepth ? torch::zeros({1, height, width}, float_options)
                                           : torch::empty({0}, float_options);
    torch::Tensor main_ids = want_main_ids ? torch::full({2, height, width}, -1, int_options)
                                           : torch::empty({0}, int_options);
    torch::Tensor final_transmittance = torch::ones({height, width}, float_options);
    torch::Tensor visibility = torch::zeros({n_primitives}, int_options);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    torch::Tensor bucket_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);
    const std::function<char*(size_t)> resize_bucket_buffers = resize_function_wrapper(bucket_buffers);

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    char* tile_buffers_blob = resize_tile_buffers(required<TileBuffers>(n_tiles));
    TileBuffers tb = TileBuffers::from_blob(tile_buffers_blob, n_tiles);
    cudaMemset(tb.instance_ranges, 0, sizeof(uint2) * n_tiles);

    char* primitive_buffers_blob = resize_primitive_buffers(required<PrimitiveBuffers>(n_primitives));
    PrimitiveBuffers pb = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    cudaMemset(pb.n_visible_primitives, 0, sizeof(uint));
    cudaMemset(pb.n_instances, 0, sizeof(uint));

    kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, 0, stream>>>(
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        raw_betas.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        pb.depth_keys.Current(),
        pb.primitive_indices.Current(),
        pb.n_touched_tiles,
        pb.screen_bounds,
        pb.mean2d,
        pb.conic_opacity,
        pb.color,
        pb.beta,
        pb.depth,
        pb.power_threshold,
        visibility.data_ptr<int>(),
        pb.n_visible_primitives,
        pb.n_instances,
        n_primitives,
        grid.x,
        grid.y,
        active_sh_bases,
        sh_rest_stride,
        static_cast<float>(width),
        static_cast<float>(height),
        focal_x, focal_y, center_x, center_y,
        near_plane, far_plane
    );
    CHECK_CUDA(config::debug, "preprocess")

    int n_visible_primitives;
    int n_instances;
    cudaMemcpyAsync(&n_visible_primitives, pb.n_visible_primitives, sizeof(uint),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&n_instances, pb.n_instances, sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (n_visible_primitives > 0) {
        cub::DeviceRadixSort::SortPairs(
            pb.cub_workspace, pb.cub_workspace_size,
            pb.depth_keys, pb.primitive_indices,
            n_visible_primitives, 0, sizeof(uint) * 8, stream
        );
        CHECK_CUDA(config::debug, "radix sort (depth)")

        kernels::forward::apply_depth_ordering_cu<<<div_round_up(n_visible_primitives, config::block_size_apply_depth_ordering), config::block_size_apply_depth_ordering, 0, stream>>>(
            pb.primitive_indices.Current(),
            pb.n_touched_tiles,
            pb.offset,
            n_visible_primitives
        );
        CHECK_CUDA(config::debug, "apply_depth_ordering")

        cub::DeviceScan::ExclusiveSum(
            pb.cub_workspace, pb.cub_workspace_size,
            pb.offset, pb.offset,
            n_visible_primitives, stream
        );
        CHECK_CUDA(config::debug, "exclusive sum (offsets)")
    }

    char* instance_buffers_blob = resize_instance_buffers(required<InstanceBuffers>(std::max(n_instances, 1), end_bit));
    InstanceBuffers ib = InstanceBuffers::from_blob(instance_buffers_blob, std::max(n_instances, 1), end_bit);

    if (n_visible_primitives > 0) {
        kernels::forward::create_instances_cu<<<div_round_up(n_visible_primitives, config::block_size_create_instances), config::block_size_create_instances, 0, stream>>>(
            pb.primitive_indices.Current(),
            pb.offset,
            pb.screen_bounds,
            pb.mean2d,
            pb.conic_opacity,
            pb.power_threshold,
            ib.keys.Current(),
            ib.primitive_indices.Current(),
            grid.x,
            n_visible_primitives
        );
        CHECK_CUDA(config::debug, "create_instances")
    }

    if (n_instances > 0) {
        cub::DeviceRadixSort::SortPairs(
            ib.cub_workspace, ib.cub_workspace_size,
            ib.keys, ib.primitive_indices,
            n_instances,
            0, end_bit, stream
        );
        CHECK_CUDA(config::debug, "radix sort (tile)")

        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, stream>>>(
            ib.keys.Current(),
            tb.instance_ranges,
            n_instances
        );
        CHECK_CUDA(config::debug, "extract_instance_ranges")
    }

    kernels::forward::extract_bucket_counts<<<div_round_up(n_tiles, config::block_size_extract_bucket_counts), config::block_size_extract_bucket_counts, 0, stream>>>(
        tb.instance_ranges,
        tb.n_buckets,
        n_tiles
    );
    CHECK_CUDA(config::debug, "extract_bucket_counts")

    cub::DeviceScan::InclusiveSum(
        tb.cub_workspace, tb.cub_workspace_size,
        tb.n_buckets, tb.buckets_offset,
        n_tiles, stream
    );
    CHECK_CUDA(config::debug, "inclusive sum (buckets)")

    int n_buckets;
    cudaMemcpyAsync(&n_buckets, tb.buckets_offset + n_tiles - 1, sizeof(uint),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    char* bucket_buffers_blob = resize_bucket_buffers(required<BucketBuffers>(std::max(n_buckets, 1)));
    BucketBuffers bb = BucketBuffers::from_blob(bucket_buffers_blob, std::max(n_buckets, 1));

    const auto blend = want_invdepth
        ? (want_main_ids ? kernels::forward::blend_cu<true, true>
                         : kernels::forward::blend_cu<true, false>)
        : (want_main_ids ? kernels::forward::blend_cu<false, true>
                         : kernels::forward::blend_cu<false, false>);
    blend<<<grid, block, 0, stream>>>(
        tb.instance_ranges,
        tb.buckets_offset,
        ib.primitive_indices.Current(),
        pb.mean2d,
        pb.conic_opacity,
        pb.color,
        pb.beta,
        pb.depth,
        bg_color.contiguous().data_ptr<float>(),
        image.data_ptr<float>(),
        want_invdepth ? invdepth.data_ptr<float>() : nullptr,
        want_main_ids ? main_ids.data_ptr<int>() : nullptr,
        final_transmittance.data_ptr<float>(),
        tb.max_n_processed,
        tb.n_processed,
        bb.tile_index,
        bb.color_transmittance,
        width, height, grid.x
    );
    CHECK_CUDA(config::debug, "blend")

    return {
        image, invdepth, main_ids, final_transmittance, visibility,
        primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
        n_instances, n_buckets, ib.primitive_indices.selector
    };
}

// Backward + fused Adam. When with_grads is true, raw gradients are written
// to returned tensors and no Adam update is applied (test mode / external
// optimizers). Returns (grad_w2c, grads_dict_tensors...).
std::vector<torch::Tensor> backward_wrapper(
    torch::Tensor& means,
    torch::Tensor& scales,
    torch::Tensor& rotations,
    torch::Tensor& opacities,
    torch::Tensor& raw_betas,
    torch::Tensor& sh_coefficients_0,
    torch::Tensor& sh_coefficients_rest,
    torch::Tensor& exp_avg_means, torch::Tensor& exp_avg_sq_means,
    torch::Tensor& exp_avg_scales, torch::Tensor& exp_avg_sq_scales,
    torch::Tensor& exp_avg_rotations, torch::Tensor& exp_avg_sq_rotations,
    torch::Tensor& exp_avg_opacities, torch::Tensor& exp_avg_sq_opacities,
    torch::Tensor& exp_avg_betas, torch::Tensor& exp_avg_sq_betas,
    torch::Tensor& exp_avg_sh_coefficients_0, torch::Tensor& exp_avg_sq_sh_coefficients_0,
    torch::Tensor& exp_avg_sh_coefficients_rest, torch::Tensor& exp_avg_sq_sh_coefficients_rest,
    const torch::Tensor& grad_image,
    const torch::Tensor& image,
    const torch::Tensor& final_transmittance,
    const torch::Tensor& primitive_buffers,
    const torch::Tensor& tile_buffers,
    const torch::Tensor& instance_buffers,
    const torch::Tensor& bucket_buffers,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const torch::Tensor& lr_scale, // [N] per-primitive step scale, or empty
    const float lr_means,
    const float lr_scales,
    const float lr_rotations,
    const float lr_opacities,
    const float lr_betas,
    const float lr_sh_0,
    const float lr_sh_rest,
    const float opacity_reg,
    const float scale_reg,
    const float adam_beta1,
    const float adam_beta2,
    const int n_instances,
    const int n_buckets,
    const int instance_primitive_indices_selector,
    const bool w2c_requires_grad,
    const bool with_grads)
{
    const int n_primitives = means.size(0);
    const int sh_rest_stride = check_sh_rest(sh_coefficients_rest, active_sh_bases);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // Interleaved lobe moments: one float2 pool holding (exp_avg, exp_avg_sq)
    // adjacent per parameter, signalled by an empty exp_avg_sq tensor.
    const bool interleaved_sh_rest_moments =
        exp_avg_sh_coefficients_rest.numel() > 0 && exp_avg_sq_sh_coefficients_rest.numel() == 0;
    TORCH_CHECK(!(interleaved_sh_rest_moments && active_sh_bases >= 0),
                "interleaved sh_rest moments are only supported for the spherical-beta "
                "colour model (active_sh_bases < 0)");
    TORCH_CHECK(!interleaved_sh_rest_moments ||
                exp_avg_sh_coefficients_rest.numel() >= 2 * sh_coefficients_rest.numel(),
                "the interleaved sh_rest moment pool needs 2 * sh_rest.numel() elements");
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor grad_colors = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_opacities = torch::zeros({n_primitives, 1}, float_options);
    torch::Tensor grad_raw_betas_accum = torch::zeros({n_primitives, 1}, float_options);
    torch::Tensor grad_mean2d_helper = torch::zeros({n_primitives, 2}, float_options);
    torch::Tensor grad_conic_helper = torch::zeros({3, n_primitives}, float_options);
    torch::Tensor grad_w2c = torch::zeros({3, 4}, float_options);

    // optional raw-gradient outputs
    torch::Tensor go_means, go_scales, go_rotations, go_opacities, go_betas, go_sh0, go_shrest;
    if (with_grads) {
        go_means = torch::zeros_like(means);
        go_scales = torch::zeros_like(scales);
        go_rotations = torch::zeros_like(rotations);
        go_opacities = torch::zeros_like(opacities);
        go_betas = torch::zeros_like(raw_betas);
        go_sh0 = torch::zeros_like(sh_coefficients_0);
        go_shrest = torch::zeros_like(sh_coefficients_rest);
    }

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    char* primitive_buffers_blob = reinterpret_cast<char*>(primitive_buffers.data_ptr());
    char* tile_buffers_blob = reinterpret_cast<char*>(tile_buffers.data_ptr());
    char* instance_buffers_blob = reinterpret_cast<char*>(instance_buffers.data_ptr());
    char* bucket_buffers_blob = reinterpret_cast<char*>(bucket_buffers.data_ptr());
    PrimitiveBuffers pb = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    TileBuffers tb = TileBuffers::from_blob(tile_buffers_blob, n_tiles);
    InstanceBuffers ib = InstanceBuffers::from_blob(instance_buffers_blob, std::max(n_instances, 1), end_bit);
    BucketBuffers bb = BucketBuffers::from_blob(bucket_buffers_blob, std::max(n_buckets, 1));
    ib.primitive_indices.selector = instance_primitive_indices_selector;

    if (n_buckets > 0) {
        kernels::backward::blend_backward_cu<<<n_buckets, 32, 0, stream>>>(
            tb.instance_ranges,
            tb.buckets_offset,
            ib.primitive_indices.Current(),
            pb.mean2d,
            pb.conic_opacity,
            pb.color,
            pb.beta,
            bg_color.contiguous().data_ptr<float>(),
            grad_image.contiguous().data_ptr<float>(),
            image.data_ptr<float>(),
            final_transmittance.data_ptr<float>(),
            tb.max_n_processed,
            tb.n_processed,
            bb.tile_index,
            bb.color_transmittance,
            reinterpret_cast<float2*>(grad_mean2d_helper.data_ptr<float>()),
            grad_conic_helper.data_ptr<float>(),
            grad_opacities.data_ptr<float>(),
            grad_raw_betas_accum.data_ptr<float>(),
            reinterpret_cast<float3*>(grad_colors.data_ptr<float>()),
            n_primitives,
            width, height, grid.x
        );
        CHECK_CUDA(config::debug, "blend_backward")
    }

    // keep the (possibly copied) per-primitive step-scale storage alive across the async launch
    const torch::Tensor lr_scale_c = lr_scale.numel() > 0 ? lr_scale.contiguous() : lr_scale;

    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward, 0, stream>>>(
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        raw_betas.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        exp_avg_means.data_ptr<float>(), exp_avg_sq_means.data_ptr<float>(),
        exp_avg_scales.data_ptr<float>(), exp_avg_sq_scales.data_ptr<float>(),
        exp_avg_rotations.data_ptr<float>(), exp_avg_sq_rotations.data_ptr<float>(),
        exp_avg_opacities.data_ptr<float>(), exp_avg_sq_opacities.data_ptr<float>(),
        exp_avg_betas.data_ptr<float>(), exp_avg_sq_betas.data_ptr<float>(),
        exp_avg_sh_coefficients_0.data_ptr<float>(), exp_avg_sq_sh_coefficients_0.data_ptr<float>(),
        interleaved_sh_rest_moments ? nullptr : exp_avg_sh_coefficients_rest.data_ptr<float>(),
        interleaved_sh_rest_moments ? nullptr : exp_avg_sq_sh_coefficients_rest.data_ptr<float>(),
        interleaved_sh_rest_moments ? reinterpret_cast<float2*>(exp_avg_sh_coefficients_rest.data_ptr<float>()) : nullptr,
        with_grads ? go_means.data_ptr<float>() : nullptr,
        with_grads ? go_scales.data_ptr<float>() : nullptr,
        with_grads ? go_rotations.data_ptr<float>() : nullptr,
        with_grads ? go_opacities.data_ptr<float>() : nullptr,
        with_grads ? go_betas.data_ptr<float>() : nullptr,
        with_grads ? go_sh0.data_ptr<float>() : nullptr,
        with_grads ? go_shrest.data_ptr<float>() : nullptr,
        w2c_requires_grad ? grad_w2c.data_ptr<float>() : nullptr,
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        pb.n_touched_tiles,
        reinterpret_cast<float2*>(grad_mean2d_helper.data_ptr<float>()),
        grad_conic_helper.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        grad_raw_betas_accum.data_ptr<float>(),
        reinterpret_cast<float3*>(grad_colors.data_ptr<float>()),
        n_primitives,
        active_sh_bases,
        sh_rest_stride,
        static_cast<float>(width),
        static_cast<float>(height),
        focal_x, focal_y, center_x, center_y,
        lr_scale_c.numel() > 0 ? lr_scale_c.data_ptr<float>() : nullptr,
        lr_means,
        lr_scales,
        lr_rotations,
        lr_opacities,
        lr_betas,
        lr_sh_0,
        lr_sh_rest,
        opacity_reg,
        scale_reg,
        adam_beta1,
        adam_beta2
    );
    CHECK_CUDA(config::debug, "preprocess_backward")

    if (with_grads) {
        return {grad_w2c, go_means, go_scales, go_rotations, go_opacities, go_betas, go_sh0, go_shrest};
    }
    return {grad_w2c};
}

} // namespace faster_beta

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &faster_beta::forward_wrapper);
    m.def("backward", &faster_beta::backward_wrapper);
}
