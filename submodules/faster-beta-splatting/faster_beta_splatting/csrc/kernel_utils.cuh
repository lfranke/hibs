// Derived from Faster-GS (Apache-2.0); see README.md.
#pragma once

#include "config.h"
#include "helper_math.h"
#include "buffer_utils.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace faster_beta::kernels {

    __device__ inline float sigmoid(const float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    // Beta kernel weight: w(power) = (1 - power/4.5)^b for power in [0, 4.5).
    // power = 0.5 * d^T Sigma^-1 d (positive convention).
    __device__ inline float beta_weight(const float power, const float b) {
        const float one_minus_u = 1.0f - power * config::beta_support_rcp;
        if (one_minus_u <= 0.0f) return 0.0f;
        return __powf(one_minus_u, b);
    }

    // Largest power with alpha >= 1/255: (1-u)^b = 1/(255 o)
    //   -> u_cut = 1 - (255 o)^(-1/b) -> power_cut = 4.5 * u_cut
    __device__ inline float beta_power_threshold(const float opacity, const float b) {
        const float cut = opacity * config::min_alpha_threshold_rcp; // 255 o >= 1 guaranteed by caller
        const float u_cut = 1.0f - __powf(cut, -1.0f / b);
        return config::beta_support * fmaxf(u_cut, 0.0f);
    }

    __device__ inline mat3x3 convert_quaternion_to_rotation_matrix(
        const float4& quaternion,
        float& norm_sq)
    {
        auto [r, x, y, z] = quaternion;
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float rx = r * x, ry = r * y, rz = r * z;
        norm_sq = r * r + xx + yy + zz;
        const float norm_sq_rcp = 1.0f / norm_sq;
        return {
            1.0f - 2.0f * (yy + zz) * norm_sq_rcp, 2.0f * (xy - rz) * norm_sq_rcp, 2.0f * (xz + ry) * norm_sq_rcp,
            2.0f * (xy + rz) * norm_sq_rcp, 1.0f - 2.0f * (xx + zz) * norm_sq_rcp, 2.0f * (yz - rx) * norm_sq_rcp,
            2.0f * (xz - ry) * norm_sq_rcp, 2.0f * (yz + rx) * norm_sq_rcp, 1.0f - 2.0f * (xx + yy) * norm_sq_rcp
        };
    }

    __device__ inline float4 convert_quaternion_to_rotation_matrix_backward(
        const float4& quaternion,
        const mat3x3& dL_dR)
    {
        auto [r, x, y, z] = quaternion;
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float rx = r * x, ry = r * y, rz = r * z;
        const float norm_sq = r * r + xx + yy + zz;
        const float norm_sq_rcp = 1.0f / norm_sq;
        const float dL_dxx = dL_dR.m22 + dL_dR.m33;
        const float dL_dyy = dL_dR.m11 + dL_dR.m33;
        const float dL_dzz = dL_dR.m11 + dL_dR.m22;
        const float dL_drz = dL_dR.m21 - dL_dR.m12;
        const float dL_dxy = dL_dR.m21 + dL_dR.m12;
        const float dL_dry = dL_dR.m13 - dL_dR.m31;
        const float dL_dxz = dL_dR.m13 + dL_dR.m31;
        const float dL_drx = dL_dR.m32 - dL_dR.m23;
        const float dL_dyz = dL_dR.m32 + dL_dR.m23;
        const float two_over_norm_sq = 2.0f * norm_sq_rcp;
        const float dL_dnorm_helper = two_over_norm_sq * (xy * dL_dxy + xz * dL_dxz + yz * dL_dyz + rx * dL_drx + ry * dL_dry + rz * dL_drz - xx * dL_dxx - yy * dL_dyy - zz * dL_dzz);
        return two_over_norm_sq * make_float4(
            x * dL_drx + y * dL_dry + z * dL_drz - r * dL_dnorm_helper,
            r * dL_drx - 2.0f * x * dL_dxx + y * dL_dxy + z * dL_dxz - x * dL_dnorm_helper,
            r * dL_dry + x * dL_dxy - 2.0f * y * dL_dyy + z * dL_dyz - y * dL_dnorm_helper,
            r * dL_drz + x * dL_dxz + y * dL_dyz - 2.0f * z * dL_dzz - z * dL_dnorm_helper
        );
    }

    // based on StopThePop (see Faster-GS attribution)
    __device__ inline bool will_primitive_contribute(
        const float2& mean,
        const float3& conic,
        const uint tile_x,
        const uint tile_y,
        const float power_threshold)
    {
        const float2 rect_min = make_float2(static_cast<float>(tile_x * config::tile_width), static_cast<float>(tile_y * config::tile_height));
        const float2 rect_max = make_float2(static_cast<float>((tile_x + 1) * config::tile_width - 1), static_cast<float>((tile_y + 1) * config::tile_height - 1));

        const float x_min_diff = rect_min.x - mean.x;
        const float x_left = static_cast<float>(x_min_diff >= 0.0f);
        const float not_in_x_range = x_left + static_cast<float>(mean.x > rect_max.x);

        const float y_min_diff = rect_min.y - mean.y;
        const float y_above = static_cast<float>(y_min_diff >= 0.0f);
        const float not_in_y_range = y_above + static_cast<float>(mean.y > rect_max.y);

        if (not_in_y_range + not_in_x_range == 0.0f) return true;
        else {
            const float2 closest_corner = make_float2(
                lerp(rect_max.x, rect_min.x, x_left),
                lerp(rect_max.y, rect_min.y, y_above)
            );

            const float2 diff = mean - closest_corner;

            const float2 d = make_float2(
                copysignf(static_cast<float>(config::tile_width - 1), x_min_diff),
                copysignf(static_cast<float>(config::tile_height - 1), y_min_diff)
            );

            const float2 t = make_float2(
                not_in_y_range * __saturatef((d.x * conic.x * diff.x + d.x * conic.y * diff.y) / (d.x * conic.x * d.x)),
                not_in_x_range * __saturatef((d.y * conic.y * diff.x + d.y * conic.z * diff.y) / (d.y * conic.z * d.y))
            );

            const float2 max_contribution_point = closest_corner + t * d;
            const float2 delta = mean - max_contribution_point;
            const float max_power_in_tile = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            return max_power_in_tile <= power_threshold;
        }
    }

    // based on StopThePop (see Faster-GS attribution)
    __device__ inline uint compute_exact_n_touched_tiles(
        const float2& mean2d,
        const float3& conic,
        const uint4& screen_bounds,
        const float power_threshold,
        const uint tile_count,
        const bool active)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint lane_idx = warp.thread_rank();

        const float2 mean2d_shifted = mean2d - 0.5f;

        uint n_touched_tiles = 0;
        const uint screen_bounds_width = screen_bounds.y - screen_bounds.x;
        for (uint instance_idx = 0; active && instance_idx < tile_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) n_touched_tiles++;
        }

        const bool compute_cooperatively = active && tile_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (remaining_threads == 0) return n_touched_tiles;

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);

            const uint2 min_screen_bounds_coop = make_uint2(
                warp.shfl(screen_bounds.x, current_lane),
                warp.shfl(screen_bounds.z, current_lane)
            );
            const uint screen_bounds_width_coop = warp.shfl(screen_bounds_width, current_lane);
            const uint tile_count_coop = warp.shfl(tile_count, current_lane);

            const float2 mean2d_shifted_coop = make_float2(
                warp.shfl(mean2d_shifted.x, current_lane),
                warp.shfl(mean2d_shifted.y, current_lane)
            );
            const float3 conic_coop = make_float3(
                warp.shfl(conic.x, current_lane),
                warp.shfl(conic.y, current_lane),
                warp.shfl(conic.z, current_lane)
            );
            const float power_threshold_coop = warp.shfl(power_threshold, current_lane);

            const uint remaining_tile_count = tile_count_coop - config::n_sequential_threshold;
            const uint n_iterations = div_round_up(remaining_tile_count, warp_size);
            for (uint i = 0; i < n_iterations; i++) {
                const uint instance_idx = i * warp_size + lane_idx + config::n_sequential_threshold;
                const uint tile_x = min_screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                const uint tile_y = min_screen_bounds_coop.y + (instance_idx / screen_bounds_width_coop);
                const bool contributes = instance_idx < tile_count_coop && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint contributes_ballot = warp.ballot(contributes);
                const uint n_contributes = __popc(contributes_ballot);
                n_touched_tiles += (current_lane == lane_idx) * n_contributes;
            }
        }

        return n_touched_tiles;
    }

    // Fused Adam step, or a raw-gradient store when grads_out is given (test
    // mode / external optimizers). One thread owns each primitive, so the
    // gradient store needs no atomics.
    //
    // Runtime-index variant, for parameter layouts whose per-element offset
    // is not a compile-time constant (the spherical-beta lobe loop). The two
    // moments live in SEPARATE exp_avg / exp_avg_sq arrays, so an external
    // optimizer's existing state tensors can be passed straight in.
    __device__ inline void apply_update_dyn(
        const float grad,
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        float* grads_out,
        const uint element_idx,
        const float step_size,
        const float adam_beta1,
        const float adam_beta2)
    {
        if (grads_out != nullptr) {
            grads_out[element_idx] = grad;
            return;
        }
        const float grad_sq = grad * grad;
        const float moment1 = fmaf(adam_beta1, exp_avg[element_idx] - grad, grad);
        const float moment2 = fmaf(adam_beta2, exp_avg_sq[element_idx] - grad_sq, grad_sq);
        const float denom = sqrtf(moment2) + config::epsilon;
        param[element_idx] -= step_size * moment1 / denom;
        exp_avg[element_idx] = moment1;
        exp_avg_sq[element_idx] = moment2;
    }

    // Interleaved-moment variant: exp_avg and exp_avg_sq of one parameter
    // live ADJACENT in a single float2 pool, so the pair is one 8-byte load
    // and one 8-byte store instead of two 4-byte accesses into two separate
    // streams. The arithmetic below is identical to apply_update_dyn, in the
    // same order, so the two produce bit-identical results.
    __device__ inline void apply_update_dyn_interleaved(
        const float grad,
        float* param,
        float2* moments,
        float* grads_out,
        const uint element_idx,
        const float step_size,
        const float adam_beta1,
        const float adam_beta2)
    {
        if (grads_out != nullptr) {
            grads_out[element_idx] = grad;
            return;
        }
        const float grad_sq = grad * grad;
        const float2 m = moments[element_idx];
        const float moment1 = fmaf(adam_beta1, m.x - grad, grad);
        const float moment2 = fmaf(adam_beta2, m.y - grad_sq, grad_sq);
        const float denom = sqrtf(moment2) + config::epsilon;
        param[element_idx] -= step_size * moment1 / denom;
        moments[element_idx] = make_float2(moment1, moment2);
    }

    // Picks the moment layout at compile time; the caller instantiates on
    // whether an interleaved pool was handed in.
    template <bool interleaved_moments>
    __device__ inline void apply_update_dyn_any(
        const float grad,
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        float2* moments,
        float* grads_out,
        const uint element_idx,
        const float step_size,
        const float adam_beta1,
        const float adam_beta2)
    {
        if constexpr (interleaved_moments) {
            apply_update_dyn_interleaved(grad, param, moments, grads_out, element_idx, step_size, adam_beta1, adam_beta2);
        } else {
            apply_update_dyn(grad, param, exp_avg, exp_avg_sq, grads_out, element_idx, step_size, adam_beta1, adam_beta2);
        }
    }

    // Adam betas are runtime parameters: the i3dgs pipeline trains its
    // primitives with (0.5, 0.99), not the 3DGS-default (0.9, 0.999).
    template <uint n_attributes, uint offset>
    __device__ inline void apply_update(
        const float grad,
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        float* grads_out,
        const uint primitive_idx,
        const float step_size,
        const float adam_beta1,
        const float adam_beta2)
    {
        apply_update_dyn(grad, param, exp_avg, exp_avg_sq, grads_out,
                         primitive_idx * n_attributes + offset,
                         step_size, adam_beta1, adam_beta2);
    }

}
