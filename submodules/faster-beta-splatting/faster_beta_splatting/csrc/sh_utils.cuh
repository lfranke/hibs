// Derived from Faster-GS (Apache-2.0); see README.md.
// Colour models: deg-3 spherical harmonics and the spherical-beta lobes of
// Deformable Beta Splatting, plus their fused-Adam backwards.
#pragma once

#include "config.h"
#include "kernel_utils.cuh"
#include "helper_math.h"

namespace faster_beta::kernels {

    #define DEF inline constexpr float
    // degree 0
    DEF C0 = 0.28209479177387814f;
    // degree 1
    DEF C1 = 0.48860251190291987f;
    // degree 2
    DEF C2a = 1.0925484305920792f;
    DEF C2b = 0.94617469575755997f;
    DEF C2c = 0.31539156525251999f;
    DEF C2d = 0.54627421529603959f;
    DEF C2e = 1.8923493915151202f;
    // degree 3
    DEF C3a = 0.59004358992664352f;
    DEF C3b = 1.7701307697799304f;
    DEF C3c = 2.8906114426405538f;
    DEF C3d = 0.45704579946446572f;
    DEF C3e = 2.2852289973223288f;
    DEF C3f = 1.865881662950577f;
    DEF C3g = 1.1195289977703462f;
    DEF C3h = 1.4453057213202769f;
    DEF C3i = 3.5402615395598609f;
    DEF C3j = 4.5704579946446566f;
    DEF C3k = 5.597644988851731f;
    #undef DEF

    __device__ inline float3 convert_sh_to_color(
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint active_sh_bases)
    {
        float3 result = 0.5f + C0 * sh_coefficients_0[primitive_idx];
        if (active_sh_bases > 1) {
            const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * config::n_sh_bases_rest;
            auto [x, y, z] = normalize(position - cam_position);
            result = result - C1 * y * coefficients_ptr[0]
                            + C1 * z * coefficients_ptr[1]
                            - C1 * x * coefficients_ptr[2];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                result = result + C2a * xy * coefficients_ptr[3]
                                - C2a * yz * coefficients_ptr[4]
                                + (C2b * zz - C2c) * coefficients_ptr[5]
                                - C2a * xz * coefficients_ptr[6]
                                + C2d * (xx - yy) * coefficients_ptr[7];
                if (active_sh_bases > 9) {
                    result = result + y * (C3a * yy - C3b * xx) * coefficients_ptr[8]
                                    + C3c * xy * z * coefficients_ptr[9]
                                    + y * (C3d - C3e * zz) * coefficients_ptr[10]
                                    + z * (C3f * zz - C3g) * coefficients_ptr[11]
                                    + x * (C3d - C3e * zz) * coefficients_ptr[12]
                                    + C3h * z * (xx - yy) * coefficients_ptr[13]
                                    + x * (C3b * yy - C3a * xx) * coefficients_ptr[14];
                }
            }
        }
        return result;
    }

    // Spherical-beta color (Deformable Beta Splatting, Liu et al. 2025),
    // selected by a negative active_sh_bases = -K:
    //   C(v) = 0.5 + C0 * sh0 + sum_k a_k * max(0, dot(mu_k, v))^(4 exp(b_k))
    // K lobes of 6 floats (r, g, b, theta, phi, beta_raw) sit at the front of
    // the sh_rest buffer. sh_rest_stride is that buffer's per-primitive size
    // in FLOATS, taken from the tensor itself: 45 for the padded deg-3 SH
    // storage [N, 15, 3], 6K for the tight layout [N, K, 6]. The packing
    // within a lobe is the same either way.
    __device__ inline float3 convert_sb_to_color(
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const int n_lobes,
        const uint sh_rest_stride)
    {
        float3 result = 0.5f + C0 * sh_coefficients_0[primitive_idx];
        const float* p = reinterpret_cast<const float*>(sh_coefficients_rest) + primitive_idx * sh_rest_stride;
        const float3 dir = normalize(position - cam_position);
        for (int k = 0; k < n_lobes; k++) {
            const float* lp = p + 6 * k;
            float st, ct, sp, cp;
            __sincosf(lp[3], &st, &ct);
            __sincosf(lp[4], &sp, &cp);
            const float dt = st * cp * dir.x + st * sp * dir.y + ct * dir.z;
            if (dt > 0.0f) {
                const float w = __powf(dt, 4.0f * __expf(lp[5]));
                result += w * make_float3(lp[0], lp[1], lp[2]);
            }
        }
        return result;
    }

    // Backward for the spherical-beta color model. Applies the fused Adam
    // update to sh0 and the touched lobe floats and returns the position
    // gradient through the normalized view direction. The lobe moments are
    // either two separate arrays or one interleaved float2 pool, chosen by
    // the template argument.
    template <bool interleaved_moments>
    __device__ inline float3 convert_sb_to_color_backward(
        float3* __restrict__ sh_coefficients_0,
        float3* __restrict__ sh_coefficients_rest,
        float* __restrict__ exp_avg_sh_0,
        float* __restrict__ exp_avg_sq_sh_0,
        float* __restrict__ exp_avg_sh_rest,
        float* __restrict__ exp_avg_sq_sh_rest,
        float2* __restrict__ moments_sh_rest,
        float* __restrict__ grads_out_sh_0,
        float* __restrict__ grads_out_sh_rest,
        const float3* __restrict__ grad_colors,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const int n_lobes,
        const uint sh_rest_stride,
        const float step_size_sh_0,
        const float step_size_sh_rest,
        const float adam_beta1,
        const float adam_beta2)
    {
        const float3 grad_color = grad_colors[primitive_idx];

        const float3 dL_dsh_coefficients_0 = C0 * grad_color;
        apply_update<3, 0>(dL_dsh_coefficients_0.x, reinterpret_cast<float*>(sh_coefficients_0), exp_avg_sh_0, exp_avg_sq_sh_0, grads_out_sh_0, primitive_idx, step_size_sh_0, adam_beta1, adam_beta2);
        apply_update<3, 1>(dL_dsh_coefficients_0.y, reinterpret_cast<float*>(sh_coefficients_0), exp_avg_sh_0, exp_avg_sq_sh_0, grads_out_sh_0, primitive_idx, step_size_sh_0, adam_beta1, adam_beta2);
        apply_update<3, 2>(dL_dsh_coefficients_0.z, reinterpret_cast<float*>(sh_coefficients_0), exp_avg_sh_0, exp_avg_sq_sh_0, grads_out_sh_0, primitive_idx, step_size_sh_0, adam_beta1, adam_beta2);

        float* param = reinterpret_cast<float*>(sh_coefficients_rest);
        const uint base_idx = primitive_idx * sh_rest_stride;
        const float* p = param + base_idx;

        const float3 dir_orig = position - cam_position;
        const float3 dir = normalize(dir_orig);
        float3 dL_ddir = make_float3(0.0f);

        for (int k = 0; k < n_lobes; k++) {
            const float* lp = p + 6 * k;
            const uint li = base_idx + 6 * k;
            float st, ct, sp, cp;
            __sincosf(lp[3], &st, &ct);
            __sincosf(lp[4], &sp, &cp);
            const float3 mu = make_float3(st * cp, st * sp, ct);
            const float dt = dot(mu, dir);
            // dt <= 0: the lobe did not contribute; matching the reference,
            // its parameters receive zero gradient (and, unlike SH, no fused
            // update / moment decay -- the buffer floats are simply untouched).
            if (dt <= 0.0f) continue;

            const float bexp = 4.0f * __expf(lp[5]);
            const float w = __powf(dt, bexp);
            const float3 a = make_float3(lp[0], lp[1], lp[2]);

            apply_update_dyn_any<interleaved_moments>(w * grad_color.x, param, exp_avg_sh_rest, exp_avg_sq_sh_rest, moments_sh_rest, grads_out_sh_rest, li + 0, step_size_sh_rest, adam_beta1, adam_beta2);
            apply_update_dyn_any<interleaved_moments>(w * grad_color.y, param, exp_avg_sh_rest, exp_avg_sq_sh_rest, moments_sh_rest, grads_out_sh_rest, li + 1, step_size_sh_rest, adam_beta1, adam_beta2);
            apply_update_dyn_any<interleaved_moments>(w * grad_color.z, param, exp_avg_sh_rest, exp_avg_sq_sh_rest, moments_sh_rest, grads_out_sh_rest, li + 2, step_size_sh_rest, adam_beta1, adam_beta2);

            const float dL_dw = dot(a, grad_color);
            const float dL_ddt = dL_dw * bexp * __powf(dt, bexp - 1.0f);

            // dmu/dtheta = (ct cp, ct sp, -st); dmu/dphi = (-st sp, st cp, 0)
            const float g_theta = dL_ddt * (ct * cp * dir.x + ct * sp * dir.y - st * dir.z);
            const float g_phi = dL_ddt * (-st * sp * dir.x + st * cp * dir.y);
            const float g_braw = dL_dw * w * __logf(fmaxf(dt, 1e-8f)) * bexp;
            apply_update_dyn_any<interleaved_moments>(g_theta, param, exp_avg_sh_rest, exp_avg_sq_sh_rest, moments_sh_rest, grads_out_sh_rest, li + 3, step_size_sh_rest, adam_beta1, adam_beta2);
            apply_update_dyn_any<interleaved_moments>(g_phi, param, exp_avg_sh_rest, exp_avg_sq_sh_rest, moments_sh_rest, grads_out_sh_rest, li + 4, step_size_sh_rest, adam_beta1, adam_beta2);
            apply_update_dyn_any<interleaved_moments>(g_braw, param, exp_avg_sh_rest, exp_avg_sq_sh_rest, moments_sh_rest, grads_out_sh_rest, li + 5, step_size_sh_rest, adam_beta1, adam_beta2);

            dL_ddir += dL_ddt * mu;
        }

        // chain through dir = normalize(position - cam_position)
        const float xx = dir_orig.x * dir_orig.x, yy = dir_orig.y * dir_orig.y, zz = dir_orig.z * dir_orig.z;
        const float xy = dir_orig.x * dir_orig.y, xz = dir_orig.x * dir_orig.z, yz = dir_orig.y * dir_orig.z;
        const float norm_sq = xx + yy + zz;
        return make_float3(
            (yy + zz) * dL_ddir.x - xy * dL_ddir.y - xz * dL_ddir.z,
            -xy * dL_ddir.x + (xx + zz) * dL_ddir.y - yz * dL_ddir.z,
            -xz * dL_ddir.x - yz * dL_ddir.y + (xx + yy) * dL_ddir.z
        ) * rsqrtf(norm_sq * norm_sq * norm_sq);
    }

    // Backward through the SH color evaluation. Applies the fused Adam update
    // to sh_coefficients_0/rest (or stores raw gradients in test mode) and
    // returns the position gradient contributed through the view direction.
    __device__ inline float3 convert_sh_to_color_backward(
        float3* __restrict__ sh_coefficients_0,
        float3* __restrict__ sh_coefficients_rest,
        float* __restrict__ exp_avg_sh_0,
        float* __restrict__ exp_avg_sq_sh_0,
        float* __restrict__ exp_avg_sh_rest,
        float* __restrict__ exp_avg_sq_sh_rest,
        float* __restrict__ grads_out_sh_0,
        float* __restrict__ grads_out_sh_rest,
        const float3* __restrict__ grad_colors,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint active_sh_bases,
        const float step_size_sh_0,
        const float step_size_sh_rest,
        const float adam_beta1,
        const float adam_beta2)
    {
        const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * config::n_sh_bases_rest;
        const float3 grad_color = grad_colors[primitive_idx];

        const float3 dL_dsh_coefficients_0 = C0 * grad_color;
        apply_update<3, 0>(dL_dsh_coefficients_0.x, reinterpret_cast<float*>(sh_coefficients_0), exp_avg_sh_0, exp_avg_sq_sh_0, grads_out_sh_0, primitive_idx, step_size_sh_0, adam_beta1, adam_beta2);
        apply_update<3, 1>(dL_dsh_coefficients_0.y, reinterpret_cast<float*>(sh_coefficients_0), exp_avg_sh_0, exp_avg_sq_sh_0, grads_out_sh_0, primitive_idx, step_size_sh_0, adam_beta1, adam_beta2);
        apply_update<3, 2>(dL_dsh_coefficients_0.z, reinterpret_cast<float*>(sh_coefficients_0), exp_avg_sh_0, exp_avg_sq_sh_0, grads_out_sh_0, primitive_idx, step_size_sh_0, adam_beta1, adam_beta2);

        // one macro per rest coefficient: gradient = basis * grad_color
        #define FBS_SH_REST_UPDATE(BASIS_IDX, BASIS_VALUE) \
        { \
            const float3 dL_dc = (BASIS_VALUE) * grad_color; \
            apply_update<config::n_sh_bases_rest * 3, (BASIS_IDX) * 3 + 0>(dL_dc.x, reinterpret_cast<float*>(sh_coefficients_rest), exp_avg_sh_rest, exp_avg_sq_sh_rest, grads_out_sh_rest, primitive_idx, step_size_sh_rest, adam_beta1, adam_beta2); \
            apply_update<config::n_sh_bases_rest * 3, (BASIS_IDX) * 3 + 1>(dL_dc.y, reinterpret_cast<float*>(sh_coefficients_rest), exp_avg_sh_rest, exp_avg_sq_sh_rest, grads_out_sh_rest, primitive_idx, step_size_sh_rest, adam_beta1, adam_beta2); \
            apply_update<config::n_sh_bases_rest * 3, (BASIS_IDX) * 3 + 2>(dL_dc.z, reinterpret_cast<float*>(sh_coefficients_rest), exp_avg_sh_rest, exp_avg_sq_sh_rest, grads_out_sh_rest, primitive_idx, step_size_sh_rest, adam_beta1, adam_beta2); \
        }

        float3 dcolor_dposition = make_float3(0.0f);
        if (active_sh_bases > 1) {
            auto [x_raw, y_raw, z_raw] = position - cam_position;
            auto [x, y, z] = normalize(make_float3(x_raw, y_raw, z_raw));
            FBS_SH_REST_UPDATE(0, -C1 * y)
            FBS_SH_REST_UPDATE(1, C1 * z)
            FBS_SH_REST_UPDATE(2, -C1 * x)
            float3 grad_direction_x = -C1 * coefficients_ptr[2];
            float3 grad_direction_y = -C1 * coefficients_ptr[0];
            float3 grad_direction_z = C1 * coefficients_ptr[1];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                FBS_SH_REST_UPDATE(3, C2a * xy)
                FBS_SH_REST_UPDATE(4, -C2a * yz)
                FBS_SH_REST_UPDATE(5, C2b * zz - C2c)
                FBS_SH_REST_UPDATE(6, -C2a * xz)
                FBS_SH_REST_UPDATE(7, C2d * (xx - yy))
                grad_direction_x = grad_direction_x + C2a * y * coefficients_ptr[3]
                                                    - C2a * z * coefficients_ptr[6]
                                                    + C2a * x * coefficients_ptr[7];
                grad_direction_y = grad_direction_y + C2a * x * coefficients_ptr[3]
                                                    - C2a * z * coefficients_ptr[4]
                                                    - C2a * y * coefficients_ptr[7];
                grad_direction_z = grad_direction_z - C2a * y * coefficients_ptr[4]
                                                    + C2e * z * coefficients_ptr[5]
                                                    - C2a * x * coefficients_ptr[6];
                if (active_sh_bases > 9) {
                    FBS_SH_REST_UPDATE(8, y * (C3a * yy - C3b * xx))
                    FBS_SH_REST_UPDATE(9, C3c * xy * z)
                    FBS_SH_REST_UPDATE(10, y * (C3d - C3e * zz))
                    FBS_SH_REST_UPDATE(11, z * (C3f * zz - C3g))
                    FBS_SH_REST_UPDATE(12, x * (C3d - C3e * zz))
                    FBS_SH_REST_UPDATE(13, C3h * z * (xx - yy))
                    FBS_SH_REST_UPDATE(14, x * (C3b * yy - C3a * xx))
                    grad_direction_x = grad_direction_x - C3i * xy * coefficients_ptr[8]
                                                        + C3c * yz * coefficients_ptr[9]
                                                        + (C3d - C3e * zz) * coefficients_ptr[12]
                                                        + C3c * xz * coefficients_ptr[13]
                                                        + C3b * (yy - xx) * coefficients_ptr[14];
                    grad_direction_y = grad_direction_y + C3b * (yy - xx) * coefficients_ptr[8]
                                                        + C3c * xz * coefficients_ptr[9]
                                                        + (C3d - C3e * zz) * coefficients_ptr[10]
                                                        - C3c * yz * coefficients_ptr[13]
                                                        + C3i * xy * coefficients_ptr[14];
                    grad_direction_z = grad_direction_z + C3c * xy * coefficients_ptr[9]
                                                        - C3j * yz * coefficients_ptr[10]
                                                        + (C3k * zz - C3g) * coefficients_ptr[11]
                                                        - C3j * xz * coefficients_ptr[12]
                                                        + C3h * (xx - yy) * coefficients_ptr[13];
                }
            }

            const float3 grad_direction = make_float3(
                dot(grad_direction_x, grad_color),
                dot(grad_direction_y, grad_color),
                dot(grad_direction_z, grad_color)
            );
            const float xx_raw = x_raw * x_raw, yy_raw = y_raw * y_raw, zz_raw = z_raw * z_raw;
            const float xy_raw = x_raw * y_raw, xz_raw = x_raw * z_raw, yz_raw = y_raw * z_raw;
            const float norm_sq = xx_raw + yy_raw + zz_raw;
            dcolor_dposition = make_float3(
                (yy_raw + zz_raw) * grad_direction.x - xy_raw * grad_direction.y - xz_raw * grad_direction.z,
                -xy_raw * grad_direction.x + (xx_raw + zz_raw) * grad_direction.y - yz_raw * grad_direction.z,
                -xz_raw * grad_direction.x - yz_raw * grad_direction.y + (xx_raw + yy_raw) * grad_direction.z
            ) * rsqrtf(norm_sq * norm_sq * norm_sq);
        }
        #undef FBS_SH_REST_UPDATE
        return dcolor_dposition;
    }

}
