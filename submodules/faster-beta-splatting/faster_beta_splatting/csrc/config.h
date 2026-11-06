// faster-beta-splatting: fused-Adam beta-kernel rasterizer for online
// incremental reconstruction. Derived from Faster-GS (Hahlbohm et al.,
// CVPR 2026, Apache-2.0); beta kernel and spherical-beta colour from
// Deformable Beta Splatting (Liu et al., SIGGRAPH 2025). See README.md.
#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace faster_beta::config {
    DEF bool debug = false;
    // rendering constants
    DEF float dilation = 0.3f;
    // AA opacity compensation h = sqrt(det/det_dilated), clamped below at
    // this ratio for numerical stability
    DEF float aa_scaling_min = 0.000025f;
    DEF float min_cov2d_determinant = 1e-6f;
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp;
    // alpha = min(0.99, opacity * weight) keeps the backward's 1/(1-alpha)
    // bounded (<= 100) for near-opaque primitives
    DEF float max_alpha = 0.99f;
    DEF float one_minus_alpha_eps = 1e-6f;
    DEF float transmittance_threshold = 1e-4f;
    // beta kernel: support ends at power = beta_support (u = power / beta_support)
    DEF float beta_support = 4.5f; // 0.5 * 3^2 (compact support at 3 sigma)
    DEF float beta_support_rcp = 1.0f / beta_support;
    DEF float one_minus_u_eps = 1e-6f;
    // block size constants
    DEF int block_size_preprocess = 128;
    DEF int block_size_preprocess_backward = 128;
    DEF int block_size_apply_depth_ordering = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int block_size_extract_bucket_counts = 256;
    DEF int tile_width = 16;
    DEF int tile_height = 16;
    DEF int block_size_blend = tile_width * tile_height;
    DEF int n_sequential_threshold = 4;
    // optimization constants (learning rates and Adam betas are runtime
    // parameters; only the denominator epsilon is fixed)
    DEF float epsilon = 1e-15f;
    // SH colour is always the deg-3 storage layout [N, 15, 3]
    DEF int n_sh_bases_rest = 15;
}

namespace config = faster_beta::config;

#undef DEF
