import gc
import math

import torch
import torch.nn.functional as F
from scene.mono_depth import align_depth
from utils import RGB2SH, depth2points, inverse_sigmoid, make_torch_sampler, sample

from hibs.kernel import (
    BETA_SOFT,
    BETA_SOLID,
    log_scale_compensation,
    observability,
    raw_from_beta,
)


def sample_prior(idepth, uv, width, height):
    grid = idepth.reshape(-1, *idepth.shape[-2:])[0]
    rows, cols = grid.shape
    x = ((uv[:, 0] + 0.5) * (cols / width) - 0.5).clamp(0.1, cols - 1.1)
    y = ((uv[:, 1] + 0.5) * (rows / height) - 0.5).clamp(0.1, rows - 1.1)
    x0, y0 = x.floor().long(), y.floor().long()
    fx, fy = (x - x0).unsqueeze(-1), (y - y0).unsqueeze(-1)
    top = grid[y0, x0].unsqueeze(-1) * (1 - fx) + grid[y0, x0 + 1].unsqueeze(-1) * fx
    bottom = grid[y0 + 1, x0].unsqueeze(-1) * (1 - fx) + grid[y0 + 1, x0 + 1].unsqueeze(-1) * fx
    return (top * (1 - fy) + bottom * fy).squeeze(-1)


class MvsDisagreement:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @torch.no_grad()
    def __call__(self, uv, ref_keyframe, keyframes, mono_idepth):
        depth, accurate = self._inner(uv, ref_keyframe, keyframes, mono_idepth)
        prior = sample_prior(mono_idepth, uv, ref_keyframe.width, ref_keyframe.height)
        prior = prior.clamp_min(self.idepth_range + 1e-6)
        # Eq. 5 in the paper
        distance = (1.0 / depth.clamp_min(1e-8) - prior).abs()
        return depth, torch.where(accurate, distance, torch.full_like(distance, -1.0))


@torch.no_grad()
def max_covisible_baseline(scene, keyframe_id, k=8):
    keyframe = scene.keyframes[keyframe_id]
    n_kf = len(scene.keyframes)
    others = [other for i, other in enumerate(keyframe.desc_kpts.matches)
              if i < k and 0 <= other < n_kf and other != keyframe.index]
    if not others:
        return 0.0
    centres = torch.stack([scene.keyframes[other].approx_centre for other in others])
    return float((centres - keyframe.approx_centre).norm(dim=-1).max())


def spawn_from_confidence(c, opacity, o_confirmed, beta_neutral,
                          beta_solid=BETA_SOLID, beta_soft=BETA_SOFT):
    # Eq. 6 in the paper
    beta = beta_soft + (beta_solid - beta_soft) * c
    neutral = torch.full_like(beta, beta_neutral)
    return (opacity + (o_confirmed - opacity) * c,
            raw_from_beta(beta),
            log_scale_compensation(neutral, beta))


def init_f_rest(n, rows, lobes, amplitude=0.0, device="cuda"):
    f_rest = torch.zeros(n, rows, 3, device=device)
    if lobes <= 0 or n == 0:
        return f_rest
    flat = f_rest.view(n, -1)
    for k in range(lobes):
        flat[:, 6 * k:6 * k + 3] = amplitude
        flat[:, 6 * k + 3] = (torch.rand(n, device=device) * 2 - 1).acos()
        flat[:, 6 * k + 4] = torch.rand(n, device=device) * (2 * math.pi)
    return f_rest


class SpawnMixin:
    def _make_dummy_ext_tensor(self):
        nodes = self.hierarchy.active_gaussians
        ext = super()._make_dummy_ext_tensor()
        for key in ("beta", "res_native", "err_stats"):
            ext[key] = nodes[key]["val"][:0]
        return ext

    def _res_guard(self, depth, sample_mask, main_gaussians_map):
        gamma = self.args.spawn_res_guard
        n_act = self.hierarchy.active_gaussians_count
        if gamma <= 0 or main_gaussians_map is None or n_act == 0:
            return torch.ones_like(depth, dtype=torch.bool)
        native = self.hierarchy.active_gaussians["res_native"]["val"][:n_act]
        ids = main_gaussians_map[0][sample_mask].long()
        density = self.f / depth.clamp_min(1e-6)
        below = (ids >= 0) & (ids < n_act) & (density < gamma * native[ids.clamp(0, n_act - 1)])
        return ~below

    @torch.no_grad()
    def _estimate_gaussian_depths(self, keyframe, keyframe_id, sample_mask, mono_idepth,
                                  mono_depth_conf, main_gaussians_map=None):
        if not isinstance(self.guided_mvs, MvsDisagreement):
            self.guided_mvs = MvsDisagreement(self.guided_mvs)

        sampled_uv = self.uv[sample_mask]
        matches = self.keyframes[keyframe_id].desc_kpts.matches
        prev_kfs = [self.keyframes[i] for i in matches][:self.guided_mvs.n_cams + 1]
        for i, other in enumerate(prev_kfs):
            if other.index == keyframe.index:
                prev_kfs.pop(i)
                break

        depth, idist = self.guided_mvs(sampled_uv, keyframe, prev_kfs, mono_idepth)
        accurate_mask = idist >= 0
        mvs_conf = (1.0 - idist / self.guided_mvs.idepth_range).clamp(0.0, 1.0) * accurate_mask

        conf = sample(mono_depth_conf, sampled_uv.view(1, 1, -1, 2),
                      self.width, self.height)[0, 0, 0]
        valid = ((conf > self.args.depth_conf_threshold) & (depth > 1e-6)
                 & self._res_guard(depth, sample_mask, main_gaussians_map))
        sample_mask[sample_mask.clone()] = valid
        return sampled_uv[valid], depth[valid], accurate_mask[valid], mvs_conf[valid], sample_mask

    @torch.no_grad()
    def _check_occlusions(self, sampled_uv, depth, accurate_mask, mvs_conf, sample_mask,
                          rendered_depth):
        if rendered_depth is None:
            return sampled_uv, depth, accurate_mask, mvs_conf, sample_mask
        valid = depth < self.args.occlusion_depth_factor * rendered_depth[sample_mask]
        sample_mask[sample_mask.clone()] = valid
        return sampled_uv[valid], depth[valid], accurate_mask[valid], mvs_conf[valid], sample_mask

    @torch.no_grad()
    def _apply_removal(self, keyframe_id, keep):
        keep[self.id_in_hierarchy != -1] = True
        with self.lock:
            self.hierarchy.add_and_prune(self._make_dummy_ext_tensor(), keep)
        render_pkg = self.render_from_id(keyframe_id)
        return 1 / render_pkg["invdepth"][0].clamp_min(1e-8), render_pkg["mainGaussID"]

    @torch.no_grad()
    def _remove_coarse_gaussians(self, keyframe, keyframe_id, main_gaussians_map,
                                 accurate_mask, sample_mask):
        accurate_sample_mask = sample_mask.clone()
        accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
        main = main_gaussians_map[:, accurate_sample_mask]
        ids, counts = torch.unique(main[main >= 0], return_counts=True)
        keep = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
        keep[ids] = (counts < self.args.coarse_removal_count) | (
            (self.kf_id[ids] - keyframe.index).abs() < self.args.coarse_removal_kf_window)
        return self._apply_removal(keyframe_id, keep)[0]

    @torch.no_grad()
    def _init_gaussian_attributes(self, keyframe, sampled_uv, depth, accurate_mask, sample_mask,
                                  init_proba, img, mvs_conf=None):
        R = keyframe.get_R()
        t = keyframe.get_t()
        dense_pts = depth2points(sampled_uv, depth.unsqueeze(-1), self.f, self.centre)
        dense_pts = (dense_pts - t) @ R
        has_pt3d = keyframe.desc_kpts.get_has_pt3d()
        new_pts = torch.cat([dense_pts, keyframe.desc_kpts.get_pts3d()[has_pt3d]], dim=0)

        sampler = make_torch_sampler(keyframe.desc_kpts.kpts[has_pt3d], self.width, self.height)
        sampler = sampler[None, None]
        match_colors = F.grid_sample(img[None], sampler, mode="bilinear",
                                     align_corners=True).view(3, -1)
        f_dc = torch.cat([img[:, sample_mask], match_colors], dim=1)
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))

        match_proba = F.grid_sample(init_proba[None, None], sampler, mode="bilinear",
                                    align_corners=True).view(-1)
        proba = torch.cat([init_proba[sample_mask], match_proba], dim=0)
        scales = 1 / torch.sqrt(proba)
        scales.clamp_(1, self.max_scale_screen_size)
        scales.mul_(1 / self.f)
        scales *= torch.linalg.vector_norm(new_pts - keyframe.approx_centre[None], dim=-1)
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

        n_total = f_dc.shape[0]
        n_dense = sampled_uv.shape[0]
        opacities = torch.ones(n_total, 1, device="cuda")
        opacities[:n_dense] *= (self.args.init_opacity_accurate * accurate_mask[..., None]
                                + self.args.init_opacity_inaccurate * ~accurate_mask[..., None])
        opacities[n_dense:] *= self.args.init_opacity_triangulated
        betas = raw_from_beta(torch.full((n_total, 1), self.args.beta_init, device="cuda"))

        if self.args.confidence_spawn and mvs_conf is not None and n_dense > 0:
            baseline = max_covisible_baseline(self, keyframe.index)
            c = (mvs_conf * observability(depth, baseline, self.args.obs_min_angle_deg)).clamp(0, 1)
            c = c ** self.args.confidence_gamma
            opacity, beta_raw, compensation = spawn_from_confidence(
                c, opacities[:n_dense, 0], self.args.confirmed_opacity, self.args.beta_init)
            opacities[:n_dense, 0] = opacity
            if self.args.shape_confidence:
                betas[:n_dense, 0] = beta_raw
                scales[:n_dense] += compensation.unsqueeze(-1)
        opacities = inverse_sigmoid(opacities.clamp(1e-4, 0.9999))

        rots = torch.zeros(n_total, 4, device="cuda")
        rots[:, 0] = 1

        extension_tensors = {
            "xyz": new_pts,
            "f_dc": f_dc,
            "f_rest": init_f_rest(n_total, self.hierarchy.f_rest_rows, self.args.sb_lobes,
                                  self.args.lobe_init_amp),
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
            "beta": betas,
            "res_native": torch.zeros(n_total, device="cuda"),
            "err_stats": torch.zeros(n_total, 3, device="cuda"),
        }
        return new_pts, extension_tensors

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        if self.args.skip_gs:
            return
        keyframe = self.keyframes[keyframe_id]
        if keyframe.info["is_test"]:
            return

        mono_idepth, mono_depth_conf = self.depth_estimator(keyframe.image[None])
        init_proba, rendered_depth, sample_mask, img, main_ids = self._compute_init_probability(
            keyframe, keyframe_id)

        if keyframe.desc_kpts.get_has_pt3d().sum() > self.args.min_pts3d_for_depth_align:
            mono_idepth = align_depth(
                mono_idepth, keyframe, self.width, self.height,
                grid_size=self.args.depth_grid_size,
                outlier_mult=self.args.depth_align_outlier_mult,
                grid_outlier_mult=self.args.depth_grid_outlier_mult,
                min_pts_per_block=self.args.depth_grid_min_pts_per_block,
                min_kpts_per_block=self.args.depth_grid_min_kpts_per_block,
                scale_tolerance=self.args.depth_grid_scale_tolerance)
        else:
            mono_depth_conf = torch.zeros_like(mono_depth_conf)

        sampled_uv, depth, accurate_mask, mvs_conf, sample_mask = self._estimate_gaussian_depths(
            keyframe, keyframe_id, sample_mask, mono_idepth, mono_depth_conf, main_ids)

        if self.xyz.shape[0] > 0 and main_ids is not None and self.args.coarse_removal_count > 0:
            rendered_depth = self._remove_coarse_gaussians(
                keyframe, keyframe_id, main_ids, accurate_mask, sample_mask)

        sampled_uv, depth, accurate_mask, mvs_conf, sample_mask = self._check_occlusions(
            sampled_uv, depth, accurate_mask, mvs_conf, sample_mask, rendered_depth)

        new_pts, extension_tensors = self._init_gaussian_attributes(
            keyframe, sampled_uv, depth, accurate_mask, sample_mask, init_proba, img, mvs_conf)
        self._prune_and_filter_gaussians(keyframe, new_pts, extension_tensors)

        do_relocate = self.args.relocate
        do_solidify = self.args.beta_solidify > 0
        if (do_relocate or do_solidify) and len(self.xyz) > 0 and not keyframe.is_test:
            from hibs.lifecycle import accumulate_error, relocate, solidify
            pkg = self.render_from_id(keyframe_id)
            err_g, cnt_g = accumulate_error(self, keyframe_id, pkg)
            if do_relocate:
                relocate(self, keyframe_id, err_g, pkg)
            if do_solidify:
                for key, count in solidify(self, keyframe_id, cnt_g, pkg).items():
                    self.run_stats[key] = self.run_stats.get(key, 0) + count

        self.max_n_active_gaussians = max(getattr(self, "max_n_active_gaussians", 0),
                                          self.hierarchy.n_active_gaussians)
        free, total = torch.cuda.mem_get_info()
        if 1 - free / total > self.args.gpu_mem_clear_fraction:
            gc.collect()
            torch.cuda.empty_cache()
