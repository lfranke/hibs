import math

import torch
import faster_beta_splatting as fbs
from utils import fov2focal

ZERO_BG = torch.zeros(3, device="cuda")
TOP_VIEW_MAX = 200_000


class RenderMixin:
    def render_from_id(self, keyframe_id, scaling_modifier=1, bg=ZERO_BG,
                       compensate_exposure=True, step_in_backward=False, lr_scale=None,
                       need_aux=True):
        keyframe = self.keyframes[keyframe_id]
        render_pkg = self.render(self.width, self.height, keyframe.get_Rt().T,
                                 scaling_modifier, bg, step_in_backward=step_in_backward,
                                 lr_scale=lr_scale, need_aux=need_aux)
        if compensate_exposure:
            render_pkg["render"] = self.apply_compensation(keyframe, render_pkg["render"])
        return render_pkg

    def apply_compensation(self, keyframe, image):
        if self.ppisp is None:
            return super().apply_compensation(keyframe, image)
        corrected = self.ppisp(image.permute(1, 2, 0).contiguous(),
                               camera_idx=0, frame_idx=keyframe.index)
        return corrected.permute(2, 0, 1).clamp(0, 1)

    def _prepare_render_params(self, top_view, scaling_modifier):
        params = super()._prepare_render_params(top_view, scaling_modifier)
        if params is None or not self.args.beta_kernel:
            return params
        n = self.hierarchy.active_gaussians_count
        beta = self.hierarchy.active_gaussians["beta"]["val"][:n]
        step = len(self.xyz) // TOP_VIEW_MAX
        if top_view and step > 1:
            beta = beta[::step].contiguous()
        return (*params, beta)

    def _focal(self):
        state = (id(self.f), self.f._version)
        if getattr(self, "_focal_state", None) != state:
            self._focal_state = state
            self._focal_value = self.f.item()
        return self._focal_value

    def _lobe_view(self, f_rest):
        k = self.hierarchy.sb_lobes
        if k > 0 and f_rest.shape[1] == 2 * k:
            return f_rest.view(-1, k, 6)
        return f_rest

    def _learning_rates(self):
        rates = getattr(self, "_fbs_rates", None)
        if rates is None:
            a = self.args
            rest_lr = a.feature_lr if a.sb_lobes > 0 else a.feature_lr / 20.0
            rates = self._fbs_rates = fbs.LearningRates(
                means=a.position_lr, scales=a.scaling_lr, rotations=a.rotation_lr,
                opacities=a.opacity_lr, betas=a.beta_lr, sh0=a.feature_lr, sh_rest=rest_lr)
        return rates

    def _adam_states(self, lr_scale):
        nodes = self.hierarchy.active_gaussians

        def moments(key):
            node = nodes[key]
            return node["moments"] if "moments" in node else (node["exp_avg"], node["exp_avg_sq"])

        return fbs.AdamStates(
            moments("xyz"), moments("scaling"), moments("rotation"), moments("opacity"),
            moments("beta"), moments("f_dc"), moments("f_rest"),
            beta1=self.args.adam_beta1, beta2=self.args.adam_beta2,
            opacity_reg=self.args.opacity_reg, scale_reg=self.args.scale_reg,
            lr_scale=lr_scale)

    def render(self, width, height, view_matrix, scaling_modifier=1,
               bg=ZERO_BG, top_view=False, fov_x=None, fov_y=None,
               step_in_backward=False, lr_scale=None, need_aux=True):
        if not self.args.beta_kernel:
            pkg = super().render(width, height, view_matrix, scaling_modifier, bg, top_view,
                                 fov_x, fov_y)
            return dict(pkg, step_in_backward=False)
        cam_centre = -(view_matrix[:3, :3] @ view_matrix[3, :3])
        if fov_x is None and fov_y is None:
            f = self._focal()
        elif fov_x is not None and fov_y is not None:
            f = (fov2focal(fov_x, width) + fov2focal(fov_y, height)) / 2
        else:
            raise ValueError("provide both fov_x and fov_y, or neither")

        if self.inference_mode and not top_view:
            self.hierarchy.update_cut(cam_centre, self.render_tau)

        with self.lock:
            params = self._prepare_render_params(top_view, scaling_modifier)
            if params is None:
                empty = torch.zeros(len(self.xyz), device="cuda")
                return {
                    "render": torch.zeros(3, height, width, device="cuda"),
                    "invdepth": torch.zeros(1, height, width, device="cuda"),
                    "mainGaussID": torch.zeros(2, height, width, device="cuda", dtype=torch.int32),
                    "final_T": torch.ones(height, width, device="cuda"),
                    "radii": empty,
                    "visibility_filter": empty > 0,
                    "step_in_backward": False,
                }
            xyz, opacity, f_dc, f_rest, log_scaling, rotation, beta = params
            if not top_view and scaling_modifier != 1:
                log_scaling = log_scaling + math.log(scaling_modifier)
            step_in_backward = (step_in_backward and not self.inference_mode
                                and self.gaussian_optimizer is not None)
            settings = fbs.RasterizerSettings(
                width=int(width), height=int(height), focal_x=float(f), focal_y=float(f),
                center_x=width / 2.0, center_y=height / 2.0,
                active_sh_bases=(self.active_sh_degree if self.active_sh_degree < 0
                                 else (self.active_sh_degree + 1) ** 2),
                near_plane=self.args.render_near_plane, far_plane=self.args.render_far_plane,
                invdepth=need_aux, main_ids=need_aux)
            w2c = view_matrix.transpose(0, 1)[:3]
            rest = self._lobe_view(f_rest)
            if step_in_backward:
                args = (xyz.detach(), log_scaling.detach(), rotation.detach(), opacity.detach(),
                        beta.detach(), f_dc.detach(), rest.detach())
                color, invdepth, main_ids, final_t, visibility = fbs.rasterize(
                    w2c, cam_centre, *args, bg, self._adam_states(lr_scale),
                    self._learning_rates(), settings, self.autograd_dummy)
            else:
                color, invdepth, main_ids, final_t, visibility = fbs.rasterize_with_grads(
                    w2c, cam_centre, xyz, log_scaling, rotation, opacity, beta, f_dc, rest,
                    bg, settings)

        return {
            "render": color,
            "invdepth": invdepth,
            "mainGaussID": main_ids,
            "final_T": final_t,
            "radii": visibility,
            "visibility_filter": visibility > 0,
            "step_in_backward": step_in_backward,
        }
