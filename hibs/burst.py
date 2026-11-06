import numpy as np
import torch
from fused_ssim import fused_ssim

from hibs import largescale


class BurstMixin:
    def _sample_keyframe(self, finetuning):
        last = self.keyframes[-1]
        if (np.random.rand() > self.use_last_frame_proba or self.last_trained_id == -1
                or finetuning or (last.is_test and not last.optimize_pose)):
            proba = self._kf_proba_np
            if proba is None or len(proba) != self.training_proba.shape[0]:
                proba = self._kf_proba_np = (
                    self.training_proba / self.training_proba.sum()).cpu().numpy()
            return int(np.random.choice(len(proba), p=proba))
        return -1

    def _regularizers(self):
        n = self.hierarchy.active_gaussians_count
        nodes = self.hierarchy.active_gaussians
        loss = 0.0
        if self.args.opacity_reg > 0:
            loss = loss + self.args.opacity_reg * torch.sigmoid(nodes["opacity"]["val"][:n]).mean()
        if self.args.scale_reg > 0:
            loss = loss + self.args.scale_reg * torch.exp(nodes["scaling"]["val"][:n]).mean()
        return loss

    def optimization_step(self, keyframe_id=None, finetuning=False):
        if len(self.xyz) == 0:
            return
        if keyframe_id is None:
            keyframe_id = self._sample_keyframe(finetuning)
        keyframe = self.keyframes[keyframe_id]
        is_test = keyframe.is_test

        gt_image = keyframe.image.to(device="cuda", non_blocking=True)
        mask = keyframe.mask.to(device="cuda", non_blocking=True) if keyframe.mask is not None else None

        with self._scene_mutation_lock():
            keyframe.zero_grad()
            if self.gaussian_optimizer is not None:
                self.gaussian_optimizer.zero_grad()
            if self.ppisp_optimizer is not None:
                self.ppisp_optimizer.zero_grad(set_to_none=True)

            step_in_backward = (self.args.fused_raster and not is_test
                                and self.gaussian_optimizer is not None)
            density = lr_scale = None
            if self.args.native_res_weight > 0 and not is_test and self.hierarchy.active_gaussians_count > 0:
                density = largescale.view_density(self, keyframe)
                if density is not None and step_in_backward:
                    lr_scale = largescale.step_scale(self, density)

            render_pkg = self.render_from_id(
                keyframe_id, bg=torch.rand(3, device="cuda", generator=self.gs_rng),
                step_in_backward=step_in_backward, lr_scale=lr_scale, need_aux=False)
            step_in_backward = render_pkg["step_in_backward"]
            image = render_pkg["render"]

            if mask is not None:
                image = image * mask
                gt_image = gt_image * mask

            l1 = (image - gt_image).abs().mean()
            dssim = 1 - fused_ssim(image[None], gt_image[None])
            loss = self.lambda_dssim * dssim + (1 - self.lambda_dssim) * l1
            if not is_test and not step_in_backward:
                loss = loss + self._regularizers()
            if self.ppisp is not None and not is_test:
                loss = loss + self.ppisp.get_regularization_loss()

            loss.backward()
            keyframe.step()
            if not is_test:
                if not step_in_backward:
                    self.gaussian_optimizer.step(
                        render_pkg["visibility_filter"], self.hierarchy.active_gaussians_count)
                if self.ppisp_optimizer is not None:
                    self.ppisp_optimizer.step()

            if density is not None:
                largescale.update_res_native(self, density * (render_pkg["radii"] > 0))

        self.last_trained_id = keyframe_id
