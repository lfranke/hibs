# faster-beta-splatting: fused-Adam beta-kernel rasterizer for online
# incremental reconstruction. Derived from Faster-GS (Hahlbohm et al.,
# CVPR 2026, Apache-2.0); beta kernel and spherical-beta colour from
# Deformable Beta Splatting (Liu et al., SIGGRAPH 2025). See README.md.
#
# Two entry points:
#   rasterize(...)            renders and, on loss.backward(), computes
#                             gradients AND applies the Adam step to every
#                             primitive parameter in one fused pass (the
#                             parameters/moments are updated in place; only
#                             the w2c matrix receives an autograd gradient,
#                             for online pose refinement).
#   rasterize_with_grads(...) test/interop mode: backward computes and
#                             stores raw gradients on the input tensors'
#                             .grad (no Adam step).
#
# Conventions:
#   w2c            [3, 4] row-major world-to-camera (x_cam = w2c @ [x, 1])
#   means          [N, 3]      raw (world space)
#   scales         [N, 3]      log-scale
#   rotations      [N, 4]      unnormalized quaternion (r, x, y, z)
#   opacities      [N, 1]      raw (sigmoid applied inside)
#   raw_betas      [N, 1]      kernel exponent b = 4 * exp(raw)
#   sh0            [N, 1, 3] or [N, 3]   (SH band 0, the diffuse term)
#   sh_rest        the colour buffer, see below
#   bg             [3] constant or [3, H, W] per-pixel background
#
# Three colour/storage modes, selected by RasterizerSettings.active_sh_bases:
#   active_sh_bases > 0   deg-3 spherical harmonics, sh_rest [N, 15, 3]
#                         C(v) = 0.5 + C0 * sh0 + sum_b Y_b(v) * c_b
#   active_sh_bases = -K  K spherical-beta lobes of 6 floats each,
#                         (r, g, b, theta, phi, beta_raw) per lobe:
#                         C(v) = 0.5 + C0 * sh0
#                                + sum_k a_k * max(0, dot(mu_k, v))^(4 exp(b_k))
#                         stored either PADDED into the deg-3 SH buffer
#                         [N, 15, 3] (lobes in the leading 6K of 45 floats,
#                         the tail unused) or TIGHT as [N, K, 6]. The
#                         per-primitive stride is read off the tensor, and
#                         the packing inside a lobe is the same either way,
#                         so the two convert by a reshape/slice:
#                           tight = padded.reshape(N, -1)[:, :6*K].reshape(N, K, 6)
#
# Outputs: image [3,H,W], invdepth [1,H,W], main_ids [2,H,W] (best /
# second-best contributor by blending weight, -1 = none), final
# transmittance [H,W]. Only `image` is differentiable.

from dataclasses import dataclass
from typing import Optional, Union

import torch

from . import _C

MomentState = Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class AdamStates:
    """Adam moment state per parameter group, updated in place.

    Every group is a (exp_avg, exp_avg_sq) pair of tensors with the same
    element layout as the parameters, so an external optimizer's existing
    state tensors (e.g. the i3dgs hierarchy's) can be passed straight in.
    The update matches the classic 3DGS SparseGaussianAdam (no bias
    correction).

    `sh_rest` additionally accepts a SINGLE INTERLEAVED tensor holding
    (exp_avg, exp_avg_sq) adjacent per parameter, i.e. 2 * sh_rest.numel()
    elements, read as float2. The lobe loop does 6 moment read-modify-writes
    per lobe in a memory-bound kernel, and pairing them into one float2
    stream halves those transactions. The arithmetic is unchanged, so the
    two layouts produce bit-identical results. Interleaved moments are only
    available for the spherical-beta colour model (active_sh_bases < 0).
    """
    means: "tuple[torch.Tensor, torch.Tensor]"
    scales: "tuple[torch.Tensor, torch.Tensor]"
    rotations: "tuple[torch.Tensor, torch.Tensor]"
    opacities: "tuple[torch.Tensor, torch.Tensor]"
    betas: "tuple[torch.Tensor, torch.Tensor]"
    sh0: "tuple[torch.Tensor, torch.Tensor]"
    sh_rest: MomentState
    beta1: float = 0.9
    beta2: float = 0.999
    # L1 regularizer coefficients (3DGS-MCMC / Deformable Beta Splatting):
    # loss += opacity_reg * mean(sigmoid(o)) + scale_reg * mean(exp(s)).
    # Applied inside the fused backward as additive gradients on the touched
    # primitives; pass the raw coefficients, the wrapper pre-divides by the
    # primitive count.
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    # Optional [N] float32 per-primitive step scale: the whole Adam update of
    # primitive i is multiplied by lr_scale[i] inside the fused backward
    # (equivalent to a post-step blend toward the pre-step value, with no
    # extra traffic).
    lr_scale: Optional[torch.Tensor] = None

    @property
    def sh_rest_interleaved(self) -> bool:
        return isinstance(self.sh_rest, torch.Tensor)

    @staticmethod
    def for_params(means, scales, rotations, opacities, betas, sh0, sh_rest,
                   beta1: float = 0.9, beta2: float = 0.999,
                   interleaved_sh_rest: bool = False) -> "AdamStates":
        def m(t):
            return (torch.zeros(t.numel(), device=t.device, dtype=torch.float32),
                    torch.zeros(t.numel(), device=t.device, dtype=torch.float32))
        rest = (torch.zeros(2 * sh_rest.numel(), device=sh_rest.device,
                            dtype=torch.float32)
                if interleaved_sh_rest else m(sh_rest))
        return AdamStates(m(means), m(scales), m(rotations), m(opacities),
                          m(betas), m(sh0), rest, beta1=beta1, beta2=beta2)

    def flat(self):
        moments = [t for pair in (self.means, self.scales, self.rotations,
                                  self.opacities, self.betas, self.sh0)
                   for t in pair]
        if self.sh_rest_interleaved:
            # an empty second slot tells the kernel to read one float2 pool
            return moments + [self.sh_rest, self.sh_rest.new_empty(0)]
        return moments + list(self.sh_rest)


@dataclass
class LearningRates:
    means: float = 1.6e-4
    scales: float = 5e-3
    rotations: float = 1e-3
    opacities: float = 2.5e-2
    betas: float = 1e-3
    sh0: float = 2.5e-3
    sh_rest: float = 1.25e-4


@dataclass
class RasterizerSettings:
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    # > 0: number of active SH bases; = -K: K spherical-beta lobes
    active_sh_bases: int = 16
    near_plane: float = 0.01
    far_plane: float = 10000.0
    invdepth: bool = True
    main_ids: bool = True


def _expand_bg(bg: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if bg.numel() == 3:
        return bg.reshape(3, 1, 1).expand(3, height, width).contiguous()
    return bg.contiguous()


def _check_sh_rest(sh_rest: torch.Tensor, active_sh_bases: int) -> torch.Tensor:
    """Validate the colour buffer against the colour model.

    The kernel takes the per-primitive stride from the tensor, so a buffer
    that does not match `active_sh_bases` would be read with a wrong stride
    and render garbage rather than fail. Contiguity is required, not silently
    repaired: the fused backward writes the Adam step into this exact
    storage, so handing the kernel a .contiguous() COPY would update the copy
    and leave the caller's buffer untouched.
    """
    if sh_rest.dim() != 3:
        raise ValueError(f"sh_rest must be [N, rows, cols], got {tuple(sh_rest.shape)}")
    rows, cols = sh_rest.shape[1], sh_rest.shape[2]
    if active_sh_bases >= 0:
        if (rows, cols) != (15, 3):
            raise ValueError(
                f"SH colour (active_sh_bases={active_sh_bases}) needs the deg-3 "
                f"storage sh_rest [N, 15, 3], got {tuple(sh_rest.shape)}")
    else:
        k = -active_sh_bases
        if cols == 6:
            if rows != k:
                raise ValueError(
                    f"a tight lobe buffer must be [N, K, 6] with K = "
                    f"{k} = -active_sh_bases, got {tuple(sh_rest.shape)}")
        elif rows * cols < 6 * k:
            raise ValueError(
                f"sh_rest holds {rows * cols} floats per primitive, too few for "
                f"{k} lobes of 6; use a padded [N, 15, 3] or tight [N, {k}, 6] buffer")
    if not sh_rest.is_contiguous():
        raise ValueError("sh_rest must be contiguous (the fused backward "
                         "updates this storage in place)")
    return sh_rest


def _check_sh_rest_moments(adam: AdamStates, sh_rest: torch.Tensor,
                           active_sh_bases: int) -> None:
    if not adam.sh_rest_interleaved:
        return
    if active_sh_bases >= 0:
        raise ValueError("interleaved sh_rest moments are only supported for the "
                         "spherical-beta colour model (active_sh_bases < 0)")
    # The kernel indexes the pool as float2, so a short buffer would be an
    # out-of-bounds write rather than a visible error. >= not ==: callers may
    # hand over a capacity-sized pool whose prefix covers the active
    # primitives, and the kernel only ever indexes that prefix.
    if adam.sh_rest.numel() < 2 * sh_rest.numel():
        raise ValueError(
            f"the interleaved adam.sh_rest pool needs at least "
            f"2 * {sh_rest.numel()} = {2 * sh_rest.numel()} elements "
            f"(exp_avg, exp_avg_sq adjacent per parameter), got "
            f"{adam.sh_rest.numel()}")
    if not adam.sh_rest.is_contiguous():
        raise ValueError("adam.sh_rest must be contiguous")


class _RasterizeFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, autograd_dummy, w2c, cam_position,
                means, scales, rotations, opacities, raw_betas, sh0, sh_rest,
                bg, adam: AdamStates, lrs: LearningRates,
                settings: RasterizerSettings):
        bg_full = _expand_bg(bg, settings.height, settings.width)
        (image, invdepth, main_ids, final_t, visibility,
         pbuf, tbuf, ibuf, bbuf,
         n_instances, n_buckets, selector) = _C.forward(
            means, scales, rotations, opacities, raw_betas,
            sh0.contiguous(), sh_rest,
            w2c.contiguous(), cam_position.contiguous(), bg_full,
            settings.active_sh_bases, settings.width, settings.height,
            settings.focal_x, settings.focal_y, settings.center_x, settings.center_y,
            settings.near_plane, settings.far_plane,
            settings.invdepth, settings.main_ids)
        ctx.save_for_backward(image, final_t, pbuf, tbuf, ibuf, bbuf,
                              w2c, cam_position, bg_full)
        ctx.params = (means, scales, rotations, opacities, raw_betas, sh0, sh_rest)
        ctx.adam = adam
        ctx.lrs = lrs
        ctx.settings = settings
        ctx.buffer_state = (n_instances, n_buckets, selector)
        ctx.w2c_requires_grad = w2c.requires_grad
        ctx.mark_non_differentiable(invdepth, main_ids, final_t, visibility)
        return image, invdepth, main_ids, final_t, visibility, autograd_dummy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_image, _gi, _gm, _gt, _gv, _gd):
        (image, final_t, pbuf, tbuf, ibuf, bbuf,
         w2c, cam_position, bg_full) = ctx.saved_tensors
        means, scales, rotations, opacities, raw_betas, sh0, sh_rest = ctx.params
        adam: AdamStates = ctx.adam
        lrs: LearningRates = ctx.lrs
        s: RasterizerSettings = ctx.settings
        outs = _C.backward(
            means, scales, rotations, opacities, raw_betas,
            sh0, sh_rest,
            *adam.flat(),
            grad_image, image, final_t, pbuf, tbuf, ibuf, bbuf,
            w2c, cam_position, bg_full,
            s.active_sh_bases, s.width, s.height,
            s.focal_x, s.focal_y, s.center_x, s.center_y,
            adam.lr_scale if adam.lr_scale is not None
            else torch.empty(0, device=means.device),
            lrs.means, lrs.scales, lrs.rotations, lrs.opacities,
            lrs.betas, lrs.sh0, lrs.sh_rest,
            adam.opacity_reg / max(means.shape[0], 1),
            adam.scale_reg / max(means.shape[0], 1),
            adam.beta1, adam.beta2,
            *ctx.buffer_state,
            ctx.w2c_requires_grad,
            False)
        grad_w2c = outs[0] if ctx.w2c_requires_grad else None
        return (None, grad_w2c) + (None,) * 12


class _RasterizeWithGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, autograd_dummy, w2c, cam_position,
                means, scales, rotations, opacities, raw_betas, sh0, sh_rest,
                bg, settings: RasterizerSettings):
        bg_full = _expand_bg(bg, settings.height, settings.width)
        (image, invdepth, main_ids, final_t, visibility,
         pbuf, tbuf, ibuf, bbuf,
         n_instances, n_buckets, selector) = _C.forward(
            means, scales, rotations, opacities, raw_betas,
            sh0.contiguous(), sh_rest,
            w2c.contiguous(), cam_position.contiguous(), bg_full,
            settings.active_sh_bases, settings.width, settings.height,
            settings.focal_x, settings.focal_y, settings.center_x, settings.center_y,
            settings.near_plane, settings.far_plane,
            settings.invdepth, settings.main_ids)
        ctx.save_for_backward(image, final_t, pbuf, tbuf, ibuf, bbuf,
                              w2c, cam_position, bg_full,
                              means, scales, rotations, opacities, raw_betas, sh0, sh_rest)
        ctx.settings = settings
        ctx.buffer_state = (n_instances, n_buckets, selector)
        ctx.shapes = (sh0.shape, sh_rest.shape)
        ctx.mark_non_differentiable(invdepth, main_ids, final_t, visibility)
        return image, invdepth, main_ids, final_t, visibility, autograd_dummy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_image, _gi, _gm, _gt, _gv, _gd):
        (image, final_t, pbuf, tbuf, ibuf, bbuf,
         w2c, cam_position, bg_full,
         means, scales, rotations, opacities, raw_betas, sh0, sh_rest) = ctx.saved_tensors
        s: RasterizerSettings = ctx.settings
        dummy_moments = torch.empty(0, device=means.device)
        outs = _C.backward(
            means, scales, rotations, opacities, raw_betas,
            sh0, sh_rest,
            *([dummy_moments] * 14),
            grad_image, image, final_t, pbuf, tbuf, ibuf, bbuf,
            w2c, cam_position, bg_full,
            s.active_sh_bases, s.width, s.height,
            s.focal_x, s.focal_y, s.center_x, s.center_y,
            torch.empty(0, device=means.device),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            0.9, 0.999,
            *ctx.buffer_state,
            True,
            True)
        (grad_w2c, g_means, g_scales, g_rot, g_op, g_betas, g_sh0, g_shrest) = outs
        sh0_shape, sh_rest_shape = ctx.shapes
        return (None, grad_w2c, None,
                g_means, g_scales, g_rot, g_op, g_betas,
                g_sh0.view(sh0_shape), g_shrest.view(sh_rest_shape),
                None, None)


def rasterize(w2c, cam_position, means, scales, rotations, opacities,
              raw_betas, sh0, sh_rest, bg, adam: AdamStates,
              lrs: LearningRates, settings: RasterizerSettings,
              autograd_dummy: Optional[torch.Tensor] = None):
    """Fused render: backward applies the Adam step in place.

    Pass an `autograd_dummy` tensor with requires_grad=True to keep the
    graph alive (parameters themselves are non-differentiable inputs here).
    """
    if autograd_dummy is None:
        autograd_dummy = torch.zeros(1, device=means.device, requires_grad=True)
    _check_sh_rest(sh_rest, settings.active_sh_bases)
    _check_sh_rest_moments(adam, sh_rest, settings.active_sh_bases)
    return _RasterizeFused.apply(
        autograd_dummy, w2c, cam_position,
        means, scales, rotations, opacities, raw_betas, sh0, sh_rest,
        bg, adam, lrs, settings)[:5]


def rasterize_with_grads(w2c, cam_position, means, scales, rotations,
                         opacities, raw_betas, sh0, sh_rest, bg,
                         settings: RasterizerSettings):
    """Test/interop mode: behaves like a normal autograd rasterizer
    (gradients land on .grad, no optimizer update)."""
    dummy = torch.zeros(1, device=means.device, requires_grad=True)
    _check_sh_rest(sh_rest, settings.active_sh_bases)
    return _RasterizeWithGrads.apply(
        dummy, w2c, cam_position,
        means, scales, rotations, opacities, raw_betas, sh0, sh_rest,
        bg, settings)[:5]


__all__ = [
    "rasterize", "rasterize_with_grads",
    "AdamStates", "LearningRates", "RasterizerSettings",
]
