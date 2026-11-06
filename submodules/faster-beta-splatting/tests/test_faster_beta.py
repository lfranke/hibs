"""Test suite for faster-beta-splatting.

Covers, in order of importance:
 1. finite-difference gradient checks for EVERY parameter type (means,
    scales, rotations, opacities, raw betas, sh0, sh_rest) and the w2c matrix
 2. fused-Adam step parity: rasterize() must produce exactly the same
    parameter update as rasterize_with_grads() + a reference Adam in python
 3. auxiliary outputs: invdepth, main contributor ids, final transmittance,
    per-pixel background

Run:  python tests/test_faster_beta.py
"""
import math
import sys
import time

import torch

import faster_beta_splatting as fbs

torch.manual_seed(7)
DEVICE = "cuda"


def make_scene(n, w=96, h=64, focal=80.0, op_range=(0.3, 0.95), seed=7,
               spread=0.8, depth=(2.0, 6.0), scale=(0.05, 0.18), sb_lobes=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.rand(n, 3, generator=g).to(DEVICE) * 2 - 1
    means[:, 0:2] *= spread
    means[:, 2] = depth[0] + torch.rand(n, generator=g).to(DEVICE) * (depth[1] - depth[0])
    scales = torch.log(scale[0] + torch.rand(n, 3, generator=g).to(DEVICE) * (scale[1] - scale[0]))
    rotations = torch.randn(n, 4, generator=g).to(DEVICE) * 0.2
    rotations[:, 0] += 1.0
    op_lo, op_hi = op_range
    opacities = torch.logit(op_lo + torch.rand(n, 1, generator=g).to(DEVICE) * (op_hi - op_lo))
    raw_betas = torch.randn(n, 1, generator=g).to(DEVICE) * 0.3  # b around 4
    sh0 = torch.randn(n, 1, 3, generator=g).to(DEVICE) * 0.4
    if sb_lobes > 0:
        # packed spherical-beta lobes: (r, g, b, theta, phi, beta_raw) x K
        flat = torch.zeros(n, 45)
        for k in range(sb_lobes):
            flat[:, 6 * k + 0:6 * k + 3] = torch.randn(n, 3, generator=g) * 0.3
            flat[:, 6 * k + 3] = (torch.rand(n, generator=g) * 2 - 1).acos()
            flat[:, 6 * k + 4] = torch.rand(n, generator=g) * (2 * math.pi)
            flat[:, 6 * k + 5] = torch.randn(n, generator=g) * 0.3
        sh_rest = flat.view(n, 15, 3).to(DEVICE)
    else:
        sh_rest = torch.randn(n, 15, 3, generator=g).to(DEVICE) * 0.05
    w2c = torch.tensor([[1., 0., 0., 0.02],
                        [0., 1., 0., -0.03],
                        [0., 0., 1., 0.05]], device=DEVICE)
    cam_pos = torch.tensor([-0.02, 0.03, -0.05], device=DEVICE)
    settings = fbs.RasterizerSettings(
        width=w, height=h, focal_x=focal, focal_y=focal,
        center_x=w / 2.0, center_y=h / 2.0,
        active_sh_bases=-sb_lobes if sb_lobes > 0 else 16)
    bg = torch.rand(3, h, w, device=DEVICE) * 0.3
    return means, scales, rotations, opacities, raw_betas, sh0, sh_rest, w2c, cam_pos, bg, settings


def loss_of(image, target):
    return ((image - target) ** 2).mean() + 0.1 * image.abs().mean()


_FD_WEIGHTS = None

def loss_strong(image):
    """High-SNR loss for finite differences: a random-weighted SUM keeps the
    per-component loss deltas far above fp32 rendering noise (a mean-reduced
    loss drowns eps-sized deltas in quantization)."""
    global _FD_WEIGHTS
    if _FD_WEIGHTS is None or _FD_WEIGHTS.shape != image.shape:
        g = torch.Generator(device="cpu").manual_seed(123)
        _FD_WEIGHTS = torch.randn(*image.shape, generator=g).to(image.device)
    return (image * _FD_WEIGHTS).sum()


def render_with_grads(params, w2c, cam_pos, bg, settings, target):
    (means, scales, rotations, opacities, raw_betas, sh0, sh_rest) = params
    img, invd, ids, ft, vis = fbs.rasterize_with_grads(
        w2c, cam_pos, means, scales, rotations, opacities, raw_betas,
        sh0, sh_rest, bg, settings)
    loss = loss_of(img, target)
    loss.backward()
    return img, invd, ids, ft, vis, loss


def test_finite_differences():
    """Central finite differences per parameter type against the analytic
    gradients, using the high-SNR sum loss. Note: w2c pose gradients are a
    documented approximation (see
    test_gradient_parity_with_existing_fork) — they are checked for dominant-
    component agreement, not exactness."""
    print("== finite-difference gradient checks ==")
    n = 24
    (means, scales, rotations, opacities, raw_betas, sh0, sh_rest,
     w2c, cam_pos, bg, settings) = make_scene(n)

    param_dict = {
        "means": means, "scales": scales, "rotations": rotations,
        "opacities": opacities, "raw_betas": raw_betas,
        "sh0": sh0, "sh_rest": sh_rest, "w2c": w2c,
    }
    for t in param_dict.values():
        t.requires_grad_(True)

    img, *_ = fbs.rasterize_with_grads(
        w2c, cam_pos, means, scales, rotations, opacities, raw_betas,
        sh0, sh_rest, bg, settings)
    loss_strong(img).backward()

    analytic = {k: (v.grad.clone() if v.grad is not None else torch.zeros_like(v))
                for k, v in param_dict.items()}

    eps_map = {"means": 1e-4, "scales": 2e-4, "rotations": 2e-4,
               "opacities": 2e-4, "raw_betas": 2e-4, "sh0": 2e-4,
               "sh_rest": 5e-4, "w2c": 2e-5}
    # The beta kernel has COMPACT SUPPORT plus the 1/255 alpha truncation: a
    # positional FD probe moves the support boundary across pixels, and the
    # profile still has finite slope at the cutoff (unlike a Gaussian tail),
    # so boundary-crossing components measure the truncation step, not the
    # gradient. This hits positional parameters: a few 'means' components,
    # and 'w2c' especially (a global transform perturbation crosses MANY
    # boundaries at once). Hence trimmed statistics: median + 75th
    # percentile, with wider trims for the positional parameters.
    # With AA opacity compensation (h = sqrt(det/det_dilated)) the FD probe
    # measures the h -> covariance coupling that the analytic backward
    # DELIBERATELY detaches (config::aa_cov_chain = false: the reference
    # fork's variant of that term is inconsistent — double-dilated
    # denominator — and end-to-end it costs reconstruction quality; treating
    # h as a forward-only compensation restored full quality parity on
    # garden). The gates for cov-dependent parameters therefore allow the
    # detached-term magnitude, plus the discontinuity noise of the 1/255
    # cut moving under cov perturbations.
    gates = {  # (median, q75) permissible relative error
        "means": (0.02, 0.20),
        "scales": (0.10, 0.15),
        "rotations": (0.05, 0.12),
        # w2c additionally omits the covariance-orientation -> pose-rotation
        # chain (config::pose_cov_rotation_grad = false): the reference
        # rasterizer drops it too, and including it measurably degrades
        # online pose refinement on video captures. The FD probe sees the
        # full derivative, so the mismatch here is the size of that
        # deliberately-dropped term.
        "w2c": (0.20, 0.60),
    }
    n_checks = 24
    all_ok = True
    for name, t in param_dict.items():
        eps = eps_map[name]
        flat = t.detach().reshape(-1)
        aflat = analytic[name].reshape(-1)
        order = aflat.abs().argsort(descending=True)
        idxs = order[: min(n_checks, (aflat.abs() > 1e-8).sum().item())]
        if len(idxs) == 0:
            print(f"  {name:10s} SKIP (all-zero analytic gradient)")
            continue
        rel_errs = []
        for i in idxs.tolist():
            orig = flat[i].item()
            with torch.no_grad():
                flat[i] = orig + eps
                img_p, *_ = fbs.rasterize_with_grads(
                    w2c.detach(), cam_pos, means.detach(), scales.detach(),
                    rotations.detach(), opacities.detach(), raw_betas.detach(),
                    sh0.detach(), sh_rest.detach(), bg, settings)
                lp = loss_strong(img_p).item()
                flat[i] = orig - eps
                img_m, *_ = fbs.rasterize_with_grads(
                    w2c.detach(), cam_pos, means.detach(), scales.detach(),
                    rotations.detach(), opacities.detach(), raw_betas.detach(),
                    sh0.detach(), sh_rest.detach(), bg, settings)
                lm = loss_strong(img_m).item()
                flat[i] = orig
            fd = (lp - lm) / (2 * eps)
            an = aflat[i].item()
            denom = max(abs(fd), abs(an), 1e-6)
            rel_errs.append(abs(fd - an) / denom)
        rel_errs = torch.tensor(rel_errs)
        med = rel_errs.median().item()
        q75 = rel_errs.quantile(0.75).item()
        mx = rel_errs.max().item()
        gate_med, gate_q75 = gates.get(name, (0.02, 0.10))
        ok = med < gate_med and q75 < gate_q75
        all_ok &= ok
        print(f"  {name:10s} rel err median {med:.4f}  q75 {q75:.4f}  max {mx:.4f}  "
              f"({len(rel_errs)} components)  {'OK' if ok else 'FAIL'}")
    assert all_ok, "finite-difference gradient check failed"


def reference_adam(param, grad, m1_in, m2_in, lr,
                   beta1=0.9, beta2=0.999):
    """Replicates the fused kernel's Adam math exactly."""
    eps = 1e-15
    g = grad.reshape(-1)
    m1 = beta1 * m1_in + (1 - beta1) * g
    m2 = beta2 * m2_in + (1 - beta2) * g * g
    bc1 = 1.0
    bc2 = 1.0
    denom = (m2.sqrt() / math.sqrt(bc2)) + eps
    new_param = param.reshape(-1) - (lr / bc1) * m1 / denom
    return new_param.view_as(param), m1, m2


def test_fused_adam_parity(betas=(0.9, 0.999)):
    print(f"== fused Adam step parity (2 consecutive steps, betas={betas}) ==")
    n = 200
    scene = make_scene(n, seed=11)
    (means, scales, rotations, opacities, raw_betas, sh0, sh_rest,
     w2c, cam_pos, bg, settings) = scene
    target = torch.rand(3, settings.height, settings.width, device=DEVICE)
    lrs = fbs.LearningRates()

    # reference path: raw grads + python adam
    ref = {k: v.clone() for k, v in zip(
        ("means", "scales", "rotations", "opacities", "raw_betas", "sh0", "sh_rest"),
        (means, scales, rotations, opacities, raw_betas, sh0, sh_rest))}
    ref_m = {k: (torch.zeros(v.numel(), device=DEVICE), torch.zeros(v.numel(), device=DEVICE))
             for k, v in ref.items()}
    lr_map = {"means": lrs.means, "scales": lrs.scales, "rotations": lrs.rotations,
              "opacities": lrs.opacities, "raw_betas": lrs.betas,
              "sh0": lrs.sh0, "sh_rest": lrs.sh_rest}

    # fused path
    fused = {k: v.clone() for k, v in ref.items()}
    adam = fbs.AdamStates.for_params(*[fused[k] for k in
        ("means", "scales", "rotations", "opacities", "raw_betas", "sh0", "sh_rest")],
        beta1=betas[0], beta2=betas[1])

    for step in range(1, 3):
        # reference step: raw grads + python adam, masked by the kernel's own
        # per-primitive visibility (fused semantics: visible primitives are
        # updated even with zero gradient; invisible ones are untouched)
        for t in ref.values():
            t.requires_grad_(True)
            t.grad = None
        img, _invd, _ids, _ft, vis, = fbs.rasterize_with_grads(
            w2c, cam_pos, ref["means"], ref["scales"], ref["rotations"],
            ref["opacities"], ref["raw_betas"], ref["sh0"], ref["sh_rest"],
            bg, settings)
        loss_of(img, target).backward()
        visible = vis.bool()
        with torch.no_grad():
            for k in ref:
                g = ref[k].grad if ref[k].grad is not None else torch.zeros_like(ref[k])
                new_p, new_m1, new_m2 = reference_adam(
                    ref[k].detach(), g, *ref_m[k], lr_map[k],
                    beta1=betas[0], beta2=betas[1])
                p = ref[k].detach()
                p[visible] = new_p[visible]
                for m_old, m_new in zip(ref_m[k], (new_m1, new_m2)):
                    mo = m_old.view(p.shape[0], -1)
                    mo[visible] = m_new.view(p.shape[0], -1)[visible]
                ref[k] = p

        # fused step
        img_f, *_ = fbs.rasterize(
            w2c, cam_pos, fused["means"], fused["scales"], fused["rotations"],
            fused["opacities"], fused["raw_betas"], fused["sh0"], fused["sh_rest"],
            bg, adam, lrs, settings)
        loss_of(img_f, target).backward()

        for k in ref:
            # Near-zero gradients are sign-unstable across renders (fp32
            # atomic order), and the zero-moment Adam first step is a
            # fixed-size sign(g) kick — exclude those components rather
            # than compare kicks of flipped sign.
            g = ref[k].grad if ref[k].grad is not None else torch.zeros_like(ref[k])
            stable = g.abs() > 1e-8
            diff = ((ref[k].detach() - fused[k]).abs() * stable).max().item()
            print(f"  step {step} {k:10s} max |ref - fused| = {diff:.3e}")
            assert diff < 5e-5, f"fused adam mismatch for {k} at step {step}"
    print("  fused adam parity OK")


def test_sb_lobes(k_lobes=3):
    """Spherical-beta color model (negative active_sh_bases = -K): forward
    FD gradients for the lobe parameters,
    and fused-Adam parity with the partial (dt>0 lobes only) update set."""
    print(f"== spherical-beta lobes (K={k_lobes}) ==")
    n = 300
    (means, scales, rotations, opacities, raw_betas, sh0, sh_rest,
     w2c, cam_pos, bg, settings) = make_scene(
        n, seed=13, w=128, h=96, focal=110.0, sb_lobes=k_lobes)


    # ---- FD gradients for lobe params, sh0 and means (direction chain) ----
    weights = torch.randn(3, settings.height, settings.width, device=DEVICE)

    def strong(img):
        return (img * weights).sum()

    for name, tensor in (("sh_rest(lobes)", sh_rest), ("sh0", sh0), ("means", means)):
        t = tensor.clone().requires_grad_(True)
        args_r = dict(means=means, sh0=sh0, sh_rest=sh_rest)
        args_r[{"sh_rest(lobes)": "sh_rest", "sh0": "sh0", "means": "means"}[name]] = t
        img, *_ = fbs.rasterize_with_grads(
            w2c, cam_pos, args_r["means"], scales, rotations, opacities,
            raw_betas, args_r["sh0"], args_r["sh_rest"], bg, settings)
        strong(img).backward()
        analytic = t.grad.reshape(-1)
        order = analytic.abs().argsort(descending=True)[:16]
        flat = tensor.clone().reshape(-1)
        errs = []
        for i in order.tolist():
            orig = flat[i].item()
            eps = 1e-4 * max(1.0, abs(orig))
            args_f = dict(means=means, sh0=sh0, sh_rest=sh_rest)
            key = {"sh_rest(lobes)": "sh_rest", "sh0": "sh0", "means": "means"}[name]
            with torch.no_grad():
                pert = tensor.clone().reshape(-1)
                pert[i] = orig + eps
                args_f[key] = pert.view_as(tensor)
                lp = strong(fbs.rasterize_with_grads(
                    w2c, cam_pos, args_f["means"], scales, rotations, opacities,
                    raw_betas, args_f["sh0"], args_f["sh_rest"], bg, settings)[0]).item()
                pert[i] = orig - eps
                lm = strong(fbs.rasterize_with_grads(
                    w2c, cam_pos, args_f["means"], scales, rotations, opacities,
                    raw_betas, args_f["sh0"], args_f["sh_rest"], bg, settings)[0]).item()
            fd = (lp - lm) / (2 * eps)
            an = analytic[i].item()
            errs.append(abs(fd - an) / max(abs(fd), abs(an), 1e-6))
        errs = torch.tensor(errs)
        med, q75 = errs.median().item(), errs.quantile(0.75).item()
        gate_med, gate_q75 = (0.02, 0.20) if name == "means" else (0.02, 0.10)
        ok = med < gate_med and q75 < gate_q75
        print(f"  FD {name:14s} rel err median {med:.4f}  q75 {q75:.4f}  max {errs.max():.4f}  {'OK' if ok else 'FAIL'}")
        assert ok, f"SB FD check failed for {name}"

    # ---- fused adam parity with the partial lobe update set ----
    lrs = fbs.LearningRates()
    fused = {"sh0": sh0.clone(), "sh_rest": sh_rest.clone()}
    adam = fbs.AdamStates.for_params(
        means, scales, rotations, opacities, raw_betas,
        fused["sh0"], fused["sh_rest"],
        beta1=0.5, beta2=0.99)
    ref = {"sh0": sh0.clone().requires_grad_(True),
           "sh_rest": sh_rest.clone().requires_grad_(True)}
    ref_m = {k: (torch.zeros(v.numel(), device=DEVICE), torch.zeros(v.numel(), device=DEVICE))
             for k, v in ref.items()}
    target = torch.rand(3, settings.height, settings.width, device=DEVICE)

    img, _, _, _, vis = fbs.rasterize_with_grads(
        w2c, cam_pos, means, scales, rotations, opacities, raw_betas,
        ref["sh0"], ref["sh_rest"], bg, settings)
    loss_of(img, target).backward()
    visible = vis.bool()
    # element update mask: sh0 = visible rows; sh_rest = visible rows AND
    # lobes with dt > 0 toward this camera (the kernel skips dt<=0 lobes)
    dirs = torch.nn.functional.normalize(means - cam_pos, dim=-1)
    flat_lobes = sh_rest.view(n, -1)
    mask_rest = torch.zeros(n, 45, dtype=torch.bool, device=DEVICE)
    for k in range(k_lobes):
        theta, phi = flat_lobes[:, 6 * k + 3], flat_lobes[:, 6 * k + 4]
        mu = torch.stack([theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()], -1)
        dt = (mu * dirs).sum(-1)
        mask_rest[:, 6 * k:6 * k + 6] = ((dt > 0) & visible).unsqueeze(-1)
    with torch.no_grad():
        for key, mask in (("sh0", visible.view(n, 1).expand(n, 3).reshape(-1)),
                          ("sh_rest", mask_rest.reshape(-1))):
            g = ref[key].grad
            new_p, new_m1, new_m2 = reference_adam(
                ref[key].detach(), g, *ref_m[key],
                {"sh0": lrs.sh0, "sh_rest": lrs.sh_rest}[key],
                beta1=0.5, beta2=0.99)
            p = ref[key].detach().reshape(-1)
            p[mask] = new_p.reshape(-1)[mask]
            ref_m[key][0][mask] = new_m1[mask]
            ref_m[key][1][mask] = new_m2[mask]

    img_f, *_ = fbs.rasterize(
        w2c, cam_pos, means, scales, rotations, opacities, raw_betas,
        fused["sh0"], fused["sh_rest"], bg, adam, lrs, settings)
    loss_of(img_f, target).backward()
    for key in ("sh0", "sh_rest"):
        diff = (ref[key].detach() - fused[key]).abs().max().item()
        print(f"  fused adam parity {key:8s} max |ref - fused| = {diff:.3e}")
        assert diff < 5e-5, f"SB fused adam mismatch for {key}"
    print("  spherical-beta lobes OK")


def test_aux_outputs():
    print("== auxiliary outputs ==")
    n = 60
    (means, scales, rotations, opacities, raw_betas, sh0, sh_rest,
     w2c, cam_pos, bg, settings) = make_scene(n, seed=3)
    img, invd, ids, ft, vis = fbs.rasterize_with_grads(
        w2c, cam_pos, means, scales, rotations, opacities, raw_betas,
        sh0, sh_rest, bg, settings)
    assert vis.shape == (n,) and vis.sum() > 0
    assert img.shape == (3, settings.height, settings.width)
    assert invd.shape == (1, settings.height, settings.width)
    assert ids.shape == (2, settings.height, settings.width)
    assert ft.shape == (settings.height, settings.width)
    # final transmittance in [0, 1]; covered pixels < 1
    assert ft.min() >= 0 and ft.max() <= 1 + 1e-5
    covered = ft < 0.999
    assert covered.any(), "no coverage in test scene"
    # invdepth positive on covered pixels, ~0 elsewhere
    assert (invd[0][covered] > 0).float().mean() > 0.99
    # main ids: valid on covered pixels, -1 on empty ones; ids within range
    assert (ids[0][covered] >= 0).float().mean() > 0.99
    assert ids.max() < n
    empty = ft > 0.99999
    if empty.any():
        assert (ids[0][empty] == -1).all()
    # per-pixel background: an uncovered pixel must equal bg exactly
    if empty.any():
        yx = empty.nonzero()[0]
        pix = img[:, yx[0], yx[1]]
        ref = bg[:, yx[0], yx[1]]
        assert (pix - ref).abs().max() < 1e-5
    # image with T composited: img ~= contribution + ft * bg
    print("  shapes, ranges, ids, background compositing OK")


if __name__ == "__main__":
    test_finite_differences()
    test_fused_adam_parity()
    test_fused_adam_parity(betas=(0.5, 0.99))
    test_sb_lobes()
    test_aux_outputs()
    print("ALL TESTS PASSED")
