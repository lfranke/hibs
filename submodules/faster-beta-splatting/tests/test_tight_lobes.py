"""Test suite for the tight [N, K, 6] lobe buffer of faster-beta-splatting.

The tight layout is a stride change, not a math change: the kernels are the
same ones the padded [N, 15, 3] buffer uses, and only the per-primitive lobe
stride differs (6K floats instead of the deg-3 SH storage's 45). So the
central test is LAYOUT PARITY against the padded buffer, whose math
tests/test_faster_beta.py already validates (finite differences, Adam parity,
parity vs diff-gaussian-rasterization). If the two layouts agree on the same
scene, the tight indexing is right.

Covers, in order of importance:
 1. layout parity: padded vs tight render bit-identically
 2. moment-layout parity: two moment arrays vs one interleaved float2 pool
 3. raw-gradient parity (rasterize_with_grads) for every parameter
 4. fused-Adam parity: identical in-place parameter/moment updates
 5. finite-difference gradient check on the lobe floats themselves
 6. K sweep (1..7) and the guard rails (wrong shape / non-contiguous)
 7. aux outputs (invdepth, main ids, final transmittance, visibility)
 8. speed + memory A/B, tight vs padded

Run:  python tests/test_tight_lobes.py
"""
import math
import sys
import time

import torch

import faster_beta_splatting as fbs

torch.manual_seed(7)
DEVICE = "cuda"
PADDED_ROWS = 15  # deg-3 SH storage rows
PADDED, TIGHT = "padded", "tight"


# --------------------------------------------------------------------------
# scene construction: one scene, expressed in both layouts
# --------------------------------------------------------------------------

def make_scene(n, K, w=96, h=64, focal=80.0, seed=7, depth=(2.0, 6.0),
               scale=(0.05, 0.18), spread=0.8):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.rand(n, 3, generator=g).to(DEVICE) * 2 - 1
    means[:, 0:2] *= spread
    means[:, 2] = depth[0] + torch.rand(n, generator=g).to(DEVICE) * (depth[1] - depth[0])
    scales = torch.log(scale[0] + torch.rand(n, 3, generator=g).to(DEVICE) * (scale[1] - scale[0]))
    rotations = torch.randn(n, 4, generator=g).to(DEVICE) * 0.2
    rotations[:, 0] += 1.0
    opacities = torch.logit(0.3 + torch.rand(n, 1, generator=g).to(DEVICE) * 0.65)
    raw_betas = torch.randn(n, 1, generator=g).to(DEVICE) * 0.3  # b around 4
    sh0 = torch.randn(n, 1, 3, generator=g).to(DEVICE) * 0.4

    # lobes: (r, g, b, theta, phi, beta_raw) per lobe. Deliberately spread the
    # directions over the full sphere so a good fraction of lobes land on the
    # dt <= 0 "no contribution, no update" branch -- that branch is where a
    # stride bug would otherwise hide.
    lobe = torch.zeros(n, K, 6)
    lobe[:, :, 0:3] = torch.randn(n, K, 3, generator=g) * 0.3
    lobe[:, :, 3] = (torch.rand(n, K, generator=g) * 2 - 1).acos()
    lobe[:, :, 4] = torch.rand(n, K, generator=g) * (2 * math.pi)
    lobe[:, :, 5] = torch.randn(n, K, generator=g) * 0.3
    lobes = lobe.to(DEVICE).contiguous()

    w2c = torch.tensor([[1., 0., 0., 0.02],
                        [0., 1., 0., -0.03],
                        [0., 0., 1., 0.05]], device=DEVICE)
    cam_pos = torch.tensor([-0.02, 0.03, -0.05], device=DEVICE)
    bg = torch.rand(3, h, w, device=DEVICE) * 0.3
    return dict(n=n, K=K, w=w, h=h, focal=focal,
                means=means, scales=scales, rotations=rotations,
                opacities=opacities, raw_betas=raw_betas, sh0=sh0,
                lobes=lobes, w2c=w2c, cam_pos=cam_pos, bg=bg)


def make_tile_grid_scene(K, w=96, h=64, focal=80.0, seed=17, depth=3.0):
    """One tiny primitive centred in each 16x16 tile.

    Every primitive then produces exactly ONE instance, so the backward's
    per-primitive atomicAdd fires once and the gradients -- and with them the
    fused Adam step -- are bit-reproducible. With several instances per
    primitive the atomics sum in an unspecified order and only near-equality
    can be demanded.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    px = torch.arange(w // 16, dtype=torch.float32) * 16 + 8
    py = torch.arange(h // 16, dtype=torch.float32) * 16 + 8
    gy, gx = torch.meshgrid(py, px, indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    n = gx.numel()
    means = torch.stack([(gx - w / 2.0) * depth / focal,
                         (gy - h / 2.0) * depth / focal,
                         torch.full_like(gx, depth)], dim=-1).to(DEVICE)
    scales = torch.full((n, 3), math.log(0.02), device=DEVICE)
    rotations = torch.zeros(n, 4, device=DEVICE)
    rotations[:, 0] = 1.0
    opacities = torch.full((n, 1), torch.logit(torch.tensor(0.9)).item(), device=DEVICE)
    raw_betas = torch.randn(n, 1, generator=g).to(DEVICE) * 0.3
    sh0 = torch.randn(n, 1, 3, generator=g).to(DEVICE) * 0.4

    lobe = torch.zeros(n, K, 6)
    lobe[:, :, 0:3] = torch.randn(n, K, 3, generator=g) * 0.3
    lobe[:, :, 3] = (torch.rand(n, K, generator=g) * 2 - 1).acos()
    lobe[:, :, 4] = torch.rand(n, K, generator=g) * (2 * math.pi)
    lobe[:, :, 5] = torch.randn(n, K, generator=g) * 0.3

    w2c = torch.eye(3, 4, device=DEVICE)
    cam_pos = torch.zeros(3, device=DEVICE)
    bg = torch.rand(3, h, w, device=DEVICE) * 0.3
    return dict(n=n, K=K, w=w, h=h, focal=focal,
                means=means, scales=scales, rotations=rotations,
                opacities=opacities, raw_betas=raw_betas, sh0=sh0,
                lobes=lobe.to(DEVICE).contiguous(), w2c=w2c,
                cam_pos=cam_pos, bg=bg)


def to_padded(lobes):
    """[N, K, 6] -> the deg-3 SH storage [N, 15, 3], tail left at zero."""
    n, K, _ = lobes.shape
    padded = torch.zeros(n, PADDED_ROWS * 3, device=lobes.device)
    padded[:, :6 * K] = lobes.reshape(n, 6 * K)
    return padded.reshape(n, PADDED_ROWS, 3).contiguous()


def to_tight(padded, K):
    """[N, 15, 3] -> [N, K, 6]."""
    n = padded.shape[0]
    return padded.reshape(n, -1)[:, :6 * K].reshape(n, K, 6).contiguous()


def settings_for(s):
    return fbs.RasterizerSettings(
        width=s["w"], height=s["h"], focal_x=s["focal"], focal_y=s["focal"],
        center_x=s["w"] / 2.0, center_y=s["h"] / 2.0,
        active_sh_bases=-s["K"])


def params_for(s, layout):
    rest = s["lobes"] if layout == TIGHT else to_padded(s["lobes"])
    return (s["means"], s["scales"], s["rotations"], s["opacities"],
            s["raw_betas"], s["sh0"], rest)


def render(s, layout):
    return fbs.rasterize_with_grads(
        s["w2c"], s["cam_pos"], *params_for(s, layout), s["bg"], settings_for(s))


def adam_for(params, interleaved, beta1=0.5, beta2=0.99):
    return fbs.AdamStates.for_params(
        *params, beta1=beta1, beta2=beta2,
        interleaved_sh_rest=interleaved)


# --------------------------------------------------------------------------
# 1. layout parity
# --------------------------------------------------------------------------

def test_layout_parity():
    print("== layout parity: padded [N, 15, 3] vs tight [N, K, 6] ==")
    for K in (1, 3, 5, 7):
        s = make_scene(600, K)
        img_p, invd_p, ids_p, ft_p, vis_p = render(s, PADDED)
        img_t, invd_t, ids_t, ft_t, vis_t = render(s, TIGHT)
        # identical arithmetic in an identical order: only the address
        # computation changed, so this must be bit-exact, not merely close.
        assert torch.equal(img_p, img_t), f"K={K}: images differ"
        assert torch.equal(invd_p, invd_t), f"K={K}: inverse depth differs"
        assert torch.equal(ids_p, ids_t), f"K={K}: contributor ids differ"
        assert torch.equal(ft_p, ft_t), f"K={K}: final transmittance differs"
        assert torch.equal(vis_p, vis_t), f"K={K}: visibility differs"
        print(f"  K={K}: image, invdepth, ids, final_T, visibility bit-identical  OK")


# --------------------------------------------------------------------------
# 2. moment-layout parity
# --------------------------------------------------------------------------

def test_moment_layout_parity():
    """One fused Adam step with two moment arrays and with the interleaved
    float2 pool must produce bit-identical parameters and moments."""
    print("== moment layout parity: (exp_avg, exp_avg_sq) vs interleaved pool ==")
    K = 5
    s = make_tile_grid_scene(K)
    target = torch.rand(3, s["h"], s["w"], device=DEVICE)
    st = settings_for(s)
    lrs = fbs.LearningRates(sh_rest=2.5e-3)

    finals = {}
    for interleaved in (False, True):  # both on the same tight lobe buffer
        p = [t.clone() for t in params_for(s, TIGHT)]
        adam = adam_for(p, interleaved)
        dummy = torch.zeros(1, device=DEVICE, requires_grad=True)
        img = fbs.rasterize(s["w2c"], s["cam_pos"], *p, s["bg"], adam, lrs, st,
                            autograd_dummy=dummy)[0]
        (((img - target) ** 2).mean() + dummy.sum()).backward()
        finals[interleaved] = ([t.clone() for t in p], adam)

    names = ["means", "scales", "rotations", "opacities", "raw_betas", "sh0", "lobes"]
    for i, name in enumerate(names):
        assert torch.equal(finals[False][0][i], finals[True][0][i]),             f"{name} differs between moment layouts"
    print(f"  parameters bit-identical ({', '.join(names)})")

    arrays, pool = finals[False][1].sh_rest, finals[True][1].sh_rest
    for j, moment in enumerate(("exp_avg", "exp_avg_sq")):
        assert torch.equal(arrays[j], pool[j::2]), f"lobe {moment} differs"
    assert bool((arrays[0] != 0).any()), "no lobe was updated"
    print("  lobe moments bit-identical (de-interleaved)")


# --------------------------------------------------------------------------
# 3. raw-gradient parity
# --------------------------------------------------------------------------

_W = None

def loss_strong(image):
    """Random-weighted sum: high SNR, and it exercises every pixel."""
    global _W
    if _W is None or _W.shape != image.shape:
        g = torch.Generator(device="cpu").manual_seed(123)
        _W = torch.randn(*image.shape, generator=g).to(image.device)
    return (image * _W).sum()


def test_gradient_parity():
    print("== raw-gradient parity: padded vs tight ==")
    ok = True
    for K in (1, 5, 7):
        s = make_scene(600, K)
        grads = {}
        for layout in (PADDED, TIGHT):
            p = [t.clone().requires_grad_(True) for t in params_for(s, layout)]
            w2c = s["w2c"].clone().requires_grad_(True)
            img, *_ = fbs.rasterize_with_grads(
                w2c, s["cam_pos"], *p, s["bg"], settings_for(s))
            loss_strong(img).backward()
            names = ["means", "scales", "rotations", "opacities", "raw_betas",
                     "sh0", "lobes"]
            grads[layout] = {k: v.grad.clone() for k, v in zip(names, p)}
            grads[layout]["w2c"] = w2c.grad.clone()
        for name in grads[PADDED]:
            gp, gt = grads[PADDED][name], grads[TIGHT][name]
            if name == "lobes":
                gp = to_tight(gp, K)  # compare only the floats that exist here
            # The per-pixel backward accumulates with atomics, so the reduction
            # ORDER is not reproducible; compare against the scale of the
            # gradient rather than demanding bit-equality.
            scale = max(gp.abs().max().item(), 1e-12)
            rel = (gp - gt).abs().max().item() / scale
            good = rel < 2e-5
            ok &= good
            if not good:
                print(f"    K={K} {name:10s} rel {rel:.3e}  FAIL")
        # the padding must be untouched: the padded run writes zeros there,
        # the tight one has no such floats at all
        tail = grads[PADDED]["lobes"].reshape(s["n"], -1)[:, 6 * K:]
        tail_ok = bool((tail == 0).all())
        ok &= tail_ok
        print(f"  K={K}: all parameters within 2e-5 relative  "
              f"padding-tail-zero {tail_ok}  {'OK' if ok else 'FAIL'}")
    assert ok, "gradient parity between the two layouts failed"


# --------------------------------------------------------------------------
# 4. fused-Adam parity
# --------------------------------------------------------------------------

def test_fused_adam_parity(steps=3):
    print(f"== fused Adam parity: padded vs tight ({steps} steps) ==")
    K = 5
    s = make_scene(600, K)
    target = torch.rand(3, s["h"], s["w"], device=DEVICE)
    lrs = fbs.LearningRates(sh_rest=2.5e-3)
    st = settings_for(s)

    finals = {}
    for layout in (PADDED, TIGHT):
        p = [t.clone() for t in params_for(s, layout)]
        adam = adam_for(p, layout == TIGHT)
        for _ in range(steps):
            dummy = torch.zeros(1, device=DEVICE, requires_grad=True)
            img = fbs.rasterize(s["w2c"], s["cam_pos"], *p, s["bg"],
                                adam, lrs, st, autograd_dummy=dummy)[0]
            (((img - target) ** 2).mean() + dummy.sum()).backward()
        finals[layout] = ([t.clone() for t in p], [t.clone() for t in adam.flat()])

    names = ["means", "scales", "rotations", "opacities", "raw_betas", "sh0", "lobes"]
    ok = True
    for i, name in enumerate(names):
        a, b = finals[PADDED][0][i], finals[TIGHT][0][i]
        if name == "lobes":
            a = to_tight(a, K)
        rel = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)
        good = rel < 2e-5
        ok &= good
        print(f"  param  {name:10s} rel {rel:.3e}  {'OK' if good else 'FAIL'}")
    for i, name in enumerate(names):
        for j, moment in enumerate(("exp_avg", "exp_avg_sq")):
            a = finals[PADDED][1][2 * i + j]
            if name == "lobes":
                # the padded run keeps two separate padded arrays, the tight
                # one a single interleaved float2 pool -- de-interleave
                a = to_tight(a.reshape(s["n"], PADDED_ROWS, 3), K).reshape(-1)
                b = finals[TIGHT][1][12][j::2]
            else:
                b = finals[TIGHT][1][2 * i + j]
            rel = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)
            good = rel < 2e-5
            ok &= good
            if not good:
                print(f"  moment {name:10s} {moment:10s} rel {rel:.3e}  FAIL")
    print("  moments: all within 2e-5 relative" if ok else "  moments: FAIL")

    # the lobe padding must have received no Adam update at all (dt <= 0
    # lobes are skipped; the tail floats are never addressed)
    tail = finals[PADDED][0][6].reshape(s["n"], -1)[:, 6 * K:]
    assert bool((tail == 0).all()), "the padded run updated the padding floats"
    assert ok, "fused Adam parity between the two layouts failed"
    print("  fused adam parity OK")


# --------------------------------------------------------------------------
# 5. finite differences on the lobe floats
# --------------------------------------------------------------------------

def test_finite_differences_lobes():
    """Central finite differences on the lobe parameters themselves. The
    other parameter types are covered by tests/test_faster_beta.py via the
    parity tests above; what is new here is the [N, K, 6] indexing."""
    print("== finite differences on the lobe floats ==")
    K = 5
    s = make_scene(24, K, w=96, h=64)
    lobes = s["lobes"].clone().requires_grad_(True)
    st = settings_for(s)
    img, *_ = fbs.rasterize_with_grads(
        s["w2c"], s["cam_pos"], s["means"], s["scales"], s["rotations"],
        s["opacities"], s["raw_betas"], s["sh0"], lobes, s["bg"], st)
    loss_strong(img).backward()
    analytic = lobes.grad.clone().reshape(-1)

    flat = lobes.detach().reshape(-1)
    eps = 5e-4
    # probe the largest-gradient components of each of the 6 slots, so an
    # index-swap between (r,g,b) / theta / phi / beta_raw cannot slip through
    ok = True
    slot_names = ["r", "g", "b", "theta", "phi", "beta_raw"]
    for slot in range(6):
        sel = torch.arange(slot, analytic.numel(), 6, device=analytic.device)
        order = analytic[sel].abs().argsort(descending=True)
        idxs = sel[order][: min(8, int((analytic[sel].abs() > 1e-8).sum()))]
        if len(idxs) == 0:
            print(f"  slot {slot} ({slot_names[slot]:8s}) SKIP (zero gradient)")
            continue
        rel_errs = []
        for i in idxs.tolist():
            orig = flat[i].item()
            with torch.no_grad():
                flat[i] = orig + eps
                lp = loss_strong(fbs.rasterize_with_grads(
                    s["w2c"], s["cam_pos"], s["means"], s["scales"], s["rotations"],
                    s["opacities"], s["raw_betas"], s["sh0"], lobes.detach(),
                    s["bg"], st)[0]).item()
                flat[i] = orig - eps
                lm = loss_strong(fbs.rasterize_with_grads(
                    s["w2c"], s["cam_pos"], s["means"], s["scales"], s["rotations"],
                    s["opacities"], s["raw_betas"], s["sh0"], lobes.detach(),
                    s["bg"], st)[0]).item()
                flat[i] = orig
            fd = (lp - lm) / (2 * eps)
            an = analytic[i].item()
            rel_errs.append(abs(fd - an) / max(abs(fd), abs(an), 1e-6))
        r = torch.tensor(rel_errs)
        good = r.median().item() < 0.02 and r.quantile(0.75).item() < 0.10
        ok &= good
        print(f"  slot {slot} ({slot_names[slot]:8s}) rel err median "
              f"{r.median().item():.4f}  q75 {r.quantile(0.75).item():.4f}  "
              f"({len(r)} components)  {'OK' if good else 'FAIL'}")
    assert ok, "finite-difference check on the lobe floats failed"


# --------------------------------------------------------------------------
# 6. guard rails
# --------------------------------------------------------------------------

def _rejects(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"expected {needle!r} in {e}"
        return
    raise AssertionError(f"expected a ValueError mentioning {needle!r}")


def test_guard_rails():
    print("== guard rails ==")
    s = make_scene(64, 5)
    st = settings_for(s)
    base = (s["w2c"], s["cam_pos"], s["means"], s["scales"], s["rotations"],
            s["opacities"], s["raw_betas"], s["sh0"])

    # a tight buffer whose K disagrees with active_sh_bases is a stride error
    st_wrong = settings_for({**s, "K": 4})
    _rejects(lambda: fbs.rasterize_with_grads(*base, s["lobes"], s["bg"], st_wrong),
             "[N, K, 6]")
    print("  tight [N, 5, 6] with active_sh_bases=-4 rejected             OK")

    # a padded buffer too small for the requested lobe count
    st_big = settings_for({**s, "K": 8})
    _rejects(lambda: fbs.rasterize_with_grads(
        *base, to_padded(s["lobes"]), s["bg"], st_big), "too few")
    print("  padded [N, 15, 3] with active_sh_bases=-8 rejected           OK")

    # the SH path insists on the deg-3 storage
    st_sh = fbs.RasterizerSettings(
        width=s["w"], height=s["h"], focal_x=s["focal"], focal_y=s["focal"],
        center_x=s["w"] / 2.0, center_y=s["h"] / 2.0, active_sh_bases=16)
    _rejects(lambda: fbs.rasterize_with_grads(*base, s["lobes"], s["bg"], st_sh),
             "[N, 15, 3]")
    print("  tight buffer with SH colour rejected                         OK")

    # a non-contiguous view must be rejected: the fused backward writes in
    # place, so a silent .contiguous() copy would drop every lobe update
    nc = s["lobes"].transpose(0, 1).transpose(0, 1)[::2]
    assert not nc.is_contiguous()
    _rejects(lambda: fbs.rasterize_with_grads(
        s["w2c"], s["cam_pos"], s["means"][::2], s["scales"][::2],
        s["rotations"][::2], s["opacities"][::2], s["raw_betas"][::2],
        s["sh0"][::2], nc, s["bg"], st), "contiguous")
    print("  non-contiguous lobe buffer rejected                          OK")

    # interleaved moments are a lobe-only optimization
    p = list(params_for(s, TIGHT))
    sh_params = (p[0], p[1], p[2], p[3], p[4], p[5], to_padded(s["lobes"]))
    adam = adam_for(sh_params, True)
    _rejects(lambda: fbs.rasterize(s["w2c"], s["cam_pos"], *sh_params, s["bg"],
                                   adam, fbs.LearningRates(), st_sh),
             "interleaved")
    print("  interleaved moments with SH colour rejected                  OK")

    # a moment pool that is too short would be an out-of-bounds write
    short = adam_for(p, True)
    short.sh_rest = short.sh_rest[:-2]
    _rejects(lambda: fbs.rasterize(s["w2c"], s["cam_pos"], *p, s["bg"],
                                   short, fbs.LearningRates(), st),
             "interleaved")
    print("  short interleaved moment pool rejected                       OK")


# --------------------------------------------------------------------------
# 7. auxiliary outputs
# --------------------------------------------------------------------------

def test_aux_outputs():
    print("== auxiliary outputs ==")
    s = make_scene(2000, 5, w=128, h=96)
    img, invd, ids, ft, vis = render(s, TIGHT)
    assert img.shape == (3, s["h"], s["w"]), img.shape
    assert invd.shape == (1, s["h"], s["w"]), invd.shape
    assert ids.shape == (2, s["h"], s["w"]), ids.shape
    assert ft.shape == (s["h"], s["w"]), ft.shape
    assert vis.shape == (s["n"],), vis.shape
    assert bool((ft >= 0).all() and (ft <= 1).all()), "final_T out of [0, 1]"
    assert bool((ids >= -1).all() and (ids < s["n"]).all()), "contributor id out of range"
    assert bool((invd >= 0).all()), "negative inverse depth"
    assert int(vis.sum()) > 0, "nothing visible"

    # a fully transparent scene must show exactly the background
    s2 = make_scene(2000, 5, w=128, h=96)
    s2["opacities"] = torch.full_like(s2["opacities"], -30.0)
    img_bg, _, ids_bg, ft_bg, _ = render(s2, TIGHT)
    assert torch.allclose(img_bg, s2["bg"], atol=1e-5), "background not composited"
    assert bool((ft_bg > 0.999).all()), "transmittance not ~1 for a clear scene"
    print(f"  shapes, ranges, ids, background compositing OK "
          f"({int(vis.sum())}/{s['n']} visible)")


# --------------------------------------------------------------------------
# 8. speed + memory
# --------------------------------------------------------------------------

def _bench(s, layout, iters=50, warmup=15):
    p = [t.clone() for t in params_for(s, layout)]
    adam = adam_for(p, layout == TIGHT)
    lrs = fbs.LearningRates()
    st = settings_for(s)
    bg = torch.zeros(3, device=DEVICE)
    dummy = torch.zeros(1, device=DEVICE, requires_grad=True)
    t0 = 0.0
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        img = fbs.rasterize(s["w2c"], s["cam_pos"], *p, bg, adam, lrs, st,
                            autograd_dummy=dummy)[0]
        (img.sum() + dummy.sum()).backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3
    # parameter + exp_avg + exp_avg_sq for the lobe buffer
    mib = p[6].numel() * 4 * 3 / 2 ** 20
    return dt, mib


def test_speed(repeats=3):
    """A/B at realistic reconstruction sizes. The two layouts are interleaved
    and the MEDIAN of several repeats is reported: a single A-then-B pass
    attributes clock/thermal drift to whichever layout ran second."""
    print("== speed / memory A/B (fwd + bwd + fused adam) ==")
    K = 5
    print(f"  {'N':>9} {'res':>11} | {'padded':>9} {'tight':>9} {'speedup':>8} | "
          f"{'lobe VRAM pad/tight':>22}")
    for n, (w, h) in [(300_000, (980, 720)), (1_000_000, (980, 720)),
                      (2_000_000, (1600, 1060))]:
        # scales small enough that the per-tile instance count stays in the
        # range a real reconstruction produces (large blobs make the blend
        # dominate and hide the per-primitive traffic this layout changes)
        s = make_scene(n, K, w=w, h=h, focal=h * 1.1, scale=(0.01, 0.05))
        pad, tight = [], []
        for _ in range(repeats):
            t_pad, mib_pad = _bench(s, PADDED)
            t_tight, mib_tight = _bench(s, TIGHT)
            pad.append(t_pad)
            tight.append(t_tight)
        t_pad = sorted(pad)[len(pad) // 2]
        t_tight = sorted(tight)[len(tight) // 2]
        print(f"  {n:>9} {w}x{h:<6} | {t_pad:8.3f}ms {t_tight:8.3f}ms "
              f"{t_pad / t_tight:7.3f}x | {mib_pad:8.0f} / {mib_tight:.0f} MiB")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        sys.exit(1)
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    test_layout_parity()
    test_moment_layout_parity()
    test_gradient_parity()
    test_fused_adam_parity()
    test_finite_differences_lobes()
    test_guard_rails()
    test_aux_outputs()
    test_speed()
    print("\nALL TESTS PASSED")
