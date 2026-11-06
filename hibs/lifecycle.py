import math

import torch
from utils import inverse_sigmoid

from hibs.kernel import BETA_SOFT, BETA_SOLID, beta_from_raw, log_scale_compensation, raw_from_beta

SOLID_RAW = float(raw_from_beta(torch.tensor(BETA_SOLID)))
SOFT_RAW = float(raw_from_beta(torch.tensor(BETA_SOFT)))
QUANTILE_POOL = 100_000
MAX_VIEWS = 10_000


def _leaf(scene):
    return (scene.parent == -1) & (scene.id_in_hierarchy == -1)


def _covisible_view_ids(scene, keyframe, k):
    ids = []
    if k <= 0:
        return ids
    for other in keyframe.desc_kpts.matches:
        if len(ids) >= k:
            break
        if 0 <= other < len(scene.keyframes) and other != keyframe.index \
                and not scene.keyframes[other].is_test:
            ids.append(other)
    return ids


@torch.no_grad()
def _scatter_view(scene, view_id, err_g=None, cnt_g=None, pkg=None):
    if pkg is None:
        pkg = scene.render_from_id(view_id)
    ids = pkg["mainGaussID"][0].flatten().long()
    owned = ids >= 0
    ids = ids[owned]
    if err_g is not None:
        gt = scene.keyframes[view_id].image
        if gt.device != err_g.device:
            gt = gt.to(err_g.device)
        err = (gt - pkg["render"]).abs().mean(dim=0).flatten()
        err_g.index_add_(0, ids, err[owned])
    if cnt_g is not None:
        cnt_g.index_add_(0, ids, torch.ones(ids.numel(), dtype=cnt_g.dtype, device=cnt_g.device))
    return pkg


@torch.no_grad()
def _update_err_stats(err_stats, err_g, cnt_g, alpha):
    seen = (cnt_g > 0).nonzero(as_tuple=True)[0]
    if seen.numel() == 0:
        return
    e = err_g[seen] / cnt_g[seen]
    views = err_stats[seen, 2]
    blend = torch.where(views == 0, torch.ones_like(views), torch.full_like(views, alpha))
    err_stats[seen, 0] = (1.0 - blend) * err_stats[seen, 0] + blend * e
    err_stats[seen, 1] = (1.0 - blend) * err_stats[seen, 1] + blend * e * e
    err_stats[seen, 2] = (views + 1.0).clamp(max=MAX_VIEWS)


def _consistency(err_stats):
    mu = err_stats[:, 0]
    var = (err_stats[:, 1] - mu * mu).clamp_min(0.0)
    return mu, (mu * mu) / (mu * mu + var + 1e-12)


def _stack_scale_factor(o_old, o_new, n_stack):
    # 3DGS-MCMC (Kheradmand et al. 2024) scale correction for n stacked copies
    denom = torch.zeros_like(o_new)
    for i in range(1, int(n_stack.max()) + 1):
        inner = torch.zeros_like(o_new)
        for k in range(i):
            inner = inner + (math.comb(i - 1, k) * (-1) ** k / math.sqrt(k + 1)) * o_new ** (k + 1)
        denom = torch.where(n_stack >= i, denom + inner, denom)
    return o_old / denom.clamp_min(1e-8)


def _step_beta(nodes, mask, delta, bound, harden):
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return 0
    raw = nodes["beta"]["val"]
    old_raw = raw[idx, 0]
    new_raw = old_raw + delta
    new_raw = new_raw.clamp_min(bound) if harden else new_raw.clamp_max(bound)
    comp = log_scale_compensation(beta_from_raw(old_raw), beta_from_raw(new_raw))
    nodes["scaling"]["val"][idx] += comp[:, None]
    raw[idx, 0] = new_raw
    return int(idx.numel())


@torch.no_grad()
def accumulate_error(scene, keyframe_id, pkg=None):
    keyframe = scene.keyframes[keyframe_id]
    nodes = scene.hierarchy.active_gaussians
    n = scene.hierarchy.active_gaussians_count
    device = nodes["xyz"]["val"].device
    err_g = torch.zeros(n, device=device)
    cnt_g = torch.zeros(n, device=device)
    _scatter_view(scene, keyframe.index, err_g, cnt_g, pkg=pkg)
    _update_err_stats(nodes["err_stats"]["val"], err_g, cnt_g, scene.args.err_stats_alpha)
    for view_id in _covisible_view_ids(scene, keyframe, scene.args.covisible_views):
        _scatter_view(scene, view_id, err_g, cnt_g)
    return err_g, cnt_g


@torch.no_grad()
def relocate(scene, keyframe_id, err_g, pkg=None):
    args = scene.args
    if not args.relocate or scene.xyz.shape[0] == 0:
        return {}
    keyframe = scene.keyframes[keyframe_id]
    if keyframe.info["is_test"]:
        return {}
    if pkg is None:
        pkg = scene.render_from_id(keyframe_id)

    nodes = scene.hierarchy.active_gaussians
    n = scene.hierarchy.active_gaussians_count
    op = torch.sigmoid(nodes["opacity"]["val"][:n, 0])
    leaf = _leaf(scene)
    visible = pkg["radii"] > 0
    recent = (scene.kf_id - keyframe.index).abs() < args.coarse_removal_kf_window

    dead = leaf & ~recent & (op < args.reloc_dead_opacity)
    alive = visible & leaf & (op > args.reloc_live_opacity)

    has3d = keyframe.desc_kpts.get_has_pt3d()
    if args.reloc_max_target_depth_mult > 0 and int(has3d.sum()) > 10:
        R = keyframe.get_R()
        t = keyframe.get_t()
        tri_z = keyframe.desc_kpts.get_pts3d()[has3d] @ R[2] + t[2]
        far = torch.quantile(tri_z[tri_z > 0], 0.75) * args.reloc_max_target_depth_mult
        alive = alive & ((scene.xyz @ R[2] + t[2]) < far)

    dead_idx = dead.nonzero(as_tuple=True)[0]
    alive_idx = alive.nonzero(as_tuple=True)[0]
    if dead_idx.numel() == 0 or alive_idx.numel() == 0:
        return {"reloc": 0}
    if 0 < args.reloc_max < dead_idx.numel():
        dead_idx = dead_idx[op[dead_idx].argsort()[:args.reloc_max]]

    device = op.device
    err_stats = nodes["err_stats"]["val"]
    mu, cons = _consistency(err_stats[alive_idx])
    gate = torch.where(err_stats[alive_idx, 2] >= args.beta_min_views, cons, torch.ones_like(cons))
    weights = err_g[alive_idx] * gate
    if args.reloc_weighting == "opacity":
        weights = op[alive_idx]
    elif args.reloc_err_quantile > 0:
        pool = weights
        if pool.numel() > QUANTILE_POOL:
            pool = pool[torch.randperm(pool.numel(), device=device)[:QUANTILE_POOL]]
        floor = torch.quantile(pool, args.reloc_err_quantile)
        weights = torch.where(weights >= floor, weights, torch.zeros_like(weights))
    weights = weights + 1e-12

    target = alive_idx[torch.multinomial(weights, dead_idx.numel(), replacement=True)]
    order = torch.argsort(target)
    target = target[order]
    source = dead_idx[order]
    uniq, inv, counts = torch.unique(target, return_inverse=True, return_counts=True)
    first = torch.zeros_like(uniq)
    first[1:] = counts.cumsum(0)[:-1]
    rank = torch.arange(target.numel(), device=device) - first[inv]
    keep = rank < args.reloc_max_stack - 1
    src = source[keep]
    tgt = target[keep]
    if src.numel() == 0:
        return {"reloc": 0}

    uniq_t, _, cnt_t = torch.unique(tgt, return_inverse=True, return_counts=True)
    n_stack = cnt_t + 1
    o_old = op[uniq_t]
    # n copies composite back to the old opacity
    o_new = 1.0 - (1.0 - o_old).clamp_min(1e-8) ** (1.0 / n_stack.float())
    o_new = o_new.clamp_min(args.prune_opacity_threshold + 0.01)
    factor = _stack_scale_factor(o_old, o_new, n_stack)
    nodes["opacity"]["val"][uniq_t, 0] = inverse_sigmoid(o_new.clamp(1e-4, 0.995))
    nodes["scaling"]["val"][uniq_t] += factor.clamp_min(1e-6).log()[:, None]

    for key in scene.hierarchy.PRIMITIVE_KEYS:
        nodes[key]["val"][src] = nodes[key]["val"][tgt]
    nodes["kf_id"]["val"][src] = nodes["kf_id"]["val"][tgt]

    if args.reloc_jitter > 0:
        mean_scale = torch.exp(nodes["scaling"]["val"][src]).mean(dim=-1, keepdim=True)
        nodes["xyz"]["val"][src] += args.reloc_jitter * mean_scale * torch.randn(
            src.numel(), 3, device=device)

    reset = torch.cat([src, uniq_t])
    for key in scene.hierarchy.PRIMITIVE_KEYS:
        node = nodes[key]
        for attr in scene.hierarchy.OPTIM_STATE_ATTRS:
            if attr in node:
                node[attr][reset] = 0
    err_stats[reset] = 0

    return {"reloc": int(src.numel()), "reloc_targets": int(uniq_t.numel())}


@torch.no_grad()
def solidify(scene, keyframe_id, cnt_g, pkg=None):
    args = scene.args
    if args.beta_solidify <= 0 or not args.beta_kernel or scene.xyz.shape[0] == 0:
        return {}
    keyframe = scene.keyframes[keyframe_id]
    if keyframe.info["is_test"]:
        return {}

    nodes = scene.hierarchy.active_gaussians
    n = scene.hierarchy.active_gaussians_count
    op = torch.sigmoid(nodes["opacity"]["val"][:n, 0])
    old = (scene.kf_id - keyframe.index).abs() >= args.coarse_removal_kf_window
    seen = (pkg["radii"] > 0) & _leaf(scene) & (cnt_g > args.beta_min_pixels)

    err_stats = nodes["err_stats"]["val"][:n]
    mu, cons = _consistency(err_stats)
    enough = err_stats[:, 2] >= args.beta_min_views

    solid = seen & old & enough & (op > args.beta_solid_opacity) & (mu < args.beta_solid_err)
    soft = seen & enough & (mu > args.beta_solid_err) & (cons < args.beta_consistency)

    return {
        "solidified": _step_beta(nodes, solid, -args.beta_solidify, SOLID_RAW, True),
        "softened": _step_beta(nodes, soft, args.beta_solidify, SOFT_RAW, False),
    }
