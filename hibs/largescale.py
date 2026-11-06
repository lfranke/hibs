import torch


@torch.no_grad()
def view_density(scene, keyframe, z=None, visibility=None):
    n = scene.hierarchy.active_gaussians_count
    if n == 0:
        return None
    R, t = keyframe.get_R(), keyframe.get_t()
    if z is None:
        z = scene.xyz @ R[2] + t[2]
    depth = z.clamp_min(1e-6)
    u = scene.f * (scene.xyz @ R[0] + t[0]) / depth + scene.width / 2
    v = scene.f * (scene.xyz @ R[1] + t[1]) / depth + scene.height / 2
    pad = 0.05 * scene.width
    seen = (z > 1e-6) & (u >= -pad) & (u < scene.width + pad) & (v >= -pad) & (v < scene.height + pad)
    if visibility is not None and visibility.shape[0] == n:
        seen &= visibility.reshape(n, -1)[:, 0] > 0
    return scene.f / depth * seen


@torch.no_grad()
def step_scale(scene, density):
    n = scene.hierarchy.active_gaussians_count
    native = scene.hierarchy.active_gaussians["res_native"]["val"][:n]
    tol = scene.args.native_res_tol
    band = (density > 0) & (density < tol * native)
    if not bool(band.any()):
        return None
    scale = torch.ones_like(density)
    w = (density[band] / (tol * native[band])).pow(scene.args.native_res_weight)
    scale[band] = torch.where(w < 0.9, w, torch.ones_like(w))
    return scale


@torch.no_grad()
def update_res_native(scene, density):
    n = scene.hierarchy.active_gaussians_count
    native = scene.hierarchy.active_gaussians["res_native"]["val"]
    native[:n] = torch.maximum(native[:n], density)
