import math

import torch
import torch.nn.functional as F

EXCESS_FACTOR = 1.5
DETECT_DIV = 2


@torch.no_grad()
def _view_losses(scene):
    width = max(1, scene.width // DETECT_DIV)
    height = max(1, scene.height // DETECT_DIV)
    fov_x = 2 * math.atan(scene.width / (2 * scene.f.item()))
    fov_y = 2 * math.atan(scene.height / (2 * scene.f.item()))
    losses = torch.full((len(scene.keyframes),), -1.0, device="cuda")
    for keyframe in scene.keyframes:
        if keyframe.is_test or keyframe.image is None:
            continue
        pkg = scene.render(width, height, keyframe.get_Rt().T, 1, fov_x=fov_x, fov_y=fov_y)
        image = scene.apply_compensation(keyframe, pkg["render"])
        gt = F.interpolate(keyframe.image[None].cuda(), (height, width),
                           mode="bilinear", align_corners=False)[0]
        if keyframe.mask is None:
            losses[keyframe.index] = (image - gt).abs().mean()
            continue
        mask = F.interpolate(keyframe.mask[None].float().cuda(), (height, width),
                             mode="nearest")[0] > 0.5
        if not bool(mask.any()):
            continue
        mask = mask.expand_as(image)
        losses[keyframe.index] = (image - gt).abs()[mask].mean()
    return losses


@torch.no_grad()
def _sampling_weights(losses):
    valid = losses >= 0
    if not bool(valid.any()):
        return None, 0
    median = losses[valid].median()
    excess = torch.where(valid, (losses - EXCESS_FACTOR * median).clamp_min(0.0),
                         torch.zeros_like(losses))
    ruined = int((excess > 0).sum())
    if ruined == 0:
        return None, 0
    weights = 0.5 * excess / excess.sum() + 0.5 * valid.float() / valid.sum()
    return weights, ruined


def repair(scene, n_iters):
    weights, ruined = _sampling_weights(_view_losses(scene))
    if weights is None:
        return 0
    for _ in range(n_iters):
        scene.optimization_step(keyframe_id=int(torch.multinomial(weights, 1).item()))
    return ruined
