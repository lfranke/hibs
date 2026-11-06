import torch
from utils import RGB2SH

from hibs.kernel import BETA_SOFT, BETA_SOLID, beta_from_raw

VIRIDIS = torch.tensor([
    (0.2670, 0.0049, 0.3294), (0.2770, 0.0503, 0.3757), (0.2823, 0.0950, 0.4173),
    (0.2829, 0.1359, 0.4534), (0.2788, 0.1755, 0.4834), (0.2706, 0.2141, 0.5071),
    (0.2590, 0.2515, 0.5247), (0.2450, 0.2877, 0.5373), (0.2297, 0.3224, 0.5457),
    (0.2143, 0.3556, 0.5512), (0.1994, 0.3876, 0.5546), (0.1856, 0.4186, 0.5568),
    (0.1727, 0.4488, 0.5579), (0.1607, 0.4785, 0.5581), (0.1490, 0.5081, 0.5573),
    (0.1378, 0.5375, 0.5549), (0.1276, 0.5669, 0.5506), (0.1206, 0.5964, 0.5436),
    (0.1206, 0.6258, 0.5335), (0.1323, 0.6550, 0.5197), (0.1579, 0.6838, 0.5017),
    (0.1966, 0.7118, 0.4792), (0.2461, 0.7389, 0.4520), (0.3041, 0.7647, 0.4199),
    (0.3692, 0.7889, 0.3829), (0.4401, 0.8111, 0.3410), (0.5160, 0.8312, 0.2943),
    (0.5958, 0.8487, 0.2433), (0.6785, 0.8637, 0.1895), (0.7624, 0.8764, 0.1371),
    (0.8456, 0.8873, 0.0997), (0.9261, 0.8973, 0.1041), (0.9932, 0.9062, 0.1439)])
UNDER = (0.135, 0.0, 0.20)
OVER = (1.0, 1.0, 1.0)
EPS = 1e-4


def confidence_colors(beta):
    lut = VIRIDIS.to(beta.device)
    t = ((beta - BETA_SOLID) / (BETA_SOFT - BETA_SOLID)).clamp(0, 1) * (lut.shape[0] - 1)
    low = t.floor().long().clamp(0, lut.shape[0] - 2)
    frac = (t - low.float())[:, None]
    colors = (1 - frac) * lut[low] + frac * lut[low + 1]
    for bound, value in ((beta > BETA_SOFT + EPS, OVER), (beta < BETA_SOLID - EPS, UNDER)):
        flag = torch.tensor(value, device=beta.device).expand_as(colors)
        colors = torch.where(bound[:, None], flag, colors)
    return colors


class confidence_field:
    def __init__(self, scene):
        self.scene = scene
        self.active = False

    def __enter__(self):
        nodes = self.scene.hierarchy.active_gaussians
        self.n = self.scene.hierarchy.active_gaussians_count
        if "beta" not in nodes or self.n == 0:
            return False
        self.scene.lock.acquire()
        try:
            with torch.no_grad():
                colors = confidence_colors(beta_from_raw(nodes["beta"]["val"][:self.n, 0]))
                self.f_dc = nodes["f_dc"]["val"]
                self.f_rest = nodes["f_rest"]["val"]
                self.dc_backup = self.f_dc[:self.n].clone()
                self.rest_backup = self.f_rest[:self.n].clone()
                self.f_dc[:self.n] = RGB2SH(colors)[:, None, :]
                self.f_rest[:self.n] = 0
            self.active = True
        except Exception:
            self.scene.lock.release()
            raise
        return True

    def __exit__(self, *exc):
        if not self.active:
            return False
        try:
            with torch.no_grad():
                self.f_dc[:self.n] = self.dc_backup
                self.f_rest[:self.n] = self.rest_backup
        finally:
            self.active = False
            self.scene.lock.release()
        return False
