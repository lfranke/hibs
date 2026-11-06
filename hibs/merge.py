import torch
from utils import inverse_sigmoid, quat_to_rotmat, rotmat_to_quat

from hibs.kernel import KAPPA_MAX, KAPPA_MIN, beta_from_kurtosis, beta_from_raw, shrink

EPS = 1e-12
EIGH_CHUNK = 16384


def _eigh(m):
    shape, n = m.shape[:-2], m.shape[-1]
    flat = m.reshape(-1, n, n)
    flat = torch.nan_to_num(0.5 * (flat + flat.transpose(-1, -2)), nan=0.0, posinf=0.0, neginf=0.0)
    if flat.shape[0] <= EIGH_CHUNK:
        vals, vecs = torch.linalg.eigh(flat)
    else:
        chunks = [torch.linalg.eigh(flat[i:i + EIGH_CHUNK]) for i in range(0, flat.shape[0], EIGH_CHUNK)]
        vals = torch.cat([c[0] for c in chunks], 0)
        vecs = torch.cat([c[1] for c in chunks], 0)
    return vals.reshape(*shape, n), vecs.reshape(*shape, n, n)


def _sanitize(mu, scale, quat, opacity, beta_raw, raw_clamp):
    norm = quat.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    quat = torch.where(norm < 1e-6, identity, quat / norm.clamp_min(EPS))
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    scale = torch.nan_to_num(scale, nan=EPS, posinf=1e6, neginf=EPS).clamp_min(EPS)
    opacity = torch.nan_to_num(opacity, nan=EPS, posinf=1.0, neginf=EPS).clamp(EPS, 1.0)
    beta_raw = torch.nan_to_num(beta_raw, nan=0.0, posinf=raw_clamp, neginf=-raw_clamp)
    return mu, scale, quat, opacity, beta_raw


@torch.no_grad()
def merge_beta_groups(mu_i, scale_i, quat_i, opacity_i, beta_raw_i, raw_clamp=4.0, opacity_max=0.99):
    mu_i, scale_i, quat_i, opacity_i, beta_raw_i = _sanitize(
        mu_i, scale_i, quat_i, opacity_i, beta_raw_i, raw_clamp)

    rot = quat_to_rotmat(quat_i)
    sigma = rot @ ((scale_i ** 2)[..., None] * rot.transpose(-1, -2))
    sqrt_det = scale_i.prod(dim=-1).clamp_min(EPS)

    beta_i = beta_from_raw(beta_raw_i)
    c_i = shrink(beta_i)
    # E[r^2] and E[r^4] of the 3D beta kernel (1 - r^2/9)^beta
    m2_i = 13.5 / (beta_i + 2.5)
    m4_i = 303.75 / ((beta_i + 2.5) * (beta_i + 3.5))

    lg15 = torch.lgamma(torch.tensor(1.5, device=mu_i.device, dtype=mu_i.dtype))
    log_norm = lg15 + torch.lgamma(beta_i + 1.0) - torch.lgamma(beta_i + 2.5)
    mass_i = opacity_i * torch.exp(log_norm) * sqrt_det
    mass = mass_i.sum(dim=1)
    w = mass_i / mass.clamp_min(EPS)[:, None]

    # Eq. 9 in the paper
    mu = (w[..., None] * mu_i).sum(1)
    delta = mu_i - mu[:, None, :]
    cov = (w[..., None, None] * (c_i[..., None, None] * sigma
                                 + delta[..., :, None] * delta[..., None, :])).sum(1)
    eye = torch.eye(3, device=cov.device, dtype=cov.dtype)
    jitter = 1e-8 * cov.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(EPS)
    cov = cov + jitter[:, None, None] * eye
    cov_inv = torch.linalg.inv_ex(cov)[0]

    vals, vecs = _eigh(sigma)
    sqrt_sigma = vecs @ (torch.sqrt(vals.clamp_min(EPS))[..., None] * vecs.transpose(-1, -2))
    cov_inv_delta = (cov_inv[:, None] @ delta[..., None]).squeeze(-1)
    a_i = (delta * cov_inv_delta).sum(-1)
    v2 = ((sqrt_sigma @ cov_inv_delta[..., None]).squeeze(-1) ** 2).sum(-1)
    m_i = sqrt_sigma @ cov_inv[:, None] @ sqrt_sigma
    tr = m_i.diagonal(dim1=-2, dim2=-1).sum(-1)
    tr2 = (m_i * m_i).sum(dim=(-1, -2))
    kappa = (w * (a_i ** 2 + (4.0 * m2_i / 3.0) * v2
                  + (m4_i / 15.0) * (tr ** 2 + 2.0 * tr2)
                  + (2.0 * a_i * m2_i / 3.0) * tr)).sum(1)

    # Eq. 10 in the paper
    beta = beta_from_kurtosis(kappa.clamp(KAPPA_MIN + 1e-6, KAPPA_MAX - 1e-6))
    beta_raw = torch.log((beta / 4.0).clamp_min(EPS)).clamp(-raw_clamp, raw_clamp)
    beta = beta_from_raw(beta_raw)

    sigma_star = cov / shrink(beta)[:, None, None]
    vals_s, vecs_s = _eigh(sigma_star)
    scale = torch.sqrt(vals_s.clamp_min(EPS))
    log_scale = torch.log(scale)
    flip = torch.ones_like(vecs_s)
    flip[..., 0] = torch.where(torch.linalg.det(vecs_s) < 0, -1.0, 1.0)[:, None]
    quat = rotmat_to_quat(vecs_s * flip)

    log_norm_star = lg15 + torch.lgamma(beta + 1.0) - torch.lgamma(beta + 2.5)
    opacity = (mass / (torch.exp(log_norm_star) * scale.prod(dim=-1).clamp_min(EPS))
               ).clamp(1e-4, opacity_max)
    opacity_raw = inverse_sigmoid(opacity[:, None])

    bad = ~(torch.isfinite(mu).all(-1) & torch.isfinite(log_scale).all(-1)
            & torch.isfinite(opacity_raw[:, 0]) & torch.isfinite(beta_raw))
    if bad.any():
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        log_scale = torch.nan_to_num(log_scale, nan=-5.0, posinf=-5.0, neginf=-5.0)
        quat = torch.nan_to_num(quat, nan=0.0)
        quat[bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype)
        beta_raw = torch.nan_to_num(beta_raw, nan=0.0).clamp(-raw_clamp, raw_clamp)
        opacity_raw = torch.where(bad[:, None], torch.full_like(opacity_raw, -12.0),
                                  torch.nan_to_num(opacity_raw, nan=-12.0))

    return {
        "xyz": mu,
        "opacity_raw": opacity_raw,
        "log_scale": log_scale,
        "quat": quat,
        "beta_raw": beta_raw[:, None],
        "mass_weights": torch.nan_to_num(w, nan=0.0),
        "kappa_mix": torch.nan_to_num(kappa, nan=KAPPA_MAX),
        "unrepresentable": torch.nan_to_num((KAPPA_MIN - kappa).clamp_min(0.0), nan=0.0),
    }
