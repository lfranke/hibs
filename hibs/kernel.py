import math

import torch

BETA_BASE = 4.0
BETA_SOLID = 1.0
BETA_SOFT = 5.0
KAPPA_MIN = 75.0 / 7.0
KAPPA_MAX = 15.0


def beta_from_raw(raw):
    # Eq. 1 in the paper
    return BETA_BASE * torch.exp(raw)


def raw_from_beta(beta):
    return torch.log(beta / BETA_BASE)


def half_max(beta):
    return 1.0 - torch.pow(torch.as_tensor(0.5, dtype=beta.dtype, device=beta.device), 1.0 / beta)


def log_scale_compensation(beta_old, beta_new):
    # Eq. 7 in the paper
    return 0.5 * torch.log(half_max(beta_old) / half_max(beta_new))


# Supplement, closed-form merge
def shrink(beta):
    return 4.5 / (beta + 2.5)


def kurtosis(beta):
    return 15.0 * (beta + 2.5) / (beta + 3.5)


def beta_from_kurtosis(kappa):
    return (7.0 * kappa - 75.0) / (2.0 * (15.0 - kappa))


def observability(depth, baseline, min_angle_deg):
    # Eq. 3 in the paper
    if baseline <= 0:
        return torch.zeros_like(depth)
    return (baseline / depth.clamp_min(1e-6) / math.radians(max(min_angle_deg, 1e-3))).clamp(0.0, 1.0)
