# faster-beta-splatting

A CUDA rasterizer for online incremental 3D reconstruction. It renders beta
splats and, in the same backward pass, applies the Adam step to every
primitive parameter in place — no separate optimizer kernel, no gradient
tensors round-tripping through python.

## Model

Primitives are anisotropic 3D splats with a compact-support beta kernel:

```
alpha = opacity * (1 - u)^b,   u = power / 4.5,   b = 4 * exp(raw_beta)
```

`power` is the usual `0.5 * d^T conic d`, so support ends at 3 sigma. `b` is
a learnable per-primitive exponent.

Colour comes from one of two models, selected by
`RasterizerSettings.active_sh_bases`:

- **Spherical harmonics** (`active_sh_bases > 0`): deg-3 storage,
  `sh_rest` is `[N, 15, 3]`.
- **Spherical beta lobes** (`active_sh_bases = -K`): K lobes of 6 floats
  `(r, g, b, theta, phi, beta_raw)`,

  ```
  C(v) = 0.5 + C0 * sh0 + sum_k a_k * max(0, dot(mu_k, v))^(4 exp(b_k))
  ```

  The lobes live either **padded** into the deg-3 SH buffer `[N, 15, 3]`
  (leading `6K` of 45 floats used) or **tight** as `[N, K, 6]`. The kernel
  reads the per-primitive stride off the tensor and the packing inside a lobe
  is the same either way, so a state dict converts by a reshape:

  ```python
  tight = padded.reshape(N, -1)[:, :6 * K].reshape(N, K, 6)
  ```

  The tight layout moves a third less traffic and VRAM at the flagship K = 5.

Beyond Faster-GS the module adds: the beta kernel (fused into forward,
backward and the Adam step), per-pixel background maps, world-to-camera
gradients for online pose refinement, auxiliary outputs (expected inverse
depth, per-pixel best/second-best contributor ids, final transmittance),
runtime learning rates for every parameter group, and a test mode that writes
raw gradients instead of stepping. Out-of-frustum primitives are left
untouched rather than decayed.

## API

```python
import faster_beta_splatting as fbs

settings = fbs.RasterizerSettings(width, height, focal_x, focal_y,
                                  center_x, center_y, active_sh_bases=-5)
adam = fbs.AdamStates.for_params(means, scales, rotations, opacities,
                                 raw_betas, sh0, sh_rest,
                                 beta1=0.5, beta2=0.99)
lrs = fbs.LearningRates()

dummy = torch.zeros(1, device="cuda", requires_grad=True)
image, invdepth, main_ids, final_T, visibility = fbs.rasterize(
    w2c, cam_position, means, scales, rotations, opacities, raw_betas,
    sh0, sh_rest, bg, adam, lrs, settings, autograd_dummy=dummy)
(loss(image) + dummy.sum()).backward()   # renders, then steps Adam in place
```

`rasterize_with_grads(...)` is the same render but its backward stores raw
gradients on the inputs' `.grad` and applies no update — for tests and
external optimizers.

Only `image` is differentiable; `w2c` receives an autograd gradient when it
requires one. Parameters are updated in their own storage, so they must be
contiguous.

### Tensor conventions

| name | shape | notes |
| --- | --- | --- |
| `w2c` | `[3, 4]` | row-major world-to-camera, `x_cam = w2c @ [x, 1]` |
| `means` | `[N, 3]` | world space |
| `scales` | `[N, 3]` | log-scale |
| `rotations` | `[N, 4]` | unnormalized quaternion `(r, x, y, z)` |
| `opacities` | `[N, 1]` | raw, sigmoid applied inside |
| `raw_betas` | `[N, 1]` | `b = 4 * exp(raw)` |
| `sh0` | `[N, 1, 3]` | SH band 0, the diffuse term |
| `sh_rest` | `[N, 15, 3]` or `[N, K, 6]` | colour buffer, see above |
| `bg` | `[3]` or `[3, H, W]` | constant or per-pixel background |

Outputs are `image [3,H,W]`, `invdepth [1,H,W]`, `main_ids [2,H,W]`
(best/second-best contributor by blending weight, `-1` for none),
`final_transmittance [H,W]` and `visibility [N]`.

### Adam state

Each group of `AdamStates` is an `(exp_avg, exp_avg_sq)` pair of flat tensors
laid out like the parameters, so an existing optimizer's state tensors can be
handed over directly and are updated in place. The update matches classic
SparseGaussianAdam (no bias correction).

`sh_rest` additionally accepts a single **interleaved** tensor of
`2 * sh_rest.numel()` elements holding the moment pair adjacent per parameter
(read as `float2`). The lobe loop performs six moment read-modify-writes per
lobe in a memory-bound kernel, and pairing them halves those transactions; the
arithmetic is unchanged, so both layouts give bit-identical results. Pass
`interleaved_sh_rest=True` to `AdamStates.for_params` to allocate one.
Interleaved moments are only available for the lobe colour model.

`AdamStates` also carries the MCMC L1 regularizer coefficients
(`opacity_reg`, `scale_reg`) and an optional `[N]` `lr_scale` that scales a
primitive's whole update.

## Tests

```
python tests/test_faster_beta.py    # math: finite differences, Adam parity,
                                    # parity vs diff-gaussian-rasterization
python tests/test_tight_lobes.py    # layout: padded vs tight, moment layouts
```

Both need a CUDA device. The parity tests against
`diff-gaussian-rasterization` skip themselves when that package is absent.

## Provenance

Derived from [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting)
(Hahlbohm et al., CVPR 2026), Apache-2.0. The beta kernel and the
spherical-beta colour model come from Deformable Beta Splatting (Liu et al.,
SIGGRAPH 2025). See `LICENSE`.
