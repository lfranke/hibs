import argparse
import os
import sys

import args as i3dgs_args

UPSTREAM_DEFAULTS = {
    "prune_opacity_threshold": 0.02,
}

LARGE_SCENE = {
    "native_res_weight": 2.0,
    "spawn_res_guard": 0.25,
    "repair_iters": 1000,
    "auto_masks": True,
    "confidence_spawn": True,
    "beta_solidify": 0.0,
    "relocate": False,
    "ppisp": False,
    "opacity_reg": 0.0,
    "scale_reg": 0.0,
    "sb_lobes": 0,
    "prune_opacity_threshold": 0.05,
    "min_inlier_frac_of_median": 0.0,
}

LONG_SEQUENCE = {"max_active_keyframes": 120}
LONG_SEQUENCE_IMAGES = 2000


def _add(parser):
    boolean = argparse.BooleanOptionalAction

    g = parser.add_argument_group("mode")
    g.add_argument("--large_scene", action="store_true",
                   help="configuration for large scenes that exceed VRAM")

    g = parser.add_argument_group("beta primitive")
    g.add_argument("--beta_kernel", action=boolean, default=True)
    g.add_argument("--beta_init", type=float, default=4.0)
    g.add_argument("--beta_lr", type=float, default=1e-3)
    g.add_argument("--sb_lobes", type=int, default=5,
                   help="spherical-beta colour lobes; 0 selects degree-3 spherical harmonics")
    g.add_argument("--lobe_init_amp", type=float, default=0.0,
                   help="initial colour amplitude of every lobe of a new primitive")
    g.add_argument("--sb_tight", action=boolean, default=True,
                   help="size the colour buffer to the 6K floats the lobes occupy")
    g.add_argument("--fused_raster", action=boolean, default=True,
                   help="apply the Adam step inside the rasterizer backward")

    g = parser.add_argument_group("spawning")
    g.add_argument("--confidence_spawn", action=boolean, default=True,
                   help="set spawn opacity and kernel shape from measured confidence")
    g.add_argument("--shape_confidence", action=boolean, default=True,
                   help="with --confidence_spawn, also set the spawn kernel shape; off sets opacity only")
    g.add_argument("--freeze_beta", action="store_true",
                   help="keep every kernel at --beta_init, also in merged parents")
    g.add_argument("--obs_min_angle_deg", type=float, default=2.0,
                   help="triangulation angle at which depth counts as fully observable")
    g.add_argument("--confirmed_opacity", type=float, default=0.3)
    g.add_argument("--confidence_gamma", type=float, default=4.0,
                   help="exponent applied to the spawn confidence; larger reserves solid, "
                        "opaque spawns for the most consistent depths")
    g.add_argument("--spawn_res_guard", type=float, default=0.0,
                   help="reject candidates sampled below this fraction of the covering "
                        "primitive's finest observed resolution")

    g = parser.add_argument_group("solidity control")
    g.add_argument("--beta_solidify", type=float, default=0.4,
                   help="per-keyframe step on the raw shape parameter; 0 disables the controller")
    g.add_argument("--beta_solid_opacity", type=float, default=0.7)
    g.add_argument("--beta_solid_err", type=float, default=0.03)
    g.add_argument("--beta_min_pixels", type=int, default=20)
    g.add_argument("--beta_min_views", type=int, default=3)
    g.add_argument("--beta_consistency", type=float, default=0.5)

    g = parser.add_argument_group("relocation")
    g.add_argument("--relocate", action=boolean, default=True)
    g.add_argument("--reloc_dead_opacity", type=float, default=0.10)
    g.add_argument("--reloc_live_opacity", type=float, default=0.3)
    g.add_argument("--reloc_weighting", choices=("error", "opacity"), default="error",
                   help="sample relocation targets by consistency-gated error or by opacity")
    g.add_argument("--reloc_max", type=int, default=10000)
    g.add_argument("--reloc_err_quantile", type=float, default=0.9)
    g.add_argument("--reloc_max_stack", type=int, default=8)
    g.add_argument("--reloc_jitter", type=float, default=0.5)
    g.add_argument("--reloc_max_target_depth_mult", type=float, default=3.0)
    g.add_argument("--err_stats_alpha", type=float, default=0.3)
    g.add_argument("--covisible_views", type=int, default=1)

    g = parser.add_argument_group("optimization")
    g.add_argument("--opacity_reg", type=float, default=0.01)
    g.add_argument("--scale_reg", type=float, default=0.01)
    g.add_argument("--ppisp", action=boolean, default=True,
                   help="learned exposure and colour correction instead of the affine matrix")
    g.add_argument("--ppisp_lr", type=float, default=0.002)

    g = parser.add_argument_group("hierarchy and large scenes")
    g.add_argument("--hierarchy_beta_merge", action=boolean, default=True,
                   help="closed-form moment merge; off uses the opacity-weighted one")
    g.add_argument("--native_res_weight", type=float, default=0.0)
    g.add_argument("--native_res_tol", type=float, default=0.25)
    g.add_argument("--repair_iters", type=int, default=0,
                   help="post-pass budget spent on the training views that deteriorated most")
    g.add_argument("--auto_masks", action=boolean, default=False)

    g = parser.add_argument_group("input resolution")
    g.add_argument("--target_width", type=int, default=0,
                   help="resize inputs to this width, preserving aspect")
    g.add_argument("--target_size", type=int, nargs=2, default=None, metavar=("W", "H"))
    g.add_argument("--target_crop", action="store_true",
                   help="with --target_size, resize preserving aspect then centre crop")

    g = parser.add_argument_group("registration")
    g.add_argument("--min_inlier_frac_of_median", type=float, default=0.1,
                   help="retry registrations weaker than this fraction of the scene's "
                        "median inlier count")
    g.add_argument("--bootstrap_span", type=float, default=0.0,
                   help="extend an ordered bootstrap until its first and last frame lie this many "
                        "keyframe displacements apart, then take evenly spaced frames")
    g.add_argument("--bootstrap_pool", type=int, default=0,
                   help="choose the bootstrap frames for viewpoint spread among this many candidates")

    g = parser.add_argument_group("checkpoint")
    g.add_argument("--checkpoint_format", choices=("sidecar", "embedded"), default="sidecar",
                   help="sidecar stores the shape parameters in hierarchy_shape.npz, "
                        "embedded as columns of hierarchy.ply; both formats load either way")

    g = parser.add_argument_group("help")
    g.add_argument("--help_i3dgs_args", action="store_true",
                   help="also list the options of the i3dgs backbone")
    return parser


def _explicit(argv):
    names = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0][2:].replace("-", "_")
        names.add(name[3:] if name.startswith("no_") else name)
    return names


def _apply(namespace, values, explicit):
    for dest, value in values.items():
        if dest not in explicit:
            setattr(namespace, dest, value)


def _count_images(source_path, images_dir):
    try:
        entries = os.listdir(os.path.join(source_path, images_dir))
    except OSError:
        return 0
    return sum(1 for name in entries if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))


def get_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = _add(argparse.ArgumentParser(add_help=False, allow_abbrev=False, prog="hibs"))
    wants_help = "-h" in argv or "--help" in argv
    if wants_help:
        extra.print_help()
    ours, rest = extra.parse_known_args(argv)
    if wants_help and not ours.help_i3dgs_args:
        raise SystemExit(0)
    if ours.help_i3dgs_args and not wants_help:
        rest = rest + ["--help"]

    saved = sys.argv
    sys.argv = [saved[0]] + rest
    try:
        parsed = i3dgs_args.get_args()
    finally:
        sys.argv = saved
    for key, value in vars(ours).items():
        setattr(parsed, key, value)

    explicit = _explicit(argv)
    _apply(parsed, UPSTREAM_DEFAULTS, explicit)
    if parsed.large_scene:
        _apply(parsed, LARGE_SCENE, explicit)
        if _count_images(parsed.source_path, parsed.images_dir) > LONG_SEQUENCE_IMAGES:
            _apply(parsed, LONG_SEQUENCE, explicit)
    if parsed.freeze_beta:
        parsed.beta_lr, parsed.shape_confidence, parsed.beta_solidify = 0.0, False, 0.0

    if parsed.auto_masks and not parsed.masks_dir:
        candidate = parsed.images_dir.replace("images", "masks", 1)
        if os.path.isdir(os.path.join(parsed.source_path, candidate)):
            parsed.masks_dir = candidate

    parsed.source_path = os.path.abspath(parsed.source_path)
    parsed.model_path = os.path.abspath(parsed.model_path)
    return parsed


def viewer_args(overrides=None):
    parsed = get_args(["-s", "."])
    parsed.viewer_mode = "local"
    for key, value in (overrides or {}).items():
        setattr(parsed, key, value)
    return parsed
