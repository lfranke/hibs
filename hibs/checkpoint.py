import contextlib
import os

import numpy as np
import torch
import scene.hierarchy_io as hierarchy_io

SHAPE_KEYS = ("beta", "res_native")
SIDECAR_SUFFIX = "_shape.npz"
APPEARANCE = "appearance.pt"
RENDER_KEYS = ("beta_kernel", "fused_raster", "sb_lobes", "sb_tight", "ppisp",
               "render_tau", "render_near_plane", "render_far_plane", "large_scene")


def sidecar_path(path):
    return os.path.splitext(path)[0] + SIDECAR_SUFFIX


@contextlib.contextmanager
def shape_columns(arrays):
    original = hierarchy_io._write_ply

    def write(path, elements):
        merged = np.empty(len(elements), dtype=elements.dtype.descr
                          + [(key, "<f4") for key in arrays])
        for name in elements.dtype.names:
            merged[name] = elements[name]
        for key, values in arrays.items():
            merged[key] = np.asarray(values).reshape(len(elements), -1)[:, 0]
        original(path, merged)

    hierarchy_io._write_ply = write
    try:
        yield
    finally:
        hierarchy_io._write_ply = original


def read_column(plydata, key, n):
    vertex = plydata["vertex"]
    if key not in (vertex.data.dtype.names or ()):
        return None
    return np.asarray(vertex[key], dtype=np.float32).reshape(n, -1)


def render_config(args):
    return {key: getattr(args, key) for key in RENDER_KEYS if hasattr(args, key)}


def lobes_of(metadata):
    config = (metadata or {}).get("config", {})
    return config.get("sb_lobes") or config.get("render_args", {}).get("sb_lobes", 0)


def base_dir(scene_source):
    return scene_source if os.path.isdir(scene_source) else os.path.dirname(scene_source)


def save_appearance(ppisp, num_frames, path):
    state = {key: value.detach().cpu() for key, value in ppisp.state_dict().items()}
    torch.save({"version": 1, "ppisp": {"num_cameras": 1, "num_frames": num_frames,
                                        "state_dict": state}}, path)


def load_appearance(path):
    if not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location="cuda", weights_only=False)
    if blob.get("bg_model") is not None:
        print(f"{path} holds a learned background, which this implementation does not render")
    saved = blob.get("ppisp")
    if saved is None:
        return None
    from ppisp import PPISP, PPISPConfig
    ppisp = PPISP(num_cameras=saved.get("num_cameras", 1), num_frames=saved["num_frames"],
                  config=PPISPConfig(use_controller=False))
    ppisp.load_state_dict(saved["state_dict"])
    return ppisp
