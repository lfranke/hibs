import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METRICS = ("PSNR", "SSIM", "LPIPS", "time")
PROTOCOLS = {
    "half": {"TUM": ["--target_size", "446", "336"], "*": ["--target_width", "640"]},
    "448": {"*": ["--target_size", "448", "448", "--target_crop"]},
}


def load_scenes(overrides):
    with open(os.path.join(HERE, "scenes.json")) as handle:
        registry = json.load(handle)
    roots = dict(registry["roots"], **overrides)
    scenes = {}
    for name, scene in registry["scenes"].items():
        scenes[name] = dict(scene, source=os.path.join(roots[scene["root"]], scene["path"]))
    return scenes


def protocol_flags(name, protocol):
    flags = PROTOCOLS.get(protocol, {})
    return flags.get(name.split("/")[0], flags.get("*", []))


def prune(out_dir):
    for name in ("hierarchy.ply", "hierarchy_shape.npz", "appearance.pt"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)


def finished(out_dir):
    try:
        read(out_dir)
        return True
    except (OSError, ValueError):
        return False


def run(scene, out_dir, extra, keep=True):
    if finished(out_dir):
        return True
    command = [sys.executable, os.path.join(ROOT, "train.py"),
               "-s", os.path.abspath(scene["source"]), "-m", os.path.abspath(out_dir),
               "--test_hold", str(scene["test_hold"]), "--viewer_mode", "none"]
    if scene.get("mode") == "large_scene":
        command.append("--large_scene")
    if "images" in scene:
        command += ["-i", scene["images"]]
    command += extra
    print(" ".join(command), flush=True)
    ok = subprocess.run(command, cwd=ROOT).returncode == 0
    if ok and not keep:
        prune(out_dir)
    return ok


def read(path):
    with open(os.path.join(path, "metadata.json")) as handle:
        data = json.load(handle)
    return {k: data.get(k) for k in METRICS}


_LPIPS = None


def rescore(scene, name, path, protocol=()):
    global _LPIPS
    import cv2
    import numpy as np
    import torch
    from fused_ssim import fused_ssim
    sys.path.insert(0, ROOT)
    import hibs  # noqa: F401
    from hibs.dataset import resize_to_target
    from utils import get_image_names, psnr

    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="vgg").cuda()

    size = width = None
    crop = "--target_crop" in protocol
    if "--target_size" in protocol:
        at = protocol.index("--target_size")
        size = (int(protocol[at + 1]), int(protocol[at + 2]))
    if "--target_width" in protocol:
        width = int(protocol[protocol.index("--target_width") + 1])

    images_dir = scene.get("images", "images")
    gt_dir = os.path.join(scene["source"], images_dir)
    mask_dir = os.path.join(scene["source"], images_dir.replace("images", "masks", 1))
    if not os.path.isdir(mask_dir):
        mask_dir = None
    render_dir = os.path.join(path, "test_images")
    names = sorted(get_image_names(gt_dir))[::scene["test_hold"]]
    dark_border = name.startswith("TUM")
    totals = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0}
    for image_name in names:
        gt = cv2.imread(os.path.join(gt_dir, image_name))
        stored_mask = None
        if mask_dir:
            stored_mask = cv2.imread(
                os.path.join(mask_dir, os.path.splitext(image_name)[0] + ".png"),
                cv2.IMREAD_GRAYSCALE)
        if size or width:
            gt = resize_to_target(gt, size, width or 0, crop)
            if stored_mask is not None:
                stored_mask = resize_to_target(stored_mask, size, width or 0, crop)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        raw = cv2.imread(os.path.join(render_dir, image_name))
        image = np.zeros_like(gt) if raw is None else cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if image.shape != gt.shape:
            image = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image).permute(2, 0, 1).cuda().float() / 255
        gt = torch.from_numpy(gt).permute(2, 0, 1).cuda().float() / 255
        if stored_mask is not None:
            mask = torch.from_numpy(stored_mask).cuda() > 127
        elif dark_border:
            mask = gt.sum(0) > 0
        else:
            mask = torch.ones_like(gt[0], dtype=torch.bool)
        mask = mask.expand_as(image)
        image = image * mask
        totals["PSNR"] += psnr(image[mask], gt[mask])
        totals["SSIM"] += fused_ssim(image[None], gt[None], train=False).item()
        totals["LPIPS"] += _LPIPS(image, gt).item()
    values = {k: v / len(names) for k, v in totals.items()}
    values["time"] = read(path)["time"] if finished(path) else None
    return values


def median_run(values):
    ordered = sorted(values, key=lambda v: v["PSNR"])
    return ordered[len(ordered) // 2]


def mean(rows):
    present = [r for r in rows if all(r.get(k) is not None for k in METRICS)]
    if not present:
        return {k: None for k in METRICS}
    return {k: sum(r[k] for r in present) / len(present) for k in METRICS}


def emit(title, rows, path):
    if not rows:
        print(f"{title}: nothing to report")
        return
    header = ["row", *METRICS]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows({"row": name, **values} for name, values in rows)
    width = max(len(name) for name, _ in rows)
    print(f"\n{title}")
    print(f"{'':{width}s}  {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'time':>7s}")
    for name, values in rows:
        cells = "".join(f"{values[k]:7.3f} " if values.get(k) is not None else f"{'-':>7s} "
                        for k in METRICS)
        print(f"{name:{width}s}  {cells}")
    print(f"written to {path}")


def group_of(name):
    return name.split("/")[0] if "/" in name else name
