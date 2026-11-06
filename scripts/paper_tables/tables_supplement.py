import csv
import json
import math
import os
import statistics

import numpy as np

from common import (DEEP_BLENDING, METRICS, MIPNERF360, MIPNERF360_OTF, OTF, STATIC_HIKES, TAB1,
                    TAB2, TANKS_AND_TEMPLES, TUM, finished, latex_line, metric_cells, paper_table,
                    read_table, run_path, run_tables, write)
from tables_paper import benchmark

COLOUR_BASIS = "supp6_colour_basis_deepblending"
SH = ["--sb_lobes", "0"]
POSE_ERRORS = ("ATE", "R.APE", "T.RPE", "R.RPE")
SCENE_LABELS = {"MipNeRF360/garden": r"\textsc{Garden}", "MipNeRF360/counter": r"\textsc{Counter}",
                "MipNeRF360/bonsai": r"\textsc{Bonsai}", "TUM/fr1_desk": r"\textsc{fr1/desk}",
                "StaticHikes/university2": r"\textsc{University2}"}


def scene_rows(cli, name):
    path = os.path.join(cli.out_dir, name, "scenes.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def runs_of(cli, name, rows, row, scene):
    paths = [run_path(cli, name, rows, row, scene, k) for k in range(cli.repeats)]
    result = []
    for path in filter(finished, paths):
        with open(os.path.join(path, "metadata.json")) as handle:
            result.append(json.load(handle))
    return result


def mean_of(values):
    values = list(values)
    return statistics.mean(values) if values else ""


def number(value, spec):
    return "-" if value == "" else format(value, spec)


def emit(cli, name, filename, header, rows, paper_lines=()):
    if cli.latex:
        if paper_lines:
            print("\n" + "\n".join(paper_lines))
        return
    base = os.path.join(cli.out_dir, name)
    os.makedirs(base, exist_ok=True)
    write(os.path.join(base, filename), header, rows)
    cells = [header] + [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
                        for row in rows]
    widths = [max(len(line[i]) for line in cells) for i in range(len(header))]
    print(f"\n{name}")
    for line in cells:
        print("  ".join(cell.rjust(width) for cell, width in zip(line, widths)))
    print(f"written to {os.path.join(base, filename)}")


def centres(poses):
    return -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])


def similarity_transform(source, target):
    mean_s, mean_t = source.mean(0), target.mean(0)
    u, d, vt = np.linalg.svd((target - mean_t).T @ (source - mean_s) / len(source))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ flip @ vt
    scale = (d * np.diag(flip)).sum() / ((source - mean_s) ** 2).sum(1).mean()
    return scale, rotation, mean_t - scale * rotation @ mean_s


def angle(rotation):
    return math.degrees(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))


def rms(values):
    return math.sqrt(np.mean(np.square(values)))


def pose_errors(metadata):
    frames = sorted((kf["info"]["name"], kf["Rt"], kf["info"]["gt_Rt"])
                    for kf in metadata["keyframes"] if "gt_Rt" in kf["info"])
    estimate = np.array([frame[1] for frame in frames], dtype=np.float64)
    truth = np.array([frame[2] for frame in frames], dtype=np.float64)
    scale, rotation, shift = similarity_transform(centres(estimate), centres(truth))
    aligned = centres(estimate) @ (scale * rotation).T + shift
    extent = np.linalg.norm(centres(truth).max(0) - centres(truth).min(0))
    steps = []
    for i in range(len(frames) - 1):
        step = estimate[i + 1] @ np.linalg.inv(estimate[i])
        step[:3, 3] *= scale
        steps.append(np.linalg.inv(truth[i + 1] @ np.linalg.inv(truth[i])) @ step)
    return {"ATE": 100 * rms(np.linalg.norm(aligned - centres(truth), axis=1)) / extent,
            "R.APE": rms([angle(e[:3, :3] @ rotation.T @ t[:3, :3].T)
                          for e, t in zip(estimate, truth)]),
            "T.RPE": 100 * rms([np.linalg.norm(s[:3, 3]) for s in steps]) / extent,
            "R.RPE": rms([angle(s[:3, :3]) for s in steps])}


def low_resolution(cli, name):
    return paper_table(cli, name, OTF, protocol="half",
                       paper={"layout": [("ours", ["Ours"])], "time": "mm:ss"})


def resolution_448(cli, name):
    return paper_table(cli, name, MIPNERF360_OTF, protocol="448",
                       paper={"layout": [("ours", ["Ours"])], "time": "m:ss"})


def colour_basis(cli, name):
    ok = paper_table(cli, name, DEEP_BLENDING, rows={"SB lobes": TAB2, "SH": SH},
                     group="scene", paper=False)
    if cli.latex and ok:
        table = read_table(os.path.join(cli.out_dir, name, "table.csv"))
        lines = []
        for scene in DEEP_BLENDING:
            lines += [r"\midrule"] if lines else []
            lines += [latex_line([scene.split("/")[1] if i == 0 else "", row]
                                 + metric_cells(table.get(row, {}).get(scene), METRICS[:3]))
                      for i, row in enumerate(("SB lobes", "SH"))]
        print("\n" + "\n".join(lines))
    return ok


def sh_vs_lobes(cli, name):
    parts = {"deepblending": (DEEP_BLENDING, TAB2, COLOUR_BASIS + "/sh"),
             "mipnerf360_tandt": (MIPNERF360 + TANKS_AND_TEMPLES, TAB2, SH),
             "tum_statichikes": (TUM + STATIC_HIKES, TAB1, SH)}
    ok, table = True, []
    for part, (scenes, lobes, sh) in parts.items():
        ok &= paper_table(cli, f"{name}/{part}", scenes, rows={"SB lobes": lobes, "SH": sh},
                          paper=False)
        lines = scene_rows(cli, f"{name}/{part}")
        sh_lines = {line["scene"]: line for line in lines if line["row"] == "SH"}
        for line in lines:
            if line["row"] == "SB lobes" and line["scene"] in sh_lines:
                table.append(line["scene"].split("/")
                             + [float(line[m]) - float(sh_lines[line["scene"]][m])
                                for m in ("PSNR", "SSIM", "LPIPS")])
    deltas = {"/".join(line[:2]): line[2:] for line in table}
    paper_order = [("db", DEEP_BLENDING),
                   ("MipNeRF360", ["MipNeRF360/" + s for s in ("garden", "bonsai", "counter",
                                                               "bicycle", "kitchen", "room",
                                                               "stump", "flowers", "treehill")]),
                   ("TUM", TUM), (r"T\&T", TANKS_AND_TEMPLES), ("StaticHikes", STATIC_HIKES)]
    paper_lines = []
    for dataset, scenes in paper_order:
        paper_lines += [r"\midrule"] if paper_lines else []
        for scene in filter(deltas.__contains__, scenes):
            label = scene.split("/")[1]
            label = "TUM " + label.split("_")[1] if dataset == "TUM" else label
            psnr, ssim, lpips = deltas[scene]
            paper_lines.append(latex_line([dataset, label, f"{psnr:+.2f}", f"{ssim:+.3f}",
                                           f"{lpips:+.3f}"]))
    emit(cli, name, "table.csv", ["dataset", "scene", "delta_PSNR", "delta_SSIM", "delta_LPIPS"],
         table, paper_lines)
    return ok


def tum_pose(cli, name):
    runs, table = [], []
    for scene in TUM:
        scores = [pose_errors(metadata) for metadata in runs_of(cli, TAB1, {"ours": []}, "ours", scene)]
        runs += [[scene, k] + [score[key] for key in POSE_ERRORS] for k, score in enumerate(scores)]
        if scores:
            table.append([scene, len(scores)]
                         + [mean_of(score[key] for score in scores) for key in POSE_ERRORS])
    emit(cli, name, "runs.csv", ["scene", "run", *POSE_ERRORS], runs)
    emit(cli, name, "table.csv", ["scene", "n", *POSE_ERRORS], table,
         [latex_line([scene.split("/")[1].replace("_", "/"), "Ours", f"{ate:.2f}", f"{r_ape:.2f}",
                      f"{t_rpe:.3f}", f"{r_rpe:.3f}"])
          for scene, _, ate, r_ape, t_rpe, r_rpe in table])
    return len(table) == len(TUM)


def bootstrap(cli, name):
    scene = "TUM/fr1_desk"
    rows = {"ordered first-8": TAB1,
            "ordered + parallax floor": ["--bootstrap_span", "1"],
            "spread, pool 30": ["--bootstrap_pool", "30"]}
    labels = {"ordered + parallax floor": "ordered + parallax"}
    ok = paper_table(cli, name, [scene], rows=rows, group="scene", paper=False)
    psnr = {line["row"]: float(line["PSNR"]) for line in scene_rows(cli, name)}
    table = []
    for row in rows:
        runs = runs_of(cli, name, rows, row, scene)
        if not runs:
            continue
        scores = [pose_errors(metadata) for metadata in runs]
        table.append([row, len(runs), mean_of(score["ATE"] for score in scores),
                      mean_of(score["R.APE"] for score in scores), psnr.get(row, ""),
                      mean_of(m["first image time"] for m in runs if "first image time" in m),
                      mean_of(m["config"]["f"] for m in runs)])
    emit(cli, name, "bootstrap.csv",
         ["bootstrap", "n", "ATE", "R.APE", "PSNR", "first_image_s", "f"], table,
         [latex_line([labels.get(row, row), number(ate, ".2f"), number(r_ape, ".2f"),
                      number(psnr, ".2f"), number(first, ".1f"), number(f, ".0f")])
          for row, _, ate, r_ape, psnr, first, f in table])
    return ok


def solidity(cli, name):
    scene = "MipNeRF360/garden"
    rows = {"0.1": ["--beta_solidify", "0.1"], "0.2": ["--beta_solidify", "0.2"],
            "0.4 (ours)": TAB1, "0.8": ["--beta_solidify", "0.8"]}
    ok = paper_table(cli, name, [scene], rows=rows, group="scene", paper=False)
    lines = {line["row"]: line for line in scene_rows(cli, name)}
    table = []
    for row in rows:
        if row in lines:
            runs = runs_of(cli, name, rows, row, scene)
            table.append([row, float(lines[row]["PSNR"]), float(lines[row]["LPIPS"])]
                         + [mean_of(m[key] for m in runs if key in m)
                            for key in ("solidified", "softened")])
    emit(cli, name, "solidity.csv", ["step", "PSNR", "LPIPS", "solidified", "softened"], table,
         [latex_line([row, f"{psnr:.2f}", f"{lpips:.3f}"]
                     + ["-" if n == "" else f"{n / 1000:.0f}k" for n in counts])
          for row, psnr, lpips, *counts in table])
    return ok


def relocation(cli, name):
    rows = {"full": TAB1,
            "w/o error weighting": ["--reloc_weighting", "opacity"],
            "w/o relocation": ["--no-relocate"]}
    paper = {"layout": [("full", ["Full (error-weighted + consistency gate)"]),
                        ("w/o error weighting", ["w/o error weighting (opacity-sampled)"]),
                        ("w/o relocation", ["w/o relocation"])],
             "metrics": METRICS[:3]}
    return paper_table(cli, name, ["TUM/fr2_xyz"], rows=rows, group="scene",
                       paper=paper)


def sensitivity(cli, name):
    parts = {
        "MipNeRF360/garden": {
            "o_solid 0.7 -> 0.5": ["--beta_solid_opacity", "0.5"],
            "o_solid 0.7 -> 0.9": ["--beta_solid_opacity", "0.9"],
            "theta_0 2 -> 1": ["--obs_min_angle_deg", "1"],
            "theta_0 2 -> 4": ["--obs_min_angle_deg", "4"],
            "o_1 0.3 -> 0.2": ["--confirmed_opacity", "0.2"],
            "o_1 0.3 -> 0.5": ["--confirmed_opacity", "0.5"],
        },
        "StaticHikes/university2": {
            "theta_0 2 -> 1": ["--obs_min_angle_deg", "1"],
            "theta_0 2 -> 4": ["--obs_min_angle_deg", "4"],
        },
    }
    constants = {"o_solid": r"$o_{\mathrm{solid}}$ (solidify opacity gate)",
                 "theta_0": r"$\theta_0$ (observability angle)",
                 "o_1": r"$o_{\mathrm{1}}$ (verified spawn opacity)"}
    ok, table = True, []
    for scene, probes in parts.items():
        part = f"{name}/{scene.split('/')[1]}"
        ok &= paper_table(cli, part, [scene], rows=dict(default=TAB1, **probes), group="scene",
                          paper=False)
        lines = {line["row"]: line for line in scene_rows(cli, part)}
        if "default" not in lines:
            continue
        base = lines["default"]
        for probe in filter(lines.__contains__, probes):
            line = lines[probe]
            table.append([scene, probe, float(line["PSNR"]), float(line["PSNR_sd"]),
                          float(line["PSNR"]) - float(base["PSNR"]),
                          math.hypot(float(line["PSNR_sd"]), float(base["PSNR_sd"]))])
    paper_lines, previous = [], None
    for scene, probe, psnr, psnr_sd, delta, delta_sd in table:
        if previous not in (None, scene):
            paper_lines.append(r"\midrule")
        previous = scene
        constant, change = probe.split(" ", 1)
        old, new = change.split(" -> ")
        if constant == "theta_0":
            old, new = rf"${old}^\circ$", rf"${new}^\circ$"
        paper_lines.append(latex_line([SCENE_LABELS[scene], constants[constant],
                                       rf"{old} $\to$ {new}", rf"{psnr:.2f}$\pm${psnr_sd:.2f}",
                                       rf"{delta:+.2f}\,{{\scriptsize$\pm${delta_sd:.2f}}}"]))
    emit(cli, name, "table.csv", ["scene", "probe", "PSNR", "PSNR_sd", "delta", "delta_sd"], table,
         paper_lines)
    return ok


def unordered(cli, name):
    scenes = ["MipNeRF360/garden", "MipNeRF360/counter", "MipNeRF360/bonsai", "TUM/fr1_desk"]
    rows = {"ordered": TAB1, "shuffled": ["--shuffle"]}
    ok = paper_table(cli, name, scenes, rows=rows, group="scene", paper=False)
    table = []
    for line in scene_rows(cli, name):
        runs = runs_of(cli, name, rows, line["row"], line["scene"])
        table.append([line["scene"], line["row"]]
                     + [float(line[m]) for m in ("PSNR", "SSIM", "LPIPS")]
                     + [mean_of(1 - m["unregistered frames"] / m["frames"]
                                for m in runs if "frames" in m)])
    paper_lines = []
    for scene in scenes:
        lines = [line for line in table if line[0] == scene]
        paper_lines += [r"\midrule"] if paper_lines and lines else []
        paper_lines += [latex_line([SCENE_LABELS[scene] if i == 0 else "", order,
                                    f"{psnr:.2f}", f"{ssim:.3f}", f"{lpips:.3f}",
                                    "-" if registered == "" else rf"{100 * registered:.1f}\%"])
                        for i, (_, order, psnr, ssim, lpips, registered) in enumerate(lines)]
    emit(cli, name, "unordered.csv", ["scene", "order", "PSNR", "SSIM", "LPIPS", "registered"],
         table, paper_lines)
    return ok


TABLES = {
    "4": ("supp4_lowres_tum_mipnerf360otf_statichikes",
          "low resolution: TUM at 446x336, MipNeRF360 (otf) and StaticHikes at 640 px",
          low_resolution, 5),
    "5": ("supp5_res448_mipnerf360otf", "448x448 input, MipNeRF360 (otf)", resolution_448, 5),
    "6": (COLOUR_BASIS, "colour basis, SB lobes against SH, on Deep Blending", colour_basis, 3),
    "7": ("supp7_sh_vs_lobes", "per-scene gain of SB lobes over SH on all datasets",
          sh_vs_lobes, 3),
    "8": ("supp8_tum_pose", "pose accuracy on TUM, from the runs of table 1", tum_pose, 3),
    "9": ("supp9_bootstrap_fr1desk", "bootstrap variants on TUM fr1/desk", bootstrap, 3),
    "10": ("supp10_relocation_fr2xyz", "error relocation on TUM fr2/xyz", relocation, 3),
    "11": ("supp11_solidity_garden", "solidification step on MipNeRF360 garden", solidity, 3),
    "12": ("supp12_sensitivity", "sensitivity to constants on MipNeRF360 garden and StaticHikes "
           "university2", sensitivity, 3),
    "14": (TAB2, "full 3DGS benchmark, our row is that of table 2 of the paper", benchmark, 5),
    "15": ("supp15_unordered", "ordered against shuffled input on MipNeRF360 garden, counter, "
           "bonsai and TUM fr1/desk", unordered, 3),
}

if __name__ == "__main__":
    run_tables("Supplemental tables 4-12, 14 and 15", TABLES)
