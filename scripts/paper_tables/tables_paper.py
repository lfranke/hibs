import argparse
import csv
import os
import shutil

from common import (DEEP_BLENDING, LARGE, METRICS, MIPNERF360, MIPNERF360_OTF, OTF, TAB1, TAB2,
                    TAB3, TANKS_AND_TEMPLES, finished, paper_table, run_tables, slug, write)

COMPONENTS = {
    "full": TAB1,
    "w/o SB lobes": ["--sb_lobes", "0"],
    "DBS lobe init prior": ["--lobe_init_amp", "0.1"],
    "w/o confidence": ["--no-confidence_spawn"],
    "w/o solidification": ["--beta_solidify", "0"],
    "w/o error relocation": ["--no-relocate"],
    "w/o regularizers": ["--opacity_reg", "0", "--scale_reg", "0"],
    "w/o fused rasterizer": ["--no-fused_raster"],
    "Gaussians w/o confidence": ["--no-beta_kernel", "--sb_lobes", "0", "--no-fused_raster",
                                 "--no-confidence_spawn"],
}
COMPONENTS_PAPER = {
    "layout": [("full", [r"\textbf{Full (ours)}", ""]), r"\midrule",
               ("w/o SB lobes", [r"w/o SB lobes ($\to$ SH)", r"[\cref{sec:beta_prim}]"]),
               ("DBS lobe init prior", ["DBS lobe init prior", r"[\cref{sec:beta_prim}]"]),
               ("w/o confidence", ["w/o confidence", r"[\cref{sec:spawn}]"]),
               ("w/o solidification", ["w/o solidification", r"[\cref{sec:solidity}]"]),
               ("w/o error relocation", ["w/o error relocation", r"[\cref{sec:error_reloc}]"]),
               ("w/o regularizers", ["w/o regularizers", r"[\cref{sec:optim}]"]),
               ("w/o fused rasterizer", ["w/o fused rasterizer", r"[\cref{sec:raster}]"])],
    "time": "s", "bold": {"full"}}
LARGE_SCENE = {
    "full": TAB3,
    "w/o moment merge": ["--no-hierarchy_beta_merge"],
    "w/o error repair": ["--repair_iters", "0"],
    "w/o res-weighting": ["--native_res_weight", "0"],
}
LARGE_SCENE_PAPER = {
    "layout": [("full", [r"\textbf{Full (ours)}"]), r"\midrule"]
    + [(row, [row]) for row in LARGE_SCENE if row != "full"],
    "metrics": ("PSNR", "SSIM", "LPIPS"), "bold": {"full"}}
KERNELS = {
    "Gaussian": ["--no-beta_kernel", "--sb_lobes", "0", "--no-fused_raster",
                 "--no-hierarchy_beta_merge", "--confidence_spawn", "--no-shape_confidence",
                 "--beta_solidify", "0"],
    "beta frozen at 4": ["--freeze_beta", "--hierarchy_beta_merge", "--confidence_spawn"],
    "learnable beta": ["--hierarchy_beta_merge", "--confidence_spawn", "--no-shape_confidence",
                       "--beta_solidify", "0"],
    "learnable beta, shape confidence": ["--hierarchy_beta_merge", "--confidence_spawn",
                                         "--beta_solidify", "0"],
    "full": ["--hierarchy_beta_merge", "--confidence_spawn", "--beta_solidify", "0.4"],
}
OTF_KERNELS = dict(KERNELS, full=TAB1)
KERNELS_PAPER = {"layout": [
    ("Gaussian", ["Gaussian", r"$o_{\mathrm{spawn}} $"]),
    ("beta frozen at 4", [r"$\beta$ frozen at 4", r"$o_{\mathrm{spawn}}$"]),
    ("learnable beta", [r"learnable $\beta$", r"$o_{\mathrm{spawn}} $"]),
    ("learnable beta, shape confidence",
     [r"learnable $\beta$", r"$o_{\mathrm{spawn}}$,  $\beta_{\mathrm{spawn}}$"]),
    ("full", [r"\textbf{Full (ours)}",
              r"$o_{\mathrm{spawn}}$,  $\beta_{\mathrm{spawn}}$,\text{solidify}"]),
], "time": "s", "bold": {"full"}}
KERNEL_PARTS = {
    "mipnerf360": (MIPNERF360, 5, "renders", "dataset"),
    "tandt": (TANKS_AND_TEMPLES, 3, "renders", "dataset"),
    "smallcity": (["SmallCity"], 3, "run", "scene"),
    "citywalk": (["CityWalk"], 1, "run", "scene"),
}


def main_table(cli, name):
    return paper_table(cli, name, OTF, paper={"layout": [("ours", ["Ours"])], "time": "h:mm:ss"})


def benchmark(cli, name):
    return paper_table(cli, name, MIPNERF360 + TANKS_AND_TEMPLES + DEEP_BLENDING,
                       paper={"layout": [("ours", ["", "Ours"])], "time": "h:mm:ss"})


def large_scenes(cli, name):
    return paper_table(cli, name, LARGE, metrics="run", group="scene",
                       paper={"layout": [("ours", ["Ours"])], "time": "mm:ss"})


def ablation(cli, name):
    components = paper_table(cli, f"{name}/components", OTF, rows=COMPONENTS, group="all",
                             paper=COMPONENTS_PAPER)
    large = paper_table(cli, f"{name}/large_scene", LARGE, rows=LARGE_SCENE, metrics="run",
                        group="scene", paper=LARGE_SCENE_PAPER)
    return components and large


def ablation_mipnerf360(cli, name):
    return paper_table(cli, name, MIPNERF360, rows=dict(COMPONENTS, full=TAB2), group="all",
                       paper=COMPONENTS_PAPER)


def kernel_swap(cli, name):
    if not cli.latex:
        # reuse table 5m's runs on the three MipNeRF360 scenes both tables share
        for row in [row for row, flags in OTF_KERNELS.items() if not isinstance(flags, str)]:
            for scene in MIPNERF360_OTF:
                for k in range(cli.repeats):
                    run = os.path.join(slug(row), *scene.split("/"), f"r{k}")
                    source = os.path.join(cli.out_dir, os.path.dirname(name), "mipnerf360", run)
                    target = os.path.join(cli.out_dir, name, run)
                    if finished(source) and not os.path.exists(target):
                        shutil.copytree(source, target)
    return paper_table(cli, name, OTF, rows=OTF_KERNELS, group="all", reused_repeats=5,
                       paper=KERNELS_PAPER)


def kernel_swap_datasets(cli, name):
    ok = True
    for part, (scenes, repeats, metrics, group) in KERNEL_PARTS.items():
        part_cli = argparse.Namespace(**dict(vars(cli), repeats=cli.repeats or repeats))
        ok &= paper_table(part_cli, f"{name}/{part}", scenes, rows=KERNELS, metrics=metrics,
                          group=group, paper=False)
    if cli.latex:
        return ok
    columns, table = [], {}
    # the otf part is table 5
    for part in ["mipnerf360", "tandt", "otf", "smallcity", "citywalk"]:
        path = os.path.join(cli.out_dir, name, part, "table.csv")
        if os.path.exists(path):
            for row in csv.reader(open(path)):
                if row[0] != "row":
                    column = part if row[1] == "all" else row[1]
                    columns += [column] if column not in columns else []
                    table.setdefault(row[0], {})[column] = row[2:]
    write(os.path.join(cli.out_dir, name, "table.csv"),
          ["row"] + [f"{c} {m}" for c in columns for m in METRICS],
          [[row] + [v for c in columns for v in table[row].get(c, [""] * len(METRICS))]
           for row in KERNELS if row in table])
    return ok


TABLES = {
    "1": (TAB1, "TUM, MipNeRF360 (otf) and StaticHikes", main_table, 5),
    "2": (TAB2, "MipNeRF360, Tanks and Temples and Deep Blending", benchmark, 5),
    "3": (TAB3, "large scenes, SmallCity and CityWalk", large_scenes, 5),
    "4": ("tab4_ablation", "component ablation on the otf scenes of table 1, large-scene "
          "ablation on SmallCity and CityWalk", ablation, 5),
    "4m": ("tab4_ablation/components_mipnerf360", "component ablation on the nine MipNeRF360 "
           "scenes, not in the paper and only run when named", ablation_mipnerf360, 5),
    "5": ("tab5_kernel_swap/otf", "kernel swap on the otf scenes of table 1", kernel_swap, 3),
    "5m": ("tab5_kernel_swap", "kernel swap on MipNeRF360, Tanks and Temples, SmallCity and "
           "CityWalk, not in the paper and only run when named (5, 3, 3 and 1 runs)",
           kernel_swap_datasets, None),
}

if __name__ == "__main__":
    run_tables("Tables 1-5 of the paper", TABLES, default="1,2,3,4,5")
