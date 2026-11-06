import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runs import METRICS, ROOT, finished, load_scenes, protocol_flags, read, rescore, run

TUM = ["TUM/fr1_desk", "TUM/fr2_xyz", "TUM/fr3_office"]
MIPNERF360_OTF = ["MipNeRF360/garden", "MipNeRF360/bonsai", "MipNeRF360/counter"]
STATIC_HIKES = ["StaticHikes/forest1", "StaticHikes/forest2", "StaticHikes/university2"]
OTF = TUM + MIPNERF360_OTF + STATIC_HIKES
MIPNERF360 = ["MipNeRF360/" + s for s in ("bicycle", "bonsai", "counter", "flowers", "garden",
                                          "kitchen", "room", "stump", "treehill")]
TANKS_AND_TEMPLES = ["TanksAndTemples/train", "TanksAndTemples/truck"]
DEEP_BLENDING = ["DeepBlending/drjohnson", "DeepBlending/playroom"]
LARGE = ["SmallCity", "CityWalk"]
TAB1 = "tab1_tum_mipnerf360otf_statichikes"
TAB2 = "tab2_mipnerf360_tandt_deepblending"
TAB3 = "tab3_large_smallcity_citywalk"
GROUPS = {"dataset": lambda s: s.split("/")[0], "scene": lambda s: s, "all": lambda s: "all"}


def arguments(title, tables, default=None):
    width = max(len(name) for name, *_ in tables.values())
    listing = [f"  {key:>3s}  {name:{width}s}  {description}"
               + (f" ({repeats} runs)" if repeats else "")
               for key, (name, description, _, repeats) in tables.items()]
    parser = argparse.ArgumentParser(
        description=title, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="tables, written to <out_dir>/<directory>:\n" + "\n".join(listing))
    parser.add_argument("--tables", default=default or ",".join(tables),
                        help="comma-separated table numbers, see below")
    parser.add_argument("--repeats", type=int,
                        help="runs per scene, r0 to r<repeats-1>; defaults to each table's count")
    parser.add_argument("--root", action="append", default=[], metavar="NAME=PATH",
                        help="location of a dataset root named in scenes.json")
    parser.add_argument("--scenes", help="comma-separated subset of the table's scenes")
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--prune", action="store_true",
                        help="delete each reconstruction, keeping metrics and renders")
    parser.add_argument("--report", action="store_true",
                        help="collect existing runs without training")
    parser.add_argument("--latex", action="store_true",
                        help="print the table rows as LaTeX lines, without training or rescoring")
    cli = parser.parse_args()
    unknown = [key for key in cli.tables.split(",") if key not in tables]
    if unknown:
        parser.error(f"unknown tables {','.join(unknown)}; choose from {','.join(tables)}")
    return cli


def run_tables(title, tables, default=None):
    cli = arguments(title, tables, default)
    results = []
    for key in cli.tables.split(","):
        name, _, table, repeats = tables[key]
        results.append(table(argparse.Namespace(**dict(vars(cli), repeats=cli.repeats or repeats)),
                             name))
    sys.exit(0 if all(results) else 1)


def slug(text):
    return "_".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def average(values):
    return {k: statistics.mean(v[k] for v in values) for k in METRICS}


def clock(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def write(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def read_table(path):
    table = {}
    with open(path, newline="") as handle:
        for row, column, *values in list(csv.reader(handle))[1:]:
            table.setdefault(row, {})[column] = values
    return table


TIMES = {
    "h:mm:ss": lambda s: f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}",
    "mm:ss": lambda s: f"{s // 60:02d}:{s % 60:02d}",
    "m:ss": lambda s: f"{s // 60}:{s % 60:02d}",
    "s": str,
}


def latex_line(cells, comment=False):
    return ("% " if comment else "") + " & ".join(cells) + r" \\"


def metric_cells(values, metrics=METRICS, time="m:ss", bold=False):
    cells = []
    for metric in metrics:
        value = values[METRICS.index(metric)] if values else ""
        if value == "":
            cells.append("-")
            continue
        value = float(value)
        cell = (TIMES[time](round(value)) if metric == "time" else
                f"{value:.2f}" if metric == "PSNR" else f"{value:.3f}")
        cells.append(rf"\textbf{{{cell}}}" if bold else cell)
    return cells


def latex(table, layout, columns=None, metrics=METRICS, time="m:ss", bold=()):
    columns = columns or list(dict.fromkeys(c for cells in table.values() for c in cells))
    laid_out = [item[0] for item in layout if not isinstance(item, str)]

    def line(row, labels, comment=False):
        print(latex_line(labels + [cell for column in columns for cell in
                                   metric_cells(table[row].get(column), metrics, time,
                                                row in bold)], comment))

    print()
    for item in layout:
        if isinstance(item, str):
            print(item)
        elif item[0] in table:
            line(*item)
    for row in table:
        if row not in laid_out:
            line(row, [row], comment=True)


def run_path(cli, name, rows, row, scene, k):
    if isinstance(rows[row], str):
        return os.path.join(cli.out_dir, rows[row], *scene.split("/"), f"r{k}")
    parts = ([slug(row)] if len(rows) > 1 else []) + scene.split("/") + [f"r{k}"]
    return os.path.join(cli.out_dir, name, *parts)


def paper_table(cli, name, scenes, rows=None, protocol=None, metrics="renders", group="dataset",
                reused_repeats=None, paper=None):
    rows = rows or {"ours": []}
    base = os.path.join(cli.out_dir, name)
    if cli.latex:
        path = os.path.join(base, "table.csv")
        if not os.path.exists(path):
            print(f"{path} does not exist; run without --latex first")
            return False
        if paper is not False:
            latex(read_table(path), **(paper or {"layout": [(row, [row]) for row in rows]}))
        return True
    registry = load_scenes(dict(item.split("=", 1) for item in cli.root))
    if cli.scenes:
        scenes = [s for s in scenes if s in cli.scenes.split(",")]

    def path(row, scene, k):
        return run_path(cli, name, rows, row, scene, k)

    failed = []
    for k in range(cli.repeats):
        for row, flags in rows.items():
            if isinstance(flags, str):
                continue
            for scene in scenes:
                extra = protocol_flags(scene, protocol) + flags
                if not cli.report and not run(registry[scene], path(row, scene, k), extra,
                                              keep=not cli.prune):
                    failed.append(f"{row}/{scene}/r{k}")

    runs, scene_rows, table, missing = [], [], [], {}
    for row in rows:
        groups = {}
        repeats = reused_repeats if isinstance(rows[row], str) and reused_repeats else cli.repeats
        for scene in scenes:
            values = {}
            for k in range(repeats):
                out = path(row, scene, k)
                if not finished(out):
                    if isinstance(rows[row], str):
                        missing.setdefault(row, []).append(f"{scene}/r{k}")
                    continue
                values[k] = (read(out) if metrics == "run" else
                             rescore(registry[scene], scene, out, protocol_flags(scene, protocol)))
            if not values:
                continue
            runs += [[row, scene, k] + [values[k][m] for m in METRICS] for k in values]
            values = list(values.values())
            mean = average(values)
            sd = statistics.stdev(v["PSNR"] for v in values) if len(values) > 1 else 0.0
            scene_rows.append([row, scene, len(values)] + [mean[m] for m in METRICS] + [sd])
            groups.setdefault(GROUPS[group](scene), []).append(mean)
        table += [[row, label] + [average(means)[m] for m in METRICS]
                  for label, means in groups.items()]

    os.makedirs(base, exist_ok=True)
    write(os.path.join(base, "runs.csv"), ["row", "scene", "repeat", *METRICS], runs)
    write(os.path.join(base, "scenes.csv"), ["row", "scene", "n", *METRICS, "PSNR_sd"], scene_rows)
    write(os.path.join(base, "table.csv"), ["row", group, *METRICS], table)

    titles = [label if len(rows) == 1 else row if group == "all" else f"{row} | {label}"
              for row, label, *_ in table]
    width = max([len(t) for t in titles] + [8])
    print(f"\n{name}  ({cli.repeats} runs per scene)")
    print(f"{'':{width}s} {'PSNR':>7s} {'SSIM':>6s} {'LPIPS':>6s} {'time':>6s}")
    for title, (_, _, psnr, ssim, lpips, seconds) in zip(titles, table):
        print(f"{title:{width}s} {psnr:7.2f} {ssim:6.3f} {lpips:6.3f} {clock(seconds):>6s}")
    print(f"written to {base}")
    for row, source in rows.items():
        if row in missing:
            print(f"note: row '{row}' reuses the runs of {source}, which lack "
                  f"{', '.join(missing[row])}; run that table first")
    if failed:
        print("failed:", ", ".join(failed))
    return not failed
