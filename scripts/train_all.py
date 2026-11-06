"""Reconstruct a set of scenes and score them.

    python scripts/train_all.py --datasets MipNeRF360,TUM --root i3dgs=./data/i3dgs
    python scripts/train_all.py --scenes SmallCity,CityWalk --metrics run
    python scripts/train_all.py --list
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from runs import METRICS, emit, group_of, load_scenes, mean, median_run, read, rescore, run


def select(scenes, names, datasets):
    chosen = list(scenes)
    if names:
        wanted = set(names.split(","))
        chosen = [n for n in chosen if n in wanted]
    if datasets:
        wanted = set(datasets.split(","))
        chosen = [n for n in chosen if group_of(n) in wanted]
    return chosen


def merged_group(name, merges):
    group = group_of(name)
    for target, sources in merges.items():
        if group in sources:
            return target
    return group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/all")
    parser.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--scenes", help="comma-separated scene names")
    parser.add_argument("--datasets", help="comma-separated dataset names")
    parser.add_argument("--merge", action="append", default=[], metavar="NAME=A,B",
                        help="report several datasets under one name")
    parser.add_argument("--metrics", choices=["renders", "run"], default="renders",
                        help="score the exported renders, or read each run's own metrics")
    parser.add_argument("--repeats", type=int, default=1,
                        help="run each scene this many times and report the median")
    parser.add_argument("--prune", action="store_true",
                        help="delete the reconstruction after each run, keeping metrics and renders")
    parser.add_argument("--report", action="store_true", help="collect results without running")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    cli = parser.parse_args()

    registry = load_scenes(dict(item.split("=", 1) for item in cli.root))
    if cli.list:
        print("\n".join(registry))
        return
    names = select(registry, cli.scenes, cli.datasets)
    if not names:
        raise SystemExit("no scenes selected")
    merges = {}
    for item in cli.merge:
        target, sources = item.split("=", 1)
        merges[target] = set(sources.split(","))

    failed = []
    rows = []
    for name in names:
        base = os.path.join(cli.out_dir, name)
        paths = [base] if cli.repeats == 1 else [os.path.join(base, f"r{k}")
                                                 for k in range(cli.repeats)]
        for out in paths:
            if not cli.report and not run(registry[name], out, cli.extra, keep=not cli.prune):
                failed.append(name)
        done = [p for p in paths if os.path.exists(os.path.join(p, "metadata.json"))]
        if not done:
            continue
        values = [read(p) if cli.metrics == "run" else rescore(registry[name], name, p)
                  for p in done]
        rows.append((name, median_run(values)))

    groups = {}
    for name, values in rows:
        groups.setdefault(merged_group(name, merges), []).append(values)
    for group, values in groups.items():
        if len(values) > 1:
            rows.append((f"{group} mean", mean(values)))
    if len(groups) > 1:
        rows.append(("all mean", mean([v for n, v in rows if not n.endswith(" mean")])))

    os.makedirs(cli.out_dir, exist_ok=True)
    emit("train_all", rows, os.path.join(cli.out_dir, "metrics.csv"))
    if failed:
        print("failed:", ", ".join(sorted(set(failed))))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
