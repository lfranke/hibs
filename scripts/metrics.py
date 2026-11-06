"""Score a saved reconstruction by re-rendering its test views.

    python scripts/metrics.py results/main/MipNeRF360/bonsai --scene MipNeRF360/bonsai
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from runs import PROTOCOLS, emit, load_scenes, protocol_flags, read, rescore


def render_test_views(model_dir, out_dir, render_tau):
    sys.path.insert(0, ROOT)
    import hibs  # noqa: F401
    from hibs import env
    env.configure()
    from hibs.args import viewer_args
    from hibs.scene import HibsSceneModel
    env.use_backbone_cwd()
    model = HibsSceneModel.from_scene(model_dir, viewer_args({"render_tau": render_tau}))
    os.makedirs(out_dir, exist_ok=True)
    model.save_test_frames(out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="directory holding hierarchy.ply and metadata.json")
    parser.add_argument("--scene", required=True, help="scene name in scenes.json")
    parser.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS),
                        help="input resolution the reconstruction was trained at")
    parser.add_argument("--out_dir", help="where to write the re-rendered views")
    parser.add_argument("--render_tau", type=float, default=1.0)
    parser.add_argument("--stored", action="store_true",
                        help="also score the renders the run itself exported")
    cli = parser.parse_args()

    model_dir = os.path.abspath(cli.model)
    out_dir = os.path.abspath(cli.out_dir or os.path.join(model_dir, "rendered"))
    scene = load_scenes(dict(item.split("=", 1) for item in cli.root))[cli.scene]
    protocol = protocol_flags(cli.scene, cli.protocol)

    render_test_views(model_dir, os.path.join(out_dir, "test_images"), cli.render_tau)
    rows = [("re-rendered", rescore(scene, cli.scene, out_dir, protocol))]
    if cli.stored and os.path.isdir(os.path.join(model_dir, "test_images")):
        rows.append(("exported", rescore(scene, cli.scene, model_dir, protocol)))
    rows.append(("checkpoint", read(model_dir)))
    emit(cli.scene, rows, os.path.join(out_dir, "metrics.csv"))


if __name__ == "__main__":
    main()
