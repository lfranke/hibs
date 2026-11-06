import argparse
import os

import hibs
from hibs import env

env.configure()

from graphdecoviewer.types import ViewerMode

from hibs.args import viewer_args
from hibs.scene import HibsSceneModel
from hibs.viewer import HibsViewer

MODES = {"local": ViewerMode.LOCAL, "client": ViewerMode.CLIENT, "server": ViewerMode.SERVER}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(title="mode", dest="mode", required=True)
    for name in ("local", "server"):
        p = sub.add_parser(name)
        p.add_argument("scene_source", help="scene directory or hierarchy PLY")
        p.add_argument("--render_tau", type=float, default=1.0)
    for name in ("client", "server"):
        p = sub.choices[name] if name in sub.choices else sub.add_parser(name)
        p.add_argument("--ip", default="0.0.0.0" if name == "server" else "127.0.0.1")
        p.add_argument("--port", type=int, default=6009)
    cli = parser.parse_args()

    if getattr(cli, "scene_source", None):
        cli.scene_source = os.path.abspath(cli.scene_source)
    env.use_backbone_cwd()
    mode = MODES[cli.mode]
    if mode is ViewerMode.CLIENT:
        viewer = HibsViewer(mode)
    else:
        viewer = HibsViewer(mode)
        viewer.import_server_modules()
        viewer.scene_model = HibsSceneModel.from_scene(
            cli.scene_source, viewer_args({"render_tau": cli.render_tau}))

    if cli.mode == "local":
        viewer.run()
    else:
        viewer.run(cli.ip, cli.port)


if __name__ == "__main__":
    main()
