import json
import os

import torch
from scene.scene_model import SceneModel

from hibs import checkpoint, patch
from hibs.burst import BurstMixin
from hibs.evaluation import EvalMixin
from hibs.render import RenderMixin
from hibs.spawn import SpawnMixin, MvsDisagreement


class HibsSceneModel(RenderMixin, SpawnMixin, BurstMixin, EvalMixin, SceneModel):
    def __init__(self, width, height, args, matcher=None, inference_mode=False):
        patch.install()
        self.ppisp = None
        self.ppisp_optimizer = None
        self._kf_proba_np = None
        self.run_stats = {}
        super().__init__(width, height, args, matcher, inference_mode)
        if args.sb_lobes > 0:
            self.active_sh_degree = -args.sb_lobes
        if not inference_mode:
            self.guided_mvs = MvsDisagreement(self.guided_mvs)
            if args.ppisp:
                from ppisp import PPISP, PPISPConfig
                self.ppisp = PPISP(num_cameras=1, num_frames=args.ppisp_num_frames,
                                   config=PPISPConfig(use_controller=False,
                                                      ppisp_lr=args.ppisp_lr,
                                                      scheduler_base_lr=args.ppisp_lr))
                self.ppisp_optimizer = self.ppisp.create_optimizers()[0]

    def add_keyframe(self, keyframe, vpr_feature=None, f=None):
        self._kf_proba_np = None
        return super().add_keyframe(keyframe, vpr_feature, f)

    def update_training_proba(self):
        self._kf_proba_np = None
        return super().update_training_proba()

    def load_scene(self, scene_source, metadata=None):
        if metadata is None:
            metadata = self._load_metadata_for_source(scene_source)
        super().load_scene(scene_source, metadata)
        lobes = checkpoint.lobes_of(metadata)
        if lobes:
            self.active_sh_degree = -lobes
            # the PLY pads the lobes to the full SH buffer, not the tight 2K-row layout
            self.hierarchy.sb_lobes = 0
        base = checkpoint.base_dir(scene_source)
        self.ppisp = checkpoint.load_appearance(os.path.join(base, checkpoint.APPEARANCE))

    def save(self, path, reconstruction_time=0, n_frames=0):
        metrics = super().save(path, reconstruction_time, n_frames)
        meta_path = os.path.join(path, "metadata.json")
        if path and os.path.exists(meta_path):
            with open(meta_path) as handle:
                metadata = json.load(handle)
            metadata.update(self.run_stats)
            if self.args.checkpoint_format == "embedded":
                metadata["config"]["render_args"] = checkpoint.render_config(self.args)
            else:
                metadata["config"]["sb_lobes"] = self.args.sb_lobes
            with open(meta_path, "w") as handle:
                json.dump(metadata, handle, indent=4)
        if path and self.ppisp is not None:
            checkpoint.save_appearance(self.ppisp, self.args.ppisp_num_frames,
                                       os.path.join(path, checkpoint.APPEARANCE))
        return metrics
