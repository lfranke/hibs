import contextlib
import io
import sys

from train import ReconstructionTask
from utils import get_time

from hibs import env, patch
from hibs.bootstrap import BootstrapMixin
from hibs.keyframe import HibsKeyframe
from hibs.repair import repair

LABELS = (("nGS", "nBS"),
          ("Gaussians:", "beta primitives:"),
          ("max_n_active_gaussians", "max_n_beta_primitives"))


def relabel(text):
    for old, new in LABELS:
        text = text.replace(old, new)
    return text


class _RelabelledBar:
    def __init__(self, bar):
        self._bar = bar

    def __getattr__(self, name):
        return getattr(self._bar, name)

    def set_postfix_str(self, text, *args, **kwargs):
        return self._bar.set_postfix_str(relabel(text), *args, **kwargs)


@contextlib.contextmanager
def _relabelled_output():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield
    sys.stdout.write(relabel(buffer.getvalue()))


class HibsTask(BootstrapMixin, ReconstructionTask):
    def __init__(self, args):
        patch.install()
        env.use_backbone_cwd()
        env.disable_xformers_attention()
        super().__init__(args)
        HibsKeyframe.frames = self.dataset
        if args.ppisp:
            args.ppisp_num_frames = len(self.dataset)
            args.lr_colour_corr = 0.0

    def initialize_modules(self):
        super().initialize_modules()
        HibsKeyframe.extractor = self.dense_extractor

    def update_pbar(self, pbar, *args, **kwargs):
        return super().update_pbar(_RelabelledBar(pbar), *args, **kwargs)

    def _finalize_reconstruction(self, reconstruction_time):
        self.scene_model.run_stats.update({"frames": len(self.dataset),
                                           "unregistered frames": len(self._recon_shelf_dicts)})
        if self.args.repair_iters > 0:
            start = get_time()
            self.scene_model.join_optimization_thread()
            repair(self.scene_model, self.args.repair_iters)
            reconstruction_time += get_time() - start
        with _relabelled_output():
            super()._finalize_reconstruction(reconstruction_time)

    def run_reconstruction(self):
        super().run_reconstruction()
        # upstream returns None, which skips the viewer keep-alive loop
        return True
