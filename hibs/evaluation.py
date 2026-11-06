import torch

TRAINING_CUT_TAU = 2.0


class EvalMixin:
    def harmonize_test_exposure(self):
        if self.ppisp is None:
            return super().harmonize_test_exposure()
        for index, keyframe in enumerate(self.keyframes):
            if not keyframe.is_test:
                continue
            before = next((kf for kf in reversed(self.keyframes[:index]) if not kf.is_test), None)
            after = next((kf for kf in self.keyframes[index + 1:] if not kf.is_test), None)
            neighbours = [kf.index for kf in (before, after) if kf is not None]
            if not neighbours:
                continue
            with torch.no_grad():
                self.ppisp.exposure_params[keyframe.index] = self.ppisp.exposure_params[neighbours].mean()
                self.ppisp.color_params[keyframe.index] = self.ppisp.color_params[neighbours].mean(dim=0)

    def _measure(self, call):
        training = not self.inference_mode
        self.inference_mode = True
        try:
            return call()
        finally:
            self.inference_mode = not training
            if training and self.keyframes:
                self.hierarchy.update_cut_for_test(self.keyframes[-1].get_centre(), TRAINING_CUT_TAU)

    @torch.no_grad()
    def evaluate(self, eval_poses=False, with_LPIPS=False, all=False, out_dir=""):
        return self._measure(lambda: super(EvalMixin, self).evaluate(
            eval_poses, with_LPIPS, all, out_dir))

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        return self._measure(lambda: super(EvalMixin, self).save_test_frames(out_dir))
