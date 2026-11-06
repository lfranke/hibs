import torch
from poses.pose_initializer import PoseInitializer

HISTORY = 200
WARMUP = 10
FORCE_ADD_MIN_POINTS = 6


class GatedPoseInitializer(PoseInitializer):
    def __init__(self, width, height, triangulator, matcher, max_pnp_error,
                 ba_outlier_threshold, args):
        super().__init__(width, height, triangulator, matcher, max_pnp_error,
                         ba_outlier_threshold, args)
        self.min_inlier_frac = args.min_inlier_frac_of_median
        self._inliers = torch.empty(0, device="cuda")
        self._count = None

    def _drop(self, keyframes, index):
        for keyframe in keyframes:
            keyframe.desc_kpts.matches.pop(index, None)

    def _threshold(self):
        if self.min_inlier_frac <= 0 or self._inliers.numel() < WARMUP:
            return float(self.min_num_inliers)
        return max(float(self._inliers.median()) * self.min_inlier_frac, self.min_num_inliers)

    @torch.no_grad()
    def _estimate_pose(self, xyz, uvs, cam_indices, kpt_idx, keyframes, index, force_add):
        if force_add and len(xyz) < FORCE_ADD_MIN_POINTS:
            self._drop(keyframes, index)
            return None, None, None, None, "no_inliers"
        result = super()._estimate_pose(xyz, uvs, cam_indices, kpt_idx, keyframes, index, force_add)
        self._count = None if result[0] is None else int(result[4].sum())
        return result

    @torch.no_grad()
    def initialize_incremental(self, keyframes, curr_desc_kpts, index, force_add, is_test,
                               obs_only=False):
        self._count = None
        baseline_thresh = self.baseline_thresh
        if obs_only:
            self.baseline_thresh = 0.0
        try:
            Rt, reason = super().initialize_incremental(
                keyframes, curr_desc_kpts, index, force_add, is_test)
        finally:
            self.baseline_thresh = baseline_thresh
        if Rt is None or self._count is None:
            return Rt, reason
        if not force_add and not is_test and self._count <= self._threshold():
            self._drop(keyframes, index)
            return None, "weak_inliers"
        value = torch.tensor([float(self._count)], device="cuda")
        self._inliers = torch.cat([self._inliers, value])[-HISTORY:]
        return Rt, reason
