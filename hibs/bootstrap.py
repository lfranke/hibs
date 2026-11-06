import torch
from utils import get_time


class BootstrapMixin:
    def _init_reconstruction_state(self):
        super()._init_reconstruction_state()
        self._recon_start = get_time()
        self._bootstrap_ready = False

    def _process_bootstrap_frame(self, image, info, desc_kpts, vpr_feature, frameID, pbar):
        args = self.args
        size = args.num_keyframes_miniba_bootstrap
        if (args.bootstrap_span > 0 or args.bootstrap_pool > size) and not self._bootstrap_ready:
            self._bootstrap_ready = frameID == len(self.dataset) - 1 or self._pool_complete(desc_kpts)
            if self._bootstrap_ready:
                self._recon_ordered_bootstrap = False
            else:
                # stall the upstream bootstrap until our pool is complete
                args.num_keyframes_miniba_bootstrap = len(self.dataset) + 1
        try:
            pending = super()._process_bootstrap_frame(image, info, desc_kpts, vpr_feature,
                                                       frameID, pbar)
        finally:
            args.num_keyframes_miniba_bootstrap = size
        stats = self.scene_model.run_stats
        if self._recon_bootstrap_keyframe_dicts is None and "first image time" not in stats:
            stats["first image time"] = get_time() - self._recon_start
            stats["first image frames"] = frameID + 1
        return pending

    def try_bootstrap_from_shelf(self, bootstrap_shelf, force=False):
        if not self._bootstrap_ready:
            return super().try_bootstrap_from_shelf(bootstrap_shelf, force)
        size = self.args.num_keyframes_miniba_bootstrap
        if self.args.bootstrap_pool > size:
            chosen = self._spread(bootstrap_shelf, size)
        else:
            index = torch.linspace(0, len(bootstrap_shelf) - 1, size).round().long().unique()
            chosen = [bootstrap_shelf[i] for i in index.tolist()]
        return chosen, [d["kpts"] for d in chosen]

    def _pool_complete(self, desc_kpts):
        pool = self._recon_bootstrap_keyframe_dicts
        if len(pool) + 1 < max(self.args.bootstrap_pool, self.args.num_keyframes_miniba_bootstrap):
            return False
        if self.args.bootstrap_span <= 0:
            return True
        matches = self.matcher(desc_kpts, pool[0]["kpts"], self._recon_rotation_idx)
        if len(matches.kpts) < self.args.min_init_matches_count:
            return True
        span = torch.norm(matches.kpts - matches.kpts_other, dim=-1).median()
        return span >= self.args.bootstrap_span * self.min_displacement

    def _spread(self, pool, size):
        dim = pool[0]["vpr_feature"].shape[-1]
        feats = torch.stack([d["vpr_feature"].view(-1, dim) for d in pool])
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        similarity, rotation = torch.einsum("id,jqd->ijq", feats[:, 0], feats).max(dim=-1)

        def linked(k, ref):
            matches = self.matcher(pool[k]["kpts"], pool[ref]["kpts"], rotation[ref, k].item(),
                                   remove_outliers=True)
            return len(matches.kpts) >= self.args.min_bootstrap_matches_count

        chosen = [0]
        while len(chosen) < min(size, len(pool)):
            nearest = {k: max(chosen, key=lambda c: similarity[c, k].item())
                       for k in range(len(pool)) if k not in chosen}
            closeness = {k: similarity[ref, k].item() for k, ref in nearest.items()}
            candidates = [k for k, ref in nearest.items() if linked(k, ref)]
            chosen.append(min(candidates, key=closeness.get) if candidates
                          else max(closeness, key=closeness.get))
        return [pool[k] for k in sorted(chosen)]
