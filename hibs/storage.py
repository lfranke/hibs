import contextlib
import os

import numpy as np
import torch
import scene.hierarchy
from scene.hierarchy import HierarchyStructure

from hibs import checkpoint
from hibs.checkpoint import SHAPE_KEYS, sidecar_path
from hibs.kernel import raw_from_beta
from hibs.merge import merge_beta_groups


def f_rest_rows(args):
    k = getattr(args, "sb_lobes", 0) or 0
    if k > 0 and getattr(args, "sb_tight", True):
        return 2 * k
    return (args.sh_degree + 1) ** 2 - 1


def init_nodes(args, device):
    n = getattr(args, "num_neighbors_for_hierarchy", 4)
    return {
        "xyz": {"val": torch.empty(0, 3, device=device)},
        "f_dc": {"val": torch.empty(0, 1, 3, device=device)},
        "f_rest": {"val": torch.empty(0, f_rest_rows(args), 3, device=device)},
        "scaling": {"val": torch.empty(0, 3, device=device)},
        "rotation": {"val": torch.empty(0, 4, device=device)},
        "opacity": {"val": torch.empty(0, 1, device=device)},
        "beta": {"val": torch.empty(0, 1, device=device)},
        "res_native": {"val": torch.empty(0, device=device)},
        "err_stats": {"val": torch.empty(0, 3, device=device)},
        "kf_id": {"val": torch.empty(0, dtype=torch.int32, device=device)},
        "children": {"val": torch.empty(0, n, dtype=torch.int32, device=device)},
        "parent": {"val": torch.empty(0, dtype=torch.int32, device=device)},
        "id_in_hierarchy": {"val": torch.empty(0, dtype=torch.int32, device=device)},
    }


class BetaHierarchy(HierarchyStructure):
    PRIMITIVE_KEYS = ("xyz", "f_dc", "f_rest", "scaling", "rotation", "opacity", "beta")
    OPTIM_STATE_ATTRS = ("exp_avg", "exp_avg_sq", "moments")

    def init_internal(self, args, inference_mode):
        self.sb_lobes = (getattr(args, "sb_lobes", 0) or 0) if getattr(args, "sb_tight", True) else 0
        self.ply_sh_degree = args.sh_degree
        self.embed_shape = getattr(args, "checkpoint_format", "sidecar") == "embedded"
        self.beta_merge = getattr(args, "hierarchy_beta_merge", False) and getattr(args, "beta_kernel", True)
        self.frozen_beta = (float(raw_from_beta(torch.tensor(args.beta_init)))
                            if getattr(args, "freeze_beta", False) else None)
        super().init_internal(args, inference_mode)

        active = self.active_gaussians
        active["beta"]["lr"] = getattr(args, "beta_lr", 0.001)
        if self.sb_lobes > 0:
            active["f_rest"]["lr"] = args.feature_lr
        if self.sb_lobes > 0 and getattr(args, "fused_raster", False) and not inference_mode:
            node = active["f_rest"]
            node.pop("exp_avg", None)
            node.pop("exp_avg_sq", None)
            node["moments"] = torch.zeros(*node["val"].shape, 2, device=node["val"].device)

    @property
    def f_rest_rows(self):
        return self.active_gaussians["f_rest"]["val"].shape[1]

    @property
    def lobes(self):
        k = self.sb_lobes
        if k <= 0 or self.f_rest_rows != 2 * k:
            raise ValueError(f"lobe view needs {2 * k} f_rest rows for K={k}, have {self.f_rest_rows}")
        return self.f_rest.view(-1, k, 6)

    @torch.no_grad()
    def _compute_merged_gaussians(self, selected_nn_idx, cam_centre, screen_size):
        n = self.active_gaussians_count
        child_beta = self.active_gaussians["beta"]["val"][:n][selected_nn_idx]
        if self.beta_merge:
            merged = merge_beta_groups(
                self.xyz[selected_nn_idx],
                self.scaling[selected_nn_idx],
                self.rotation[selected_nn_idx],
                self.opacity[selected_nn_idx].squeeze(-1),
                child_beta.squeeze(-1),
            )
            w = merged["mass_weights"][..., None, None]
            log_scale = merged["log_scale"]
            dist = torch.linalg.vector_norm(merged["xyz"] - cam_centre[None], dim=-1)
            oversized = self.f * torch.exp(log_scale) / dist[..., None] > self.hierarchy_max_screen_size
            if oversized.any():
                log_scale[oversized] = torch.log(
                    self.hierarchy_max_screen_size * dist[..., None].expand(-1, 3)[oversized] / self.f)
            out = {
                "xyz": merged["xyz"],
                "f_dc": (self.f_dc[selected_nn_idx] * w).sum(dim=1),
                "f_rest": (self.f_rest[selected_nn_idx] * w).sum(dim=1),
                "opacity": merged["opacity_raw"],
                "scaling": log_scale,
                "rotation": merged["quat"],
                "beta": merged["beta_raw"],
            }
        else:
            out = super()._compute_merged_gaussians(selected_nn_idx, cam_centre, screen_size)
            w = self.opacity[selected_nn_idx][..., 0] * screen_size[selected_nn_idx] ** 2
            w = (w / w.sum(dim=-1, keepdim=True)).unsqueeze(-1)
            out["beta"] = (child_beta * w).sum(dim=1)

        if self.frozen_beta is not None:
            out["beta"] = torch.full_like(out["beta"], self.frozen_beta)
        b = out["xyz"].shape[0]
        out["res_native"] = torch.zeros(b, device=out["xyz"].device)
        out["err_stats"] = torch.zeros(b, 3, device=out["xyz"].device)
        return out

    @contextlib.contextmanager
    def _padded_f_rest(self):
        rows = (self.ply_sh_degree + 1) ** 2 - 1
        live = self.f_rest_rows
        if live >= rows:
            yield
            return
        with self.lock:
            saved = []
            for pool, count in ((self.gaussian_nodes, self.cpu_nodes_count),
                                (self.active_gaussians, self.active_gaussians_count)):
                node = pool["f_rest"]
                saved.append((node, node["val"]))
                padded = torch.zeros(count, rows, 3, dtype=node["val"].dtype, device=node["val"].device)
                padded[:, :live] = node["val"][:count].detach()
                node["val"] = padded
            try:
                yield
            finally:
                for node, original in saved:
                    node["val"] = original

    def _merged_shape_arrays(self):
        n_cpu, n_act = self.cpu_nodes_count, self.active_gaussians_count
        out = {}
        for key in SHAPE_KEYS:
            stored = self.gaussian_nodes[key]["val"]
            cols = 1 if stored.dim() == 1 else stored.shape[1]
            cpu = stored[:n_cpu].reshape(n_cpu, cols).detach().cpu().numpy().copy()
            if n_act == 0:
                out[key] = cpu
                continue
            ids = self.active_gaussians["id_in_hierarchy"]["val"][:n_act].cpu().numpy()
            act = self.active_gaussians[key]["val"][:n_act].reshape(n_act, cols).detach().cpu().numpy()
            known = ids != -1
            cpu[ids[known]] = act[known]
            out[key] = np.concatenate([cpu, act[~known]])
        return out

    def save_ply(self, path):
        # RAM spikes here on large scenes (~22 GiB on CityWalk) as meshio copies the node table;
        # writing the binary PLY directly from the record array would avoid it.
        arrays = self._merged_shape_arrays()
        with self._padded_f_rest():
            if self.embed_shape:
                with checkpoint.shape_columns(arrays):
                    super().save_ply(path)
                return
            super().save_ply(path)
        np.savez_compressed(sidecar_path(path), **arrays)

    @classmethod
    def _parse_ply_data(cls, plydata, sh_degree=None, num_neighbors_for_hierarchy=None):
        data, sh_degree, neighbors = super()._parse_ply_data(plydata, sh_degree, num_neighbors_for_hierarchy)
        n = data["xyz"].shape[0]
        for key, shape in (("beta", (n, 1)), ("res_native", (n,))):
            stored = checkpoint.read_column(plydata, key, n)
            data[key] = (np.zeros(shape, dtype=np.float32) if stored is None
                         else stored.reshape(shape))
        data["err_stats"] = np.zeros((n, 3), dtype=np.float32)
        return data, sh_degree, neighbors

    @classmethod
    def from_ply(cls, path, **kwargs):
        hierarchy = super().from_ply(path, **kwargs)
        n = hierarchy.cpu_nodes_count
        sidecar = sidecar_path(path)
        if os.path.exists(sidecar):
            stored = np.load(sidecar)
            for key in SHAPE_KEYS:
                if key not in stored:
                    continue
                values = torch.from_numpy(stored[key][:n]).float()
                node = hierarchy.gaussian_nodes[key]["val"]
                node[:n] = values.reshape(-1) if node.dim() == 1 else values
        n_act = hierarchy.active_gaussians_count
        if n_act == 0:
            return hierarchy
        roots = (hierarchy.gaussian_nodes["parent"]["val"][:n] == -1).nonzero(as_tuple=True)[0]
        for key in SHAPE_KEYS:
            active = hierarchy.active_gaussians[key]["val"]
            picked = hierarchy.gaussian_nodes[key]["val"][roots[:n_act]].cuda()
            active[:n_act] = picked.reshape(-1) if active.dim() == 1 else picked
        return hierarchy


scene.hierarchy.init_nodes = init_nodes
