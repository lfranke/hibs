import torch
from scene.keyframe import Keyframe

DEPTH_MAPS = ("mono_idepth", "mono_depth_conf", "mono_idepth_dx", "mono_idepth_dy")


class HibsKeyframe(Keyframe):
    frames = None
    extractor = None

    def __init__(self, image, info, *args, **kwargs):
        self._on_disk = self._masked = False
        super().__init__(image, info, *args, **kwargs)
        frames = HibsKeyframe.frames
        self._frame = None if frames is None else frames.frame_index.get(info.get("name"))

    @property
    def image(self):
        return self._reload()[0] if self._on_disk else self._image

    @image.setter
    def image(self, value):
        self._image = value

    @property
    def mask(self):
        if self._on_disk:
            return self._reload()[1] if self._masked else None
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def feat_map(self):
        if self._feat_map is None and self._feat_dropped:
            with torch.no_grad():
                feat_map = HibsKeyframe.extractor(self.image.cuda())
            if not self._on_disk:
                self.feat_map = feat_map.cpu()
            return feat_map
        return self._feat_map

    @feat_map.setter
    def feat_map(self, value):
        self._feat_map = value
        self._feat_dropped = False

    @property
    def device(self):
        return torch.device("cpu") if self._on_disk else self._image.device

    def _reload(self):
        image, mask = HibsKeyframe.frames.frame(self._frame)
        if mask is not None and mask.dtype != torch.bool:
            mask = mask > (1 - 1e-3)
        return image, mask

    @torch.no_grad()
    def to(self, device, only_train=False):
        if self._frame is None:
            return super().to(device, only_train)
        if self.device.type == device:
            return
        if device == "cpu":
            self._masked = self._mask is not None
            self._image = self._mask = None
            self._on_disk = True
            if not only_train:
                self._feat_map, self._feat_dropped = None, True
        else:
            image, mask = self._reload()
            self._image = image.to(device)
            self._mask = None if mask is None else mask.to(device)
            self._on_disk = False
            if not only_train and self._feat_map is not None:
                self._feat_map = self._feat_map.to(device)
        for name in DEPTH_MAPS:
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, value.to(device))
        if not only_train and self.desc_kpts is not None:
            self.desc_kpts.to(device)

    def to_json(self):
        data = super().to_json()
        data["colour_corr"] = self.colour_corr.detach().cpu().numpy().tolist()
        return data

    @classmethod
    def from_json(cls, config, index, height, width):
        keyframe = super().from_json(config, index, height, width)
        stored = config.get("colour_corr")
        if stored is not None:
            with torch.no_grad():
                keyframe.colour_corr.copy_(
                    torch.tensor(stored, dtype=torch.float32, device="cuda"))
        return keyframe
