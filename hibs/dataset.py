import os

import cv2
import torch
from dataloaders.image_dataset import ImageDataset


def _interpolation(image, width):
    return cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LINEAR


def resize_to_target(image, target_size=None, target_width=0, target_crop=False):
    if target_size:
        width, height = target_size
        if not target_crop:
            return cv2.resize(image, (width, height), interpolation=_interpolation(image, width))
        scale = max(width / image.shape[1], height / image.shape[0])
        scaled_width = round(image.shape[1] * scale)
        scaled = cv2.resize(image, (scaled_width, round(image.shape[0] * scale)),
                            interpolation=_interpolation(image, scaled_width))
        x = (scaled.shape[1] - width) // 2
        y = (scaled.shape[0] - height) // 2
        return scaled[y:y + height, x:x + width]
    height = round(image.shape[0] * target_width / image.shape[1])
    return cv2.resize(image, (target_width, height),
                      interpolation=_interpolation(image, target_width))


class HibsImageDataset(ImageDataset):
    def __init__(self, args):
        self.target_size = tuple(args.target_size) if args.target_size else None
        self.target_width = args.target_width
        self.target_crop = args.target_crop
        if self.target_size or self.target_width:
            args.downsampling = 1.0
        super().__init__(args)
        self.frame_index = {os.path.basename(path): i for i, path in enumerate(self.image_paths)}

    def frame(self, index):
        image = self._load_image(self.image_paths[index], cv2.IMREAD_UNCHANGED)
        mask = None
        if image.shape[0] == 4:
            image, mask = image[:3], image[3:]
        if self.mask_dir:
            mask = self._load_image(self.mask_paths[index])[:1]
        elif self.global_mask_path is not None:
            mask = self.global_mask[0]
        return image, mask

    def __getitem__(self, index):
        image, mask = self.frame(index)
        info = self.infos[os.path.basename(self.image_paths[index])]
        if mask is not None:
            info["mask"] = mask[None]
        return image.cuda()[None], info

    def _load_image(self, image_path, mode=cv2.IMREAD_COLOR):
        if not (self.target_size or self.target_width):
            return super()._load_image(image_path, mode)
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"image at {image_path} could not be loaded")
        image = resize_to_target(image, self.target_size, self.target_width, self.target_crop)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4
                             else cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255
