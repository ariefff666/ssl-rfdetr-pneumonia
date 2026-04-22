"""
dataset_ssl.py — PyTorch Dataset for DINOv2 Self-Supervised Continual Pre-training.

Returns multiple augmented views of the same image (no labels needed).
Following DINOv2 convention:
  - 2 global crops (224x224 or image_size) → fed to both teacher and student
  - N local crops (96x96) → fed to student only
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import get_ssl_global_transform, get_ssl_local_transform


class SSLChestXrayDataset(Dataset):
    """
    Self-Supervised Learning dataset for chest X-ray images.

    Loads ALL images from a directory without using any labels.
    Returns multiple augmented views of each image for contrastive/
    self-distillation learning.

    Args:
        image_dir: Path to directory containing chest X-ray images.
        image_size: Target size for global crops.
        num_local_crops: Number of local (small) crop views to generate.
        config: Augmentation config dict from YAML.
    """

    def __init__(
        self,
        image_dir: str,
        image_size: int = 512,
        num_local_crops: int = 6,
        config: dict | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.num_local_crops = num_local_crops

        # Collect all image paths
        valid_extensions = {".png", ".jpg", ".jpeg", ".dcm"}
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in valid_extensions
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                f"Check that the directory contains PNG/JPG/DCM files."
            )

        # Build transforms
        aug_config = config or {}
        self.global_transform_1 = get_ssl_global_transform(image_size, aug_config)
        self.global_transform_2 = get_ssl_global_transform(image_size, aug_config)
        self.local_transform = get_ssl_local_transform(image_size, aug_config)

        print(f"[SSLDataset] Loaded {len(self.image_paths)} images from {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image and convert to 3-channel RGB."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")

        # Handle DICOM-converted images that might be 16-bit or single channel
        if img.dtype == np.uint16:
            img = ((img / img.max()) * 255).astype(np.uint8)

        # Convert grayscale to 3-channel (DINOv2 expects RGB)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary of augmented views:
            - 'global_1': First global crop tensor
            - 'global_2': Second global crop tensor
            - 'local_crops': List of local crop tensors
        """
        img = self._load_image(self.image_paths[idx])

        # Generate 2 global views
        global_1 = self.global_transform_1(image=img)["image"]
        global_2 = self.global_transform_2(image=img)["image"]

        # Generate N local views
        local_crops = []
        for _ in range(self.num_local_crops):
            local_crop = self.local_transform(image=img)["image"]
            local_crops.append(local_crop)

        return {
            "global_1": global_1,
            "global_2": global_2,
            "local_crops": torch.stack(local_crops),  # [N, C, H, W]
        }
