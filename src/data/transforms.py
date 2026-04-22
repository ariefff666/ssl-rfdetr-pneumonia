"""
transforms.py — Augmentation pipelines for SSL pre-training and detection.

Medical X-ray specific considerations:
- No aggressive color jittering (grayscale intensity is diagnostically meaningful)
- Moderate geometric transforms (horizontal flip OK, vertical flip NOT OK for chest X-rays)
- CLAHE for contrast enhancement
- Gaussian noise to simulate sensor variation
"""

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def get_ssl_global_transform(image_size: int = 512, config: dict | None = None) -> A.Compose:
    """
    Global crop transform for SSL teacher/student framework.

    Produces a large crop (covering 40-100% of the image area) that captures
    the overall lung structure. Used for both teacher and student views.
    """
    cfg = config or {}
    scale = cfg.get("global_crop_scale", [0.4, 1.0])
    brightness = cfg.get("brightness", 0.2)
    contrast = cfg.get("contrast", 0.2)
    blur_prob = cfg.get("gaussian_blur_prob", 0.5)

    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=scale,
            ratio=(0.75, 1.333),
            interpolation=cv2.INTER_LANCZOS4,
        ),
        A.HorizontalFlip(p=0.5),
        # Medical-safe brightness/contrast (mild)
        A.RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=contrast,
            p=0.8,
        ),
        # CLAHE for contrast enhancement (very common in medical imaging)
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=blur_prob),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_ssl_local_transform(image_size: int = 512, config: dict | None = None) -> A.Compose:
    """
    Local crop transform for SSL student network.

    Produces small crops (covering 5-40% of the image) that force the model
    to learn fine-grained features like opacity edges and rib patterns.
    Local crops are resized to 98x98 (must be multiple of 14 for DINOv2 patch_size).
    """
    cfg = config or {}
    scale = cfg.get("local_crop_scale", [0.05, 0.4])
    local_size = 98  # Must be multiple of 14 (DINOv2 patch_size). 98 = 14 × 7

    return A.Compose([
        A.RandomResizedCrop(
            size=(local_size, local_size),
            scale=scale,
            ratio=(0.75, 1.333),
            interpolation=cv2.INTER_LANCZOS4,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.8,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_detection_train_transform(image_size: int = 512) -> A.Compose:
    """
    Augmentation pipeline for RF-DETR detection training.

    RF-DETR handles its own internal augmentations, but we can apply
    medical-specific preprocessing before feeding images.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category_ids"],
        min_visibility=0.3,
    ))
