"""
prepare_coco.py — Convert RSNA Pneumonia Detection CSV to COCO JSON format.

This script reads the RSNA CSV files and produces a COCO-formatted dataset
directory structure that RF-DETR expects:

    dataset_coco/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── {patientId}.png  (symlinks or copies)
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── {patientId}.png
    └── test/
        ├── _annotations.coco.json
        └── {patientId}.png

Usage:
    python -m src.data.prepare_coco --config configs/finetune_rfdetr.yaml
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Ensure project root is in Python path (works regardless of invocation method)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_coco_annotation(
    image_records: list[dict],
    bbox_df: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    split_name: str,
) -> None:
    """
    Build a COCO-format annotation JSON and copy images for one data split.

    Args:
        image_records: List of dicts with keys 'patientId' and 'class'.
        bbox_df: DataFrame with bounding box info (patientId, x, y, width, height, Target).
        image_dir: Path to the source directory containing PNG images.
        output_dir: Root output directory (e.g., dataset_coco/).
        split_name: One of 'train', 'valid', 'test'.
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # COCO structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Lung Opacity", "supercategory": "pathology"}
        ],
    }

    patient_ids = [rec["patientId"] for rec in image_records]
    # Filter bbox_df to only patients in this split
    split_bboxes = bbox_df[bbox_df["patientId"].isin(patient_ids)]

    annotation_id = 1

    for img_idx, patient_id in enumerate(tqdm(patient_ids, desc=f"Building {split_name}")):
        # Locate source image — RSNA uses .dcm extension in folder name but they are
        # actually stored as .dcm files. On Kaggle the images are DICOM but we read PNG
        # exports. We try both .png and .dcm extensions.
        src_png = image_dir / f"{patient_id}.png"
        src_dcm = image_dir / f"{patient_id}.dcm"

        if src_png.exists():
            src_file = src_png
            filename = f"{patient_id}.png"
        elif src_dcm.exists():
            src_file = src_dcm
            filename = f"{patient_id}.dcm"
        else:
            # Skip if image not found (could be in test set only)
            continue

        # Copy image to split directory
        dst_file = split_dir / filename
        if not dst_file.exists():
            # Use symlink on Linux (Kaggle) for speed, copy on Windows
            try:
                os.symlink(src_file, dst_file)
            except (OSError, NotImplementedError):
                shutil.copy2(src_file, dst_file)

        # Get image dimensions — RSNA images are 1024x1024
        img_width, img_height = 1024, 1024

        coco["images"].append({
            "id": img_idx + 1,
            "file_name": filename,
            "width": img_width,
            "height": img_height,
        })

        # Add bounding box annotations for positive cases
        patient_boxes = split_bboxes[
            (split_bboxes["patientId"] == patient_id) & (split_bboxes["Target"] == 1)
        ]

        for _, row in patient_boxes.iterrows():
            x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_idx + 1,
                "category_id": 1,
                "bbox": [x, y, w, h],  # COCO format: [x_min, y_min, width, height]
                "area": w * h,
                "iscrowd": 0,
            })
            annotation_id += 1

    # Write annotation file
    ann_path = split_dir / "_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"  [{split_name}] {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations → {ann_path}"
    )


def main(config_path: str) -> None:
    """Main entry point for COCO dataset preparation."""
    cfg = load_config(config_path)

    train_labels_csv = Path(cfg["data"]["train_labels_csv"])
    class_info_csv = Path(cfg["data"]["class_info_csv"])
    image_dir = Path(cfg["data"]["image_dir"])
    output_dir = Path(cfg["data"]["dataset_dir"])

    print(f"Reading CSV files...")
    labels_df = pd.read_csv(train_labels_csv)
    class_df = pd.read_csv(class_info_csv)

    # Get unique patient IDs with their class info
    # Keep only one row per patient from class_info (some patients have duplicates)
    patient_classes = class_df.drop_duplicates(subset="patientId")

    # Get positive patients (have at least one bounding box)
    positive_patients = set(labels_df[labels_df["Target"] == 1]["patientId"].unique())
    all_patients = labels_df["patientId"].unique()

    print(f"Total unique patients: {len(all_patients)}")
    print(f"Patients with pneumonia (positive): {len(positive_patients)}")
    print(f"Patients without pneumonia (negative): {len(all_patients) - len(positive_patients)}")

    # Create stratified split: 80% train, 10% valid, 10% test
    # Stratify by whether patient has pneumonia or not
    patient_labels = [1 if pid in positive_patients else 0 for pid in all_patients]

    train_ids, temp_ids, train_labels_split, temp_labels = train_test_split(
        all_patients,
        patient_labels,
        test_size=0.2,
        random_state=42,
        stratify=patient_labels,
    )

    valid_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )

    print(f"\nSplit: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}")

    # Build records
    train_records = [{"patientId": pid} for pid in train_ids]
    valid_records = [{"patientId": pid} for pid in valid_ids]
    test_records = [{"patientId": pid} for pid in test_ids]

    # Build COCO annotations for each split
    print("\nBuilding COCO dataset...")
    for split_name, records in [
        ("train", train_records),
        ("valid", valid_records),
        ("test", test_records),
    ]:
        build_coco_annotation(records, labels_df, image_dir, output_dir, split_name)

    print(f"\nDone! COCO dataset saved to: {output_dir}")
    print("You can now use this directory with RF-DETR's model.train(dataset_dir=...)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RSNA CSV to COCO JSON for RF-DETR")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_rfdetr.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    main(args.config)
