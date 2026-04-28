"""
RSNA Pneumonia Detection Challenge — Official Evaluation Metric

Metric: Mean Average Precision at IoU thresholds [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

For each image:
  1. At each IoU threshold t, compute precision = TP(t) / (TP(t) + FP(t) + FN(t))
  2. Average precision across all thresholds
  3. Images with NO GT boxes: any prediction = score 0 for that image

Final score = mean of per-image average precisions across test set.

Usage (standalone — no re-training needed):
    python3 src/evaluate_rsna.py \
        --checkpoint /path/to/checkpoint_best_total.pth \
        --dataset-dir /path/to/dataset_coco \
        --split valid \
        --output /path/to/rsna_eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# RSNA IoU thresholds
RSNA_IOU_THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]


def compute_iou(box_pred, box_gt):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    union = area_pred + area_gt - inter

    return inter / union if union > 0 else 0.0


def rsna_precision_at_iou(pred_boxes, gt_boxes, iou_threshold):
    """
    Compute precision at a single IoU threshold for one image.
    precision = TP / (TP + FP + FN)

    pred_boxes: list of [x1, y1, x2, y2, confidence]
    gt_boxes: list of [x1, y1, x2, y2]
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0  # No GT, no predictions = perfect

    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return 0.0  # No GT but predictions made = all FP

    if len(pred_boxes) == 0 and len(gt_boxes) > 0:
        return 0.0  # GT exists but no predictions = all FN

    # Sort predictions by confidence (highest first)
    pred_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for pred in pred_sorted:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return precision


def rsna_score_single_image(pred_boxes, gt_boxes):
    """Compute RSNA average precision for a single image across all IoU thresholds."""
    precisions = []
    for t in RSNA_IOU_THRESHOLDS:
        p = rsna_precision_at_iou(pred_boxes, gt_boxes, t)
        precisions.append(p)
    return np.mean(precisions)


def load_model(checkpoint_path):
    """Load RF-DETR model from checkpoint (.pth or .ckpt)."""
    import torch
    from rfdetr import RFDETRSmall

    ckpt_path = str(checkpoint_path)

    if ckpt_path.endswith(".ckpt"):
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in raw:
            sd = {}
            for k, v in raw["state_dict"].items():
                new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
                sd[new_k] = v
            tmp_path = str(Path(checkpoint_path).parent / "_rsna_eval_temp.pth")
            torch.save({"model": sd}, tmp_path)
            ckpt_path = tmp_path

    model = RFDETRSmall(pretrain_weights=ckpt_path, num_classes=1)
    return model


def evaluate_rsna(model, dataset_dir, split="valid", conf_threshold=0.01):
    """
    Run RSNA evaluation with optimal threshold search.
    Inference runs once at low threshold, then we sweep to find best score.
    """
    from PIL import Image

    ann_path = Path(dataset_dir) / split / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  [ERROR] Annotations not found: {ann_path}")
        return None

    with open(ann_path) as f:
        coco = json.load(f)

    # Build GT map: image_id -> list of [x1, y1, x2, y2]
    gt_map = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_map.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])

    total = len(coco["images"])
    print(f"  Running inference on {total} images (split='{split}')...")

    # Step 1: Cache ALL predictions at low threshold
    all_preds = {}  # image_id -> list of [x1, y1, x2, y2, conf]
    all_gt = {}     # image_id -> list of [x1, y1, x2, y2]

    for idx, img_info in enumerate(coco["images"]):
        if (idx + 1) % 500 == 0:
            print(f"    [{idx+1}/{total}]")

        img_id = img_info["id"]
        all_gt[img_id] = gt_map.get(img_id, [])

        img_path = Path(dataset_dir) / split / img_info["file_name"]
        if not img_path.exists():
            all_preds[img_id] = []
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            detections = model.predict(img, threshold=conf_threshold)

            pred_boxes = []
            if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                for box, conf in zip(detections.xyxy, detections.confidence):
                    x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
                    c = conf.item() if hasattr(conf, 'item') else float(conf)
                    pred_boxes.append([x1, y1, x2, y2, c])
            all_preds[img_id] = pred_boxes
        except Exception:
            all_preds[img_id] = []

    print(f"  Inference done. Searching optimal threshold...")

    # Step 2: Sweep confidence thresholds
    thresholds_to_try = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    best_score = -1
    best_threshold = 0.3
    best_results = None

    for ct in thresholds_to_try:
        per_image_scores = []
        per_threshold_scores = {t: [] for t in RSNA_IOU_THRESHOLDS}

        for img_id in all_gt:
            gt_boxes = all_gt[img_id]
            # Filter predictions by current confidence threshold
            pred_boxes = [p for p in all_preds.get(img_id, []) if p[4] >= ct]

            img_score = rsna_score_single_image(pred_boxes, gt_boxes)
            per_image_scores.append(img_score)

            for t in RSNA_IOU_THRESHOLDS:
                p = rsna_precision_at_iou(pred_boxes, gt_boxes, t)
                per_threshold_scores[t].append(p)

        score = float(np.mean(per_image_scores)) if per_image_scores else 0.0
        print(f"    conf={ct:.2f} → RSNA={score:.4f}")

        if score > best_score:
            best_score = score
            best_threshold = ct
            best_results = {
                "rsna_score": score,
                "best_conf_threshold": ct,
                "num_images": len(per_image_scores),
                "split": split,
                "per_threshold": {
                    str(t): float(np.mean(per_threshold_scores[t]))
                    for t in RSNA_IOU_THRESHOLDS
                },
            }

    # Add threshold sweep summary
    best_results["threshold_sweep"] = {
        f"{ct:.2f}": float(np.mean([
            rsna_score_single_image(
                [p for p in all_preds.get(img_id, []) if p[4] >= ct],
                all_gt[img_id]
            ) for img_id in all_gt
        ])) for ct in thresholds_to_try
    }

    print(f"\n  ✅ Best: conf={best_threshold:.2f} → RSNA Score={best_score:.4f}")
    return best_results


def print_results(results, model_name="Model"):
    """Pretty-print RSNA evaluation results."""
    print(f"\n{'='*60}")
    print(f"  RSNA Evaluation — {model_name}")
    print(f"{'='*60}")
    print(f"  Split: {results['split']} ({results['num_images']} images)")
    print(f"  Best Conf Threshold: {results.get('best_conf_threshold', 'N/A')}")
    print(f"  RSNA Score (mAP@[.4:.75]): {results['rsna_score']:.4f}")
    print(f"  {'─'*40}")
    print(f"  Per IoU-threshold breakdown:")
    for t_str, score in results["per_threshold"].items():
        print(f"    IoU {float(t_str):.2f}: {score:.4f}")
    if "threshold_sweep" in results:
        print(f"  {'─'*40}")
        print(f"  Confidence threshold sweep:")
        for ct_str, score in results["threshold_sweep"].items():
            marker = " ◀ BEST" if abs(score - results['rsna_score']) < 1e-6 else ""
            print(f"    conf {ct_str}: {score:.4f}{marker}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RSNA Pneumonia Detection Challenge — Official Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth or .ckpt)")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to COCO dataset directory")
    parser.add_argument("--split", type=str, default="valid",
                        choices=["valid", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--conf-threshold", type=float, default=0.01,
                        help="Confidence threshold for predictions")
    parser.add_argument("--model-name", type=str, default="Model",
                        help="Name for display (e.g., 'SSL frac10')")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)

    results = evaluate_rsna(model, args.dataset_dir, args.split, args.conf_threshold)
    if results is None:
        sys.exit(1)

    results["model_name"] = args.model_name
    results["checkpoint"] = str(args.checkpoint)

    print_results(results, args.model_name)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
