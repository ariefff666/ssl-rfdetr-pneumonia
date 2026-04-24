"""
visualize.py — Comprehensive visualization for RF-DETR Pneumonia Detection thesis.

Generates all visualizations needed for thesis presentation:
  1. Dataset samples (train/valid/test with GT bounding boxes)
  2. Detection results (predicted bounding boxes from trained models)
  3. SSL vs Baseline comparison charts
  4. Training curves comparison

Usage:
    python3 src/visualize.py \
        --dataset-dir /kaggle/working/dataset_coco \
        --ssl-dir /kaggle/working/checkpoints/rfdetr/rfdetr-finetune-ssl \
        --baseline-dir /kaggle/working/checkpoints/rfdetr/rfdetr-finetune-baseline \
        --output-dir /kaggle/working/visualizations \
        --fraction 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


# ============================================================================
# Color Palette
# ============================================================================
SSL_COLOR = "#2196F3"       # Blue
BASELINE_COLOR = "#FF9800"  # Orange
GT_COLOR = "#4CAF50"        # Green
PRED_COLOR = "#F44336"      # Red
BG_COLOR = "#1a1a2e"        # Dark background
TEXT_COLOR = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#555",
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#555",
    "grid.color": "#333",
    "font.family": "sans-serif",
    "font.size": 11,
})


# ============================================================================
# 1. Dataset Sample Visualization
# ============================================================================
def visualize_dataset_samples(dataset_dir: str, output_dir: str) -> None:
    """Show sample images from train/valid/test with GT bounding boxes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        ann_path = Path(dataset_dir) / split / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"  [Skip] {split} annotations not found")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        # Build image_id -> annotations map
        img_anns = {}
        for ann in coco["annotations"]:
            img_anns.setdefault(ann["image_id"], []).append(ann)

        # Select samples: 3 with boxes + 1 without
        imgs_with_box = [img for img in coco["images"] if img["id"] in img_anns]
        imgs_no_box = [img for img in coco["images"] if img["id"] not in img_anns]

        samples = []
        if imgs_with_box:
            samples += imgs_with_box[:3]
        if imgs_no_box:
            samples += imgs_no_box[:1]

        if not samples:
            continue

        n = len(samples)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Dataset: {split.upper()} split ({len(coco['images'])} images, "
                     f"{len(coco['annotations'])} annotations)",
                     fontsize=14, fontweight="bold", y=1.02)

        for ax, img_info in zip(axes, samples):
            img_path = Path(dataset_dir) / split / img_info["file_name"]
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img, cmap="gray")
            else:
                ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")

            # Draw GT bounding boxes
            anns = img_anns.get(img_info["id"], [])
            for ann in anns:
                x, y, w, h = ann["bbox"]
                rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                         edgecolor=GT_COLOR, facecolor="none")
                ax.add_patch(rect)
                ax.text(x, y - 5, "Pneumonia", fontsize=8, color=GT_COLOR,
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.2",
                        facecolor="black", alpha=0.7))

            label = "POSITIVE" if anns else "NORMAL"
            color = GT_COLOR if anns else "#999"
            ax.set_title(f"{label} ({len(anns)} box{'es' if len(anns) != 1 else ''})",
                         fontsize=10, color=color)
            ax.axis("off")

        plt.tight_layout()
        save_path = out / f"dataset_samples_{split}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Viz] Saved: {save_path}")


# ============================================================================
# 2. Detection Result Visualization (Predicted vs GT)
# ============================================================================
def visualize_detections(dataset_dir: str, model_dir: str, model_name: str,
                         output_dir: str) -> None:
    """Run inference and visualize predicted bounding boxes vs ground truth."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find best checkpoint
    ckpt_ema = Path(model_dir) / "checkpoint_best_ema.pth"
    ckpt_reg = Path(model_dir) / "checkpoint_best_regular.pth"
    ckpt = ckpt_ema if ckpt_ema.exists() else ckpt_reg

    if not ckpt.exists():
        print(f"  [Skip] No checkpoint found in {model_dir}")
        return

    try:
        import torch
        from rfdetr import RFDETRSmall

        model = RFDETRSmall()
        model.load(str(ckpt))
        print(f"  [Viz] Loaded model from: {ckpt}")
    except Exception as e:
        print(f"  [Skip] Could not load model: {e}")
        return

    # Load validation GT
    val_ann_path = Path(dataset_dir) / "valid" / "_annotations.coco.json"
    if not val_ann_path.exists():
        print("  [Skip] Validation annotations not found")
        return

    with open(val_ann_path) as f:
        coco = json.load(f)

    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    # Pick images with GT boxes for visualization
    imgs_with_box = [img for img in coco["images"] if img["id"] in img_anns]
    samples = imgs_with_box[:6]

    if not samples:
        return

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle(f"Detection Results: {model_name}", fontsize=16, fontweight="bold", y=1.02)

    for idx, img_info in enumerate(samples):
        ax = axes[idx // n_cols][idx % n_cols]
        img_path = Path(dataset_dir) / "valid" / img_info["file_name"]

        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        ax.imshow(img, cmap="gray")

        # Draw GT boxes (green)
        for ann in img_anns.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=GT_COLOR, facecolor="none",
                                     linestyle="--")
            ax.add_patch(rect)

        # Run inference
        try:
            detections = model.predict(img, threshold=0.3)
            if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=2, edgecolor=PRED_COLOR,
                                             facecolor="none")
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f"{conf:.2f}", fontsize=8,
                            color=PRED_COLOR, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2",
                            facecolor="black", alpha=0.7))
        except Exception as e:
            ax.text(0.5, 0.05, f"Inference error", ha="center",
                    transform=ax.transAxes, fontsize=8, color="red")

        ax.axis("off")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GT_COLOR, linewidth=2, linestyle="--", label="Ground Truth"),
        Line2D([0], [0], color=PRED_COLOR, linewidth=2, label="Prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=12)

    plt.tight_layout()
    save_path = out / f"detections_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved: {save_path}")


# ============================================================================
# 3. Metrics Comparison Bar Chart (SSL vs Baseline)
# ============================================================================
def visualize_comparison(ssl_dir: str, baseline_dir: str, fraction: int,
                         output_dir: str) -> None:
    """Create professional comparison chart between SSL and Baseline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ssl_metrics = _extract_best_metrics(ssl_dir)
    baseline_metrics = _extract_best_metrics(baseline_dir)

    if not ssl_metrics or not baseline_metrics:
        print("  [Skip] Could not extract metrics for comparison")
        return

    metric_names = ["mAP@50:95", "mAP@50", "F1", "Precision", "Recall"]
    metric_keys = ["mAP_50_95", "mAP_50", "F1", "precision", "recall"]

    ssl_vals = [ssl_metrics.get(k, 0) for k in metric_keys]
    base_vals = [baseline_metrics.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width/2, ssl_vals, width, label="SSL (Domain-Adapted)",
                   color=SSL_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, base_vals, width, label="Baseline (Original DINOv2)",
                   color=BASELINE_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"RF-DETR Detection: SSL vs Baseline — {fraction}% Data",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(0, min(1.0, max(max(ssl_vals), max(base_vals)) * 1.3 + 0.05))
    ax.grid(axis="y", alpha=0.2)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Improvement annotations
    for i, (s, b) in enumerate(zip(ssl_vals, base_vals)):
        if b > 0:
            diff_pct = ((s - b) / b) * 100
            color = "#4CAF50" if diff_pct > 0 else "#F44336"
            sign = "+" if diff_pct > 0 else ""
            ax.text(i, max(s, b) + 0.02, f"{sign}{diff_pct:.1f}%",
                    ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    save_path = out / f"comparison_frac{fraction}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved: {save_path}")

    # Also save metrics as JSON
    results = {
        "fraction": fraction,
        "ssl": ssl_metrics,
        "baseline": baseline_metrics,
    }
    json_path = out / f"metrics_frac{fraction}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [Viz] Saved: {json_path}")


# ============================================================================
# 4. Training Curves Comparison
# ============================================================================
def visualize_training_curves(ssl_dir: str, baseline_dir: str, fraction: int,
                              output_dir: str) -> None:
    """Compare training curves (loss and mAP) between SSL and Baseline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ssl_csv = _find_metrics_csv(ssl_dir)
    base_csv = _find_metrics_csv(baseline_dir)

    if not ssl_csv or not base_csv:
        print("  [Skip] Training metrics CSV not found for curve comparison")
        return

    try:
        import csv

        def read_csv_metrics(csv_path):
            metrics = {"epoch": [], "val_loss": [], "val_mAP": [], "train_loss": []}
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if "val/loss" in row and row["val/loss"]:
                            metrics["epoch"].append(int(row.get("epoch", len(metrics["epoch"]))))
                            metrics["val_loss"].append(float(row["val/loss"]))
                            metrics["val_mAP"].append(float(row.get("val/mAP_50_95", 0)))
                        if "train/loss" in row and row["train/loss"]:
                            metrics["train_loss"].append(float(row["train/loss"]))
                    except (ValueError, KeyError):
                        continue
            return metrics

        ssl_m = read_csv_metrics(ssl_csv)
        base_m = read_csv_metrics(base_csv)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Training Curves: SSL vs Baseline — {fraction}% Data",
                     fontsize=16, fontweight="bold")

        # Val Loss
        if ssl_m["val_loss"] and base_m["val_loss"]:
            axes[0].plot(ssl_m["val_loss"], color=SSL_COLOR, linewidth=2, label="SSL")
            axes[0].plot(base_m["val_loss"], color=BASELINE_COLOR, linewidth=2, label="Baseline")
            axes[0].set_title("Validation Loss", fontsize=13)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(alpha=0.2)

        # Val mAP
        if ssl_m["val_mAP"] and base_m["val_mAP"]:
            axes[1].plot(ssl_m["val_mAP"], color=SSL_COLOR, linewidth=2, label="SSL")
            axes[1].plot(base_m["val_mAP"], color=BASELINE_COLOR, linewidth=2, label="Baseline")
            axes[1].set_title("Validation mAP@50:95", fontsize=13)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("mAP")
            axes[1].legend()
            axes[1].grid(alpha=0.2)

        # Train Loss
        if ssl_m["train_loss"] and base_m["train_loss"]:
            axes[2].plot(ssl_m["train_loss"], color=SSL_COLOR, linewidth=2, label="SSL")
            axes[2].plot(base_m["train_loss"], color=BASELINE_COLOR, linewidth=2, label="Baseline")
            axes[2].set_title("Training Loss", fontsize=13)
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Loss")
            axes[2].legend()
            axes[2].grid(alpha=0.2)

        plt.tight_layout()
        save_path = out / f"training_curves_frac{fraction}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [Viz] Saved: {save_path}")

    except Exception as e:
        print(f"  [Skip] Error reading CSV: {e}")


# ============================================================================
# Helper Functions
# ============================================================================
def _extract_best_metrics(run_dir: str) -> dict | None:
    """Extract best validation metrics from Lightning CSV logs."""
    run_path = Path(run_dir)

    # Try Lightning CSV metrics file
    csv_path = _find_metrics_csv(run_dir)
    if csv_path:
        try:
            import csv
            best = {"mAP_50_95": 0}
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        mAP = float(row.get("val/mAP_50_95", 0) or 0)
                        if mAP > best["mAP_50_95"]:
                            best = {
                                "mAP_50_95": mAP,
                                "mAP_50": float(row.get("val/mAP_50", 0) or 0),
                                "F1": float(row.get("val/F1", 0) or 0),
                                "precision": float(row.get("val/precision", 0) or 0),
                                "recall": float(row.get("val/recall", 0) or 0),
                            }
                    except (ValueError, KeyError):
                        continue
            if best["mAP_50_95"] > 0:
                return best
        except Exception:
            pass

    # Try eval_results.json as fallback
    for fname in ["eval_results.json", "metrics.json"]:
        p = run_path / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)

    return None


def _find_metrics_csv(run_dir: str) -> str | None:
    """Find the Lightning CSV log file in a run directory."""
    run_path = Path(run_dir)

    # Search for metrics.csv in lightning_logs or directly
    for pattern in ["**/metrics.csv", "lightning_logs/**/metrics.csv",
                    "csv_logs/**/metrics.csv"]:
        found = list(run_path.glob(pattern))
        if found:
            return str(found[0])

    return None


# ============================================================================
# 5. Summary Table (printed + saved)
# ============================================================================
def print_summary_table(ssl_dir: str, baseline_dir: str, fraction: int,
                        output_dir: str) -> None:
    """Print and save a formatted comparison summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ssl_m = _extract_best_metrics(ssl_dir)
    base_m = _extract_best_metrics(baseline_dir)

    header = f"\n{'='*70}\n  RF-DETR Results — Data Fraction: {fraction}%\n{'='*70}\n"
    header += f"  {'Metric':<20} {'SSL':>12} {'Baseline':>12} {'Δ':>10}\n"
    header += f"  {'-'*54}\n"

    lines = [header]

    if ssl_m and base_m:
        for name, key in [("mAP@50:95", "mAP_50_95"), ("mAP@50", "mAP_50"),
                          ("F1-Score", "F1"), ("Precision", "precision"),
                          ("Recall", "recall")]:
            s = ssl_m.get(key, 0)
            b = base_m.get(key, 0)
            diff = s - b
            sign = "+" if diff >= 0 else ""
            lines.append(f"  {name:<20} {s:>12.4f} {b:>12.4f} {sign}{diff:>9.4f}\n")
    else:
        lines.append("  [Metrics not yet available — training may still be running]\n")

    lines.append(f"{'='*70}\n")
    summary = "".join(lines)
    print(summary)

    save_path = out / f"summary_frac{fraction}.txt"
    with open(save_path, "w") as f:
        f.write(summary)
    print(f"  [Viz] Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Thesis Visualization Suite")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--ssl-dir", type=str, required=True)
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/visualizations")
    parser.add_argument("--fraction", type=int, required=True, help="Data fraction percentage (10, 25, 50, 100)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Thesis Visualization Suite — Fraction {args.fraction}%")
    print("=" * 70)

    # 1. Dataset samples
    print("\n[1/5] Dataset Sample Visualization...")
    visualize_dataset_samples(args.dataset_dir, args.output_dir)

    # 2. Detection results (SSL)
    print("\n[2/5] Detection Visualization (SSL)...")
    visualize_detections(args.dataset_dir, args.ssl_dir,
                         "SSL Domain-Adapted", args.output_dir)

    # 3. Detection results (Baseline)
    print("\n[3/5] Detection Visualization (Baseline)...")
    visualize_detections(args.dataset_dir, args.baseline_dir,
                         "Baseline Original", args.output_dir)

    # 4. Comparison chart
    print("\n[4/5] Comparison Chart...")
    visualize_comparison(args.ssl_dir, args.baseline_dir,
                         args.fraction, args.output_dir)

    # 5. Training curves
    print("\n[5/5] Training Curves...")
    visualize_training_curves(args.ssl_dir, args.baseline_dir,
                              args.fraction, args.output_dir)

    # Summary
    print_summary_table(args.ssl_dir, args.baseline_dir,
                        args.fraction, args.output_dir)

    print(f"\n✅ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
