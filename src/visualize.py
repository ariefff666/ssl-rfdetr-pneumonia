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

    # Find best checkpoint (try ema first, then regular, then last.ckpt)
    ckpt_ema = Path(model_dir) / "checkpoint_best_ema.pth"
    ckpt_reg = Path(model_dir) / "checkpoint_best_regular.pth"
    ckpt_total = Path(model_dir) / "checkpoint_best_total.pth"
    ckpt_last = Path(model_dir) / "last.ckpt"

    ckpt = None
    for c in [ckpt_ema, ckpt_reg, ckpt_total, ckpt_last]:
        if c.exists():
            ckpt = c
            break

    if ckpt is None:
        print(f"  [Skip] No checkpoint found in {model_dir}")
        return

    try:
        import torch
        from rfdetr import RFDETRSmall

        load_path = str(ckpt)

        # Lightning .ckpt files need conversion to .pth
        if str(ckpt).endswith(".ckpt"):
            raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            if "state_dict" in raw:
                # Strip Lightning 'model.' prefix from keys
                sd = {}
                for k, v in raw["state_dict"].items():
                    new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
                    sd[new_k] = v
                tmp_pth = Path(model_dir) / "_temp_weights.pth"
                # RF-DETR expects checkpoint['model'] format
                torch.save({"model": sd}, tmp_pth)
                load_path = str(tmp_pth)
                print(f"  [Viz] Converted .ckpt → .pth from: {ckpt.name}")

        model = RFDETRSmall(pretrain_weights=load_path, num_classes=1)
        print(f"  [Viz] Loaded model from: {ckpt.name}")
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
# 2b. Side-by-Side Detection Comparison (SSL vs Baseline on same images)
# ============================================================================
def visualize_side_by_side(dataset_dir: str, ssl_dir: str, baseline_dir: str,
                           output_dir: str) -> None:
    """Show SSL vs Baseline predictions on the same images side-by-side."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from rfdetr import RFDETRSmall
    except ImportError:
        print("  [Skip] rfdetr not installed")
        return

    def _load_model(model_dir):
        for name in ["checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
                      "checkpoint_best_total.pth", "last.ckpt"]:
            p = Path(model_dir) / name
            if not p.exists():
                continue
            load_path = str(p)
            if name.endswith(".ckpt"):
                raw = torch.load(str(p), map_location="cpu", weights_only=False)
                if "state_dict" in raw:
                    sd = {}
                    for k, v in raw["state_dict"].items():
                        new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
                        sd[new_k] = v
                    tmp = Path(model_dir) / "_temp_sbs.pth"
                    torch.save({"model": sd}, tmp)
                    load_path = str(tmp)
            return RFDETRSmall(pretrain_weights=load_path, num_classes=1)
        return None

    ssl_model = _load_model(ssl_dir)
    base_model = _load_model(baseline_dir)
    if not ssl_model or not base_model:
        print("  [Skip] Could not load both models for side-by-side")
        return

    # Load val annotations
    ann_path = Path(dataset_dir) / "valid" / "_annotations.coco.json"
    if not ann_path.exists():
        print("  [Skip] Validation annotations not found")
        return

    with open(ann_path) as f:
        coco = json.load(f)

    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    samples = [img for img in coco["images"] if img["id"] in img_anns][:4]
    if not samples:
        return

    fig, axes = plt.subplots(len(samples), 3, figsize=(18, 5 * len(samples)))
    fig.suptitle("Detection Comparison: GT vs SSL vs Baseline", fontsize=18,
                 fontweight="bold", y=1.01)

    col_titles = ["Ground Truth", "SSL (Domain-Adapted)", "Baseline (Original)"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=14, fontweight="bold",
                               color=[GT_COLOR, SSL_COLOR, BASELINE_COLOR][col])

    for row, img_info in enumerate(samples):
        img_path = Path(dataset_dir) / "valid" / img_info["file_name"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")

        for col in range(3):
            axes[row][col].imshow(img, cmap="gray")
            axes[row][col].axis("off")

        # GT boxes
        for ann in img_anns.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=GT_COLOR, facecolor="none", linestyle="--")
            axes[row][0].add_patch(rect)

        # SSL predictions
        try:
            det = ssl_model.predict(img, threshold=0.3)
            if hasattr(det, 'xyxy') and len(det.xyxy) > 0:
                for box, conf in zip(det.xyxy, det.confidence):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                             linewidth=2, edgecolor=SSL_COLOR, facecolor="none")
                    axes[row][1].add_patch(rect)
                    axes[row][1].text(x1, y1-5, f"{conf:.2f}", fontsize=8, color=SSL_COLOR,
                                     fontweight="bold", bbox=dict(facecolor="black", alpha=0.7))
        except Exception:
            pass

        # Baseline predictions
        try:
            det = base_model.predict(img, threshold=0.3)
            if hasattr(det, 'xyxy') and len(det.xyxy) > 0:
                for box, conf in zip(det.xyxy, det.confidence):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                             linewidth=2, edgecolor=BASELINE_COLOR, facecolor="none")
                    axes[row][2].add_patch(rect)
                    axes[row][2].text(x1, y1-5, f"{conf:.2f}", fontsize=8, color=BASELINE_COLOR,
                                     fontweight="bold", bbox=dict(facecolor="black", alpha=0.7))
        except Exception:
            pass

    plt.tight_layout()
    save_path = out / "side_by_side_comparison.png"
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
# 6. Data Distribution Chart
# ============================================================================
def visualize_data_distribution(dataset_dir: str, fraction: int,
                                output_dir: str) -> None:
    """Visualize positive/negative sample distribution in train/val/test."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    pos_counts = []
    neg_counts = []
    total_counts = []

    for split in splits:
        ann_path = Path(dataset_dir) / split / "_annotations.coco.json"
        if not ann_path.exists():
            pos_counts.append(0); neg_counts.append(0); total_counts.append(0)
            continue
        with open(ann_path) as f:
            coco = json.load(f)
        img_ids_with_ann = set(a["image_id"] for a in coco["annotations"])
        total = len(coco["images"])
        pos = len(img_ids_with_ann)
        neg = total - pos
        pos_counts.append(pos)
        neg_counts.append(neg)
        total_counts.append(total)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Dataset Distribution — {fraction}% Fraction",
                 fontsize=16, fontweight="bold")

    # Stacked bar
    x = np.arange(len(splits))
    axes[0].bar(x, pos_counts, color="#E53935", alpha=0.9, label="Pneumonia (+)")
    axes[0].bar(x, neg_counts, bottom=pos_counts, color="#43A047", alpha=0.9, label="Normal (−)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.capitalize() for s in splits], fontsize=12)
    axes[0].set_ylabel("Number of Images", fontsize=12)
    axes[0].set_title("Sample Counts", fontsize=13)
    axes[0].legend(fontsize=11)
    for i, (p, n) in enumerate(zip(pos_counts, neg_counts)):
        axes[0].text(i, p/2, str(p), ha="center", va="center", fontweight="bold", color="white")
        axes[0].text(i, p + n/2, str(n), ha="center", va="center", fontweight="bold", color="white")
    axes[0].grid(axis="y", alpha=0.2)

    # Pie chart for train split
    if pos_counts[0] + neg_counts[0] > 0:
        sizes = [pos_counts[0], neg_counts[0]]
        labels = [f"Pneumonia\n({pos_counts[0]})", f"Normal\n({neg_counts[0]})"]
        colors = ["#E53935", "#43A047"]
        axes[1].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize": 11})
        axes[1].set_title("Train Split Ratio", fontsize=13)

    plt.tight_layout()
    save_path = out / f"data_distribution_frac{fraction}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved: {save_path}")


# ============================================================================
# 7. Radar Chart (Spider Plot)
# ============================================================================
def visualize_radar_chart(ssl_dir: str, baseline_dir: str, fraction: int,
                          output_dir: str) -> None:
    """Create a radar chart comparing SSL vs Baseline across all metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ssl_m = _extract_best_metrics(ssl_dir)
    base_m = _extract_best_metrics(baseline_dir)

    if not ssl_m or not base_m:
        print("  [Skip] Metrics not available for radar chart")
        return

    categories = ["mAP@50:95", "mAP@50", "F1", "Precision", "Recall"]
    keys = ["mAP_50_95", "mAP_50", "F1", "precision", "recall"]

    ssl_vals = [ssl_m.get(k, 0) for k in keys]
    base_vals = [base_m.get(k, 0) for k in keys]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    ssl_vals_plot = ssl_vals + [ssl_vals[0]]
    base_vals_plot = base_vals + [base_vals[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, ssl_vals_plot, "o-", linewidth=2, color=SSL_COLOR, label="SSL")
    ax.fill(angles, ssl_vals_plot, alpha=0.15, color=SSL_COLOR)
    ax.plot(angles, base_vals_plot, "o-", linewidth=2, color=BASELINE_COLOR, label="Baseline")
    ax.fill(angles, base_vals_plot, alpha=0.15, color=BASELINE_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title(f"Performance Radar — {fraction}% Data", fontsize=15,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out / f"radar_frac{fraction}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved: {save_path}")


# ============================================================================
# 8. Improvement Delta Chart
# ============================================================================
def visualize_improvement(ssl_dir: str, baseline_dir: str, fraction: int,
                          output_dir: str) -> None:
    """Horizontal bar chart showing SSL improvement over Baseline per metric."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ssl_m = _extract_best_metrics(ssl_dir)
    base_m = _extract_best_metrics(baseline_dir)

    if not ssl_m or not base_m:
        print("  [Skip] Metrics not available for improvement chart")
        return

    names = ["mAP@50:95", "mAP@50", "F1", "Precision", "Recall"]
    keys = ["mAP_50_95", "mAP_50", "F1", "precision", "recall"]

    deltas = []
    pcts = []
    for k in keys:
        s = ssl_m.get(k, 0)
        b = base_m.get(k, 0)
        deltas.append(s - b)
        pcts.append(((s - b) / b * 100) if b > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
    bars = ax.barh(names, pcts, color=colors, alpha=0.85, edgecolor="white")

    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Improvement (%)", fontsize=12)
    ax.set_title(f"SSL Improvement over Baseline — {fraction}% Data",
                 fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    for bar, pct, delta in zip(bars, pcts, deltas):
        sign = "+" if pct >= 0 else ""
        ax.text(bar.get_width() + (0.3 if pct >= 0 else -0.3),
                bar.get_y() + bar.get_height()/2,
                f"{sign}{pct:.1f}% ({sign}{delta:.4f})",
                va="center", fontsize=10, fontweight="bold",
                ha="left" if pct >= 0 else "right")

    plt.tight_layout()
    save_path = out / f"improvement_frac{fraction}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Viz] Saved: {save_path}")


# ============================================================================
# 9. Cross-Fraction Tracker (aggregates results from all fractions)
# ============================================================================
def visualize_cross_fraction(output_dir: str) -> None:
    """If multiple fraction results exist, create a cross-fraction comparison."""
    base = Path(output_dir).parent  # e.g. /kaggle/working/visualizations
    fractions = []
    ssl_maps = []
    base_maps = []

    for frac_dir in sorted(base.glob("frac*")):
        json_path = list(frac_dir.glob("metrics_frac*.json"))
        if not json_path:
            continue
        with open(json_path[0]) as f:
            data = json.load(f)
        frac = data.get("fraction", 0)
        ssl = data.get("ssl", {})
        bl = data.get("baseline", {})
        if ssl and bl:
            fractions.append(frac)
            ssl_maps.append(ssl.get("mAP_50", 0))
            base_maps.append(bl.get("mAP_50", 0))

    if len(fractions) < 2:
        print("  [Skip] Need ≥2 fractions for cross-fraction chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fractions, ssl_maps, "o-", color=SSL_COLOR, linewidth=2.5,
            markersize=10, label="SSL (Domain-Adapted)")
    ax.plot(fractions, base_maps, "s--", color=BASELINE_COLOR, linewidth=2.5,
            markersize=10, label="Baseline (Original)")

    for x, s, b in zip(fractions, ssl_maps, base_maps):
        ax.annotate(f"{s:.3f}", (x, s), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color=SSL_COLOR,
                    fontweight="bold")
        ax.annotate(f"{b:.3f}", (x, b), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=10, color=BASELINE_COLOR,
                    fontweight="bold")

    ax.set_xlabel("Data Fraction (%)", fontsize=13)
    ax.set_ylabel("mAP@50", fontsize=13)
    ax.set_title("Data Efficiency: SSL vs Baseline across Fractions",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xticks(fractions)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    save_path = base / "cross_fraction_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
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
    parser.add_argument("--fraction", type=int, required=True,
                        help="Data fraction percentage (10, 25, 50, 100)")
    args = parser.parse_args()

    # Verify dataset path exists
    ds = Path(args.dataset_dir)
    if not ds.exists():
        print(f"  [WARN] dataset_dir {ds} not found, checking alternatives...")
        for alt in [ds.parent / "dataset_coco", ds.parent / f"dataset_coco_frac{args.fraction}"]:
            if alt.exists():
                print(f"  [INFO] Found dataset at {alt}")
                args.dataset_dir = str(alt)
                break

    total_steps = 8
    print("=" * 70)
    print(f"  Thesis Visualization Suite — Fraction {args.fraction}%")
    print("=" * 70)

    print(f"\n[1/{total_steps}] Dataset Sample Visualization...")
    visualize_dataset_samples(args.dataset_dir, args.output_dir)

    print(f"\n[2/{total_steps}] Data Distribution Chart...")
    visualize_data_distribution(args.dataset_dir, args.fraction, args.output_dir)

    print(f"\n[3/{total_steps}] Detection Visualization (SSL)...")
    visualize_detections(args.dataset_dir, args.ssl_dir,
                         "SSL Domain-Adapted", args.output_dir)

    print(f"\n[4/{total_steps}] Detection Visualization (Baseline)...")
    visualize_detections(args.dataset_dir, args.baseline_dir,
                         "Baseline Original", args.output_dir)

    print(f"\n[5/{total_steps}] Side-by-Side Detection Comparison...")
    visualize_side_by_side(args.dataset_dir, args.ssl_dir,
                           args.baseline_dir, args.output_dir)

    print(f"\n[6/{total_steps}] Comparison Bar Chart...")
    visualize_comparison(args.ssl_dir, args.baseline_dir,
                         args.fraction, args.output_dir)

    print(f"\n[7/{total_steps}] Training Curves...")
    visualize_training_curves(args.ssl_dir, args.baseline_dir,
                              args.fraction, args.output_dir)

    print(f"\n[8/{total_steps}] Cross-Fraction Tracker...")
    visualize_cross_fraction(args.output_dir)

    # Summary table (always last)
    print_summary_table(args.ssl_dir, args.baseline_dir,
                        args.fraction, args.output_dir)

    print(f"\n✅ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

