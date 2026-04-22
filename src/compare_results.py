"""
compare_results.py — Compare SSL vs Baseline RF-DETR detection results.

Loads training metadata and W&B logs to produce a side-by-side comparison
table and visualization of mAP metrics.

Usage:
    python -m src.compare_results \\
        --ssl-dir /kaggle/working/checkpoints/rfdetr/rfdetr-finetune-ssl \\
        --baseline-dir /kaggle/working/checkpoints/rfdetr/rfdetr-finetune-baseline \\
        --output-dir /kaggle/working/comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np


def load_metadata(run_dir: str) -> dict:
    """Load training metadata JSON from a run directory."""
    meta_path = Path(run_dir) / "training_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def create_comparison_table(ssl_meta: dict, baseline_meta: dict) -> str:
    """Generate a formatted comparison table."""
    rows = [
        ("Mode", ssl_meta["mode"], baseline_meta["mode"]),
        ("Model Variant", ssl_meta["model_variant"], baseline_meta["model_variant"]),
        ("Epochs", ssl_meta["epochs"], baseline_meta["epochs"]),
        ("Effective Batch Size", ssl_meta["effective_batch_size"], baseline_meta["effective_batch_size"]),
        ("Training Time (min)", ssl_meta["training_time_minutes"], baseline_meta["training_time_minutes"]),
        ("SSL Backbone", ssl_meta.get("ssl_backbone_path", "N/A") or "N/A", "None (Original DINOv2)"),
    ]

    table = "\n" + "=" * 80 + "\n"
    table += f"{'Metric':<30} {'SSL':>20} {'Baseline':>20}\n"
    table += "-" * 80 + "\n"
    for label, ssl_val, base_val in rows:
        table += f"{label:<30} {str(ssl_val):>20} {str(base_val):>20}\n"
    table += "=" * 80 + "\n"

    return table


def plot_comparison(
    ssl_dir: str,
    baseline_dir: str,
    output_dir: str,
) -> None:
    """
    Generate comparison visualizations.

    Looks for evaluation results / W&B exported CSV files in the run directories.
    If not found, creates a template plot that can be filled with actual metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try to load evaluation results
    ssl_results = _try_load_eval_results(ssl_dir)
    baseline_results = _try_load_eval_results(baseline_dir)

    if ssl_results and baseline_results:
        _plot_metric_comparison(ssl_results, baseline_results, output_path)
    else:
        print("[Compare] Evaluation results not found in run directories.")
        print("[Compare] To generate comparison plots, please run evaluation first")
        print("          or export metrics from W&B dashboard.")
        _create_template_plot(output_path)


def _try_load_eval_results(run_dir: str) -> dict | None:
    """Attempt to load evaluation metrics from a run directory."""
    # Check various possible output locations
    for filename in ["eval_results.json", "metrics.json", "results.json"]:
        path = Path(run_dir) / filename
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None


def _plot_metric_comparison(
    ssl_results: dict,
    baseline_results: dict,
    output_path: Path,
) -> None:
    """Create bar chart comparing key metrics between SSL and baseline."""
    metric_keys = ["mAP", "AP50", "AP75", "AR_max100"]
    metric_labels = ["mAP@[.5:.95]", "AP@.50", "AP@.75", "AR@100"]

    ssl_values = [ssl_results.get(k, 0) for k in metric_keys]
    baseline_values = [baseline_results.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_ssl = ax.bar(x - width / 2, ssl_values, width, label="SSL (Domain-Adapted)", color="#2196F3")
    bars_base = ax.bar(x + width / 2, baseline_values, width, label="Baseline (Original DINOv2)", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("RF-DETR Detection Performance: SSL vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars_ssl, bars_base]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    save_path = output_path / "comparison_chart.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Compare] Chart saved to: {save_path}")


def _create_template_plot(output_path: Path) -> None:
    """Create a template plot with placeholder values."""
    print("[Compare] Creating template comparison (replace with actual metrics)")

    metric_labels = ["mAP@[.5:.95]", "AP@.50", "AP@.75", "AR@100"]
    ssl_placeholder = [0.0, 0.0, 0.0, 0.0]
    baseline_placeholder = [0.0, 0.0, 0.0, 0.0]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, ssl_placeholder, width, label="SSL (Domain-Adapted)", color="#2196F3", alpha=0.5)
    ax.bar(x + width / 2, baseline_placeholder, width, label="Baseline (Original DINOv2)", color="#FF9800", alpha=0.5)

    ax.set_ylabel("Score")
    ax.set_title("RF-DETR Detection Performance: SSL vs Baseline [TEMPLATE]")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.text(0.5, 0.5, "Awaiting actual metrics\nfrom training runs",
            transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray")

    plt.tight_layout()
    save_path = output_path / "comparison_chart_template.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Compare] Template saved to: {save_path}")


def main(ssl_dir: str, baseline_dir: str, output_dir: str) -> None:
    """Main comparison pipeline."""
    print("=" * 70)
    print("RF-DETR Results Comparison: SSL vs Baseline")
    print("=" * 70)

    ssl_meta = load_metadata(ssl_dir)
    baseline_meta = load_metadata(baseline_dir)

    # Print comparison table
    table = create_comparison_table(ssl_meta, baseline_meta)
    print(table)

    # Save table
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "comparison_summary.txt", "w") as f:
        f.write(table)

    # Generate plots
    plot_comparison(ssl_dir, baseline_dir, output_dir)

    print(f"\nAll comparison results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SSL vs Baseline RF-DETR results")
    parser.add_argument("--ssl-dir", type=str, required=True, help="Path to SSL run output dir")
    parser.add_argument("--baseline-dir", type=str, required=True, help="Path to baseline run output dir")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/comparison", help="Output directory")
    args = parser.parse_args()
    main(args.ssl_dir, args.baseline_dir, args.output_dir)
