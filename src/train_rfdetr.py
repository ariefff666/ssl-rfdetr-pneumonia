"""
train_rfdetr.py — Entry point for RF-DETR fine-tuning on RSNA Pneumonia Detection.

Supports two modes:
  1. WITH SSL backbone: Load domain-adapted DINOv2 weights → fine-tune RF-DETR
  2. WITHOUT SSL (baseline): Use original DINOv2 weights → fine-tune RF-DETR

This allows direct comparison of detection performance between the two approaches.

The script uses the official `rfdetr` package API for training, which
internally handles COCO-format data loading, augmentation, and evaluation.

Usage:
    # WITH SSL backbone (after running train_ssl.py)
    python -m src.train_rfdetr --config configs/finetune_rfdetr.yaml \\
        --ssl-backbone /kaggle/working/checkpoints/ssl/backbone_epoch_50.pth \\
        --run-name rfdetr-with-ssl

    # WITHOUT SSL (baseline)
    python -m src.train_rfdetr --config configs/finetune_rfdetr.yaml \\
        --run-name rfdetr-baseline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in Python path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import yaml
import wandb

from src.utils.logger import init_wandb, finish_wandb


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_class(variant: str):
    """
    Dynamically import the correct RF-DETR model class.

    RF-DETR variants: RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
    """
    import rfdetr

    variant_map = {
        "RFDETRNano": rfdetr.RFDETRNano,
        "RFDETRSmall": rfdetr.RFDETRSmall,
        "RFDETRMedium": rfdetr.RFDETRMedium,
        "RFDETRLarge": rfdetr.RFDETRLarge,
    }

    if variant not in variant_map:
        raise ValueError(
            f"Unknown RF-DETR variant: {variant}. "
            f"Choose from: {list(variant_map.keys())}"
        )

    return variant_map[variant]


def inject_ssl_backbone(model, ssl_backbone_path: str) -> None:
    """
    Replace the DINOv2 backbone weights inside the RF-DETR model
    with our domain-adapted SSL pre-trained weights.

    This is the critical step that connects the SSL phase to detection.

    Args:
        model: An instantiated RF-DETR model object.
        ssl_backbone_path: Path to the backbone_epoch_N.pth file from SSL pre-training.
    """
    print(f"\n[SSL] Loading domain-adapted backbone from: {ssl_backbone_path}")

    ssl_state_dict = torch.load(ssl_backbone_path, map_location="cpu", weights_only=True)

    # The RF-DETR model stores its backbone under model.model.backbone
    # We need to find the DINOv2 backbone within RF-DETR's architecture
    # and replace its weights with our SSL-adapted ones.
    #
    # RF-DETR internally uses a DINOv2 ViT as backbone.
    # The exact attribute path depends on rfdetr's internal structure.
    # We attempt multiple common patterns for compatibility.
    backbone = None
    backbone_attr_paths = [
        "model.backbone",
        "model.model.backbone",
        "backbone",
    ]

    for attr_path in backbone_attr_paths:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            backbone = obj
            print(f"[SSL] Found backbone at: {attr_path}")
            break
        except AttributeError:
            continue

    if backbone is None:
        print("[SSL] WARNING: Could not locate backbone via attribute path.")
        print("[SSL] Attempting state_dict key matching instead...")
        # Fallback: match keys by pattern and load partially
        _load_ssl_weights_by_key_matching(model, ssl_state_dict)
        return

    # Load SSL weights into the backbone
    missing, unexpected = backbone.load_state_dict(ssl_state_dict, strict=False)

    if missing:
        print(f"[SSL] Missing keys (will use random init): {len(missing)}")
        for k in missing[:5]:
            print(f"       - {k}")
    if unexpected:
        print(f"[SSL] Unexpected keys (ignored): {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"       - {k}")

    loaded = len(ssl_state_dict) - len(unexpected)
    print(f"[SSL] Successfully loaded {loaded}/{len(ssl_state_dict)} SSL backbone parameters")


def _load_ssl_weights_by_key_matching(model, ssl_state_dict: dict) -> None:
    """
    Fallback: load SSL weights by matching key suffixes.

    If we can't find the exact backbone attribute, we match SSL state dict
    keys against the full model's state dict by suffix pattern.
    """
    model_state = model.state_dict() if hasattr(model, 'state_dict') else {}
    loaded_count = 0

    for ssl_key, ssl_val in ssl_state_dict.items():
        # Try to find a matching key in the model
        for model_key in model_state:
            if model_key.endswith(ssl_key) or ssl_key in model_key:
                if model_state[model_key].shape == ssl_val.shape:
                    model_state[model_key] = ssl_val
                    loaded_count += 1
                    break

    if loaded_count > 0 and hasattr(model, 'load_state_dict'):
        model.load_state_dict(model_state, strict=False)

    print(f"[SSL] Key-matching loaded {loaded_count} parameters")


def main(config_path: str, ssl_backbone_path: str | None, run_name: str | None) -> None:
    """Main RF-DETR fine-tuning pipeline."""
    cfg = load_config(config_path)

    # Determine mode
    is_ssl = ssl_backbone_path is not None
    mode_str = "WITH SSL backbone" if is_ssl else "BASELINE (original DINOv2)"

    print("=" * 70)
    print(f"RF-DETR Fine-Tuning — {mode_str}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Prepare dataset (COCO format)
    # -------------------------------------------------------------------------
    dataset_dir = cfg["data"]["dataset_dir"]

    # Check if COCO dataset exists; if not, run preparation
    coco_train_ann = Path(dataset_dir) / "train" / "_annotations.coco.json"
    if not coco_train_ann.exists():
        print("\nCOCO dataset not found. Running data preparation...")
        from src.data.prepare_coco import main as prepare_main
        prepare_main(config_path)
    else:
        print(f"\nCOCO dataset found at: {dataset_dir}")

    # -------------------------------------------------------------------------
    # Initialize RF-DETR model
    # -------------------------------------------------------------------------
    model_cfg = cfg["model"]
    ModelClass = get_model_class(model_cfg["variant"])

    print(f"\nInitializing {model_cfg['variant']}...")
    model = ModelClass()

    # Inject SSL backbone if provided
    if is_ssl:
        inject_ssl_backbone(model, ssl_backbone_path)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    train_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]
    log_cfg = cfg["logging"]

    # Resolve run name
    final_run_name = run_name or log_cfg["wandb_run_name"]
    if is_ssl:
        final_run_name += "-ssl"
    else:
        final_run_name += "-baseline"

    output_dir = Path(ckpt_cfg["output_dir"]) / final_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training...")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Grad accum: {train_cfg['grad_accum_steps']}")
    print(f"  Effective batch: {train_cfg['batch_size'] * train_cfg['grad_accum_steps']}")
    print(f"  Output: {output_dir}")
    print(f"  W&B run: {final_run_name}")

    # Build training kwargs for RF-DETR's .train() method
    train_kwargs = {
        "dataset_dir": dataset_dir,
        "epochs": train_cfg["epochs"],
        "batch_size": train_cfg["batch_size"],
        "grad_accum_steps": train_cfg["grad_accum_steps"],
        "output_dir": str(output_dir),
    }

    # Optional LR
    if "learning_rate" in train_cfg:
        train_kwargs["lr"] = train_cfg["learning_rate"]

    # W&B integration (built into rfdetr)
    if log_cfg.get("use_wandb"):
        train_kwargs["wandb"] = True
        train_kwargs["project"] = log_cfg["wandb_project"]
        train_kwargs["run"] = final_run_name

    # Early stopping
    if train_cfg.get("early_stopping"):
        train_kwargs["early_stopping"] = True
        if "early_stopping_patience" in train_cfg:
            train_kwargs["early_stopping_patience"] = train_cfg["early_stopping_patience"]
        if "early_stopping_min_delta" in train_cfg:
            train_kwargs["early_stopping_min_delta"] = train_cfg["early_stopping_min_delta"]

    start_time = time.time()
    model.train(**train_kwargs)
    total_time = time.time() - start_time

    print(f"\nTraining complete in {total_time / 60:.1f} minutes")
    print(f"Checkpoints saved to: {output_dir}")

    # -------------------------------------------------------------------------
    # Save metadata for comparison
    # -------------------------------------------------------------------------
    metadata = {
        "mode": "ssl" if is_ssl else "baseline",
        "ssl_backbone_path": ssl_backbone_path,
        "model_variant": model_cfg["variant"],
        "epochs": train_cfg["epochs"],
        "batch_size": train_cfg["batch_size"],
        "effective_batch_size": train_cfg["batch_size"] * train_cfg["grad_accum_steps"],
        "training_time_minutes": round(total_time / 60, 2),
        "output_dir": str(output_dir),
    }

    meta_path = output_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR Fine-tuning for Pneumonia Detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_rfdetr.yaml",
        help="Path to RF-DETR config YAML",
    )
    parser.add_argument(
        "--ssl-backbone",
        type=str,
        default=None,
        help="Path to SSL pre-trained backbone .pth file. "
             "If omitted, trains with original DINOv2 weights (baseline).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom W&B run name (default: from config + mode suffix)",
    )
    args = parser.parse_args()
    main(args.config, args.ssl_backbone, args.run_name)
