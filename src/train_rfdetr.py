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


def _normalize_rfdetr_key(key: str) -> str:
    """
    Normalize an RF-DETR backbone key to DINOv2-equivalent naming.

    RF-DETR uses HuggingFace's Dinov2Model internally, which has different
    key naming than Facebook's DINOv2 from torch.hub. This function converts
    RF-DETR keys to the Facebook DINOv2 naming convention.

    Example mappings:
        0.encoder.encoder.embeddings.cls_token         →  cls_token
        0.encoder.encoder.encoder.layer.0.norm1.weight →  blocks.0.norm1.weight
        0.encoder.encoder.encoder.layer.0.attention.attention.query.weight → blocks.0.attn.q.weight
        0.encoder.encoder.encoder.layer.0.layer_scale1.lambda1            → blocks.0.ls1.gamma
        0.encoder.encoder.layernorm.weight             →  norm.weight
    """
    import re

    # Remove leading index prefix (e.g., "0.")
    key = re.sub(r"^\d+\.", "", key)

    # Embeddings: strip the long prefix
    key = key.replace("encoder.encoder.embeddings.", "")

    # Patch embeddings
    key = key.replace("patch_embeddings.projection.", "patch_embed.proj.")

    # Position embeddings
    key = key.replace("position_embeddings", "pos_embed")

    # Encoder layers → blocks
    key = re.sub(r"encoder\.encoder\.encoder\.layer\.(\d+)\.", r"blocks.\1.", key)
    key = re.sub(r"encoder\.encoder\.layer\.(\d+)\.", r"blocks.\1.", key)
    key = re.sub(r"encoder\.layer\.(\d+)\.", r"blocks.\1.", key)

    # Attention
    key = key.replace("attention.output.dense.", "attn.proj.")
    key = key.replace("attention.attention.query.", "attn.q.")
    key = key.replace("attention.attention.key.", "attn.k.")
    key = key.replace("attention.attention.value.", "attn.v.")

    # Layer scale
    key = key.replace("layer_scale1.lambda1", "ls1.gamma")
    key = key.replace("layer_scale2.lambda1", "ls2.gamma")

    # Final layer norm
    key = key.replace("encoder.encoder.layernorm.", "norm.")
    key = re.sub(r"^encoder\.layernorm\.", "norm.", key)

    return key


def inject_ssl_backbone(model, ssl_backbone_path: str) -> None:
    """
    Inject SSL pre-trained DINOv2 weights into RF-DETR's backbone.

    Handles the architecture mismatch between Facebook DINOv2 (patch_size=14,
    fused QKV, fused SwiGLU w12) and RF-DETR's HuggingFace DINOv2 backbone
    (patch_size=16, separate Q/K/V, separate w1/w2) by:

    1. Normalizing key names between the two formats
    2. Splitting fused QKV weights → separate Q, K, V
    3. Splitting fused SwiGLU w12 → separate w1, w2
    4. Skipping incompatible layers (patch_embed, pos_embed)

    Args:
        model: An instantiated RF-DETR model object.
        ssl_backbone_path: Path to backbone_epoch_N.pth from SSL pre-training.
    """
    print(f"\n[SSL] Loading domain-adapted backbone from: {ssl_backbone_path}")

    ssl_state_dict = torch.load(ssl_backbone_path, map_location="cpu", weights_only=True)

    # --- Locate the backbone inside RF-DETR ---
    backbone = None
    for attr_path in ["model.backbone", "model.model.backbone", "backbone"]:
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
        print("[SSL] ERROR: Could not locate backbone in RF-DETR model.")
        return

    rfdetr_state = backbone.state_dict()

    # --- Build reverse map: normalized_key → rfdetr_original_key ---
    rfdetr_norm_map = {}  # normalized DINOv2-style key → original RF-DETR key
    for k in rfdetr_state:
        norm_k = _normalize_rfdetr_key(k)
        rfdetr_norm_map[norm_k] = k

    # --- Diagnostic info ---
    rfdetr_keys = sorted(rfdetr_state.keys())
    ssl_keys = sorted(ssl_state_dict.keys())
    print(f"[SSL] RF-DETR backbone: {len(rfdetr_keys)} parameters")
    print(f"[SSL] SSL backbone: {len(ssl_keys)} parameters")
    print(f"[SSL] RF-DETR keys (sample): {rfdetr_keys[:3]}")
    print(f"[SSL] Normalized (sample):   {[_normalize_rfdetr_key(k) for k in rfdetr_keys[:3]]}")
    print(f"[SSL] SSL keys (sample):     {ssl_keys[:3]}")

    # --- Transfer weights ---
    new_state = dict(rfdetr_state)  # Start from RF-DETR's pre-trained weights
    loaded, skipped_incompat, skipped_shape, skipped_nomap = 0, 0, 0, 0

    for ssl_key, ssl_val in ssl_state_dict.items():

        # Skip inherently incompatible layers (different patch size / resolution)
        if ssl_key in ("pos_embed", "register_tokens") or "patch_embed" in ssl_key:
            skipped_incompat += 1
            continue

        # ----- Fused QKV → separate Q, K, V -----
        if ".attn.qkv." in ssl_key:
            suffix = ssl_key.rsplit(".", 1)[-1]  # "weight" or "bias"
            block_prefix = ssl_key.split(".attn.qkv.")[0]  # "blocks.N"

            q_norm = f"{block_prefix}.attn.q.{suffix}"
            k_norm = f"{block_prefix}.attn.k.{suffix}"
            v_norm = f"{block_prefix}.attn.v.{suffix}"

            if all(nk in rfdetr_norm_map for nk in [q_norm, k_norm, v_norm]):
                dim = ssl_val.shape[0] // 3
                q_val, k_val, v_val = ssl_val[:dim], ssl_val[dim:2*dim], ssl_val[2*dim:]

                for nk, val in [(q_norm, q_val), (k_norm, k_val), (v_norm, v_val)]:
                    target_key = rfdetr_norm_map[nk]
                    if rfdetr_state[target_key].shape == val.shape:
                        new_state[target_key] = val
                        loaded += 1
                    else:
                        skipped_shape += 1
            else:
                skipped_nomap += 1
            continue

        # ----- Fused SwiGLU w12 → separate w1, w2 -----
        if ".mlp.w12." in ssl_key:
            suffix = ssl_key.rsplit(".", 1)[-1]
            block_prefix = ssl_key.split(".mlp.w12.")[0]

            w1_norm = f"{block_prefix}.mlp.w1.{suffix}"
            w2_norm = f"{block_prefix}.mlp.w2.{suffix}"

            if all(nk in rfdetr_norm_map for nk in [w1_norm, w2_norm]):
                dim = ssl_val.shape[0] // 2
                w1_val, w2_val = ssl_val[:dim], ssl_val[dim:]

                for nk, val in [(w1_norm, w1_val), (w2_norm, w2_val)]:
                    target_key = rfdetr_norm_map[nk]
                    if rfdetr_state[target_key].shape == val.shape:
                        new_state[target_key] = val
                        loaded += 1
                    else:
                        skipped_shape += 1
            else:
                skipped_nomap += 1
            continue

        # ----- Direct 1:1 mapping -----
        if ssl_key in rfdetr_norm_map:
            target_key = rfdetr_norm_map[ssl_key]
            if rfdetr_state[target_key].shape == ssl_val.shape:
                new_state[target_key] = ssl_val
                loaded += 1
            else:
                skipped_shape += 1
                print(f"[SSL]   Shape mismatch: {ssl_key} "
                      f"SSL={list(ssl_val.shape)} vs RF-DETR={list(rfdetr_state[target_key].shape)}")
        else:
            skipped_nomap += 1

    # --- Apply updated weights ---
    backbone.load_state_dict(new_state)

    total = len(ssl_state_dict)
    print(f"\n[SSL] === Injection Summary ===")
    print(f"[SSL]   Loaded:       {loaded}/{total} parameters")
    print(f"[SSL]   Incompatible: {skipped_incompat} (patch_embed, pos_embed — expected)")
    print(f"[SSL]   Shape mismatch: {skipped_shape}")
    print(f"[SSL]   No mapping:   {skipped_nomap}")
    pct = (loaded / total * 100) if total > 0 else 0
    print(f"[SSL]   Transfer rate: {pct:.1f}%")


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
    data_fraction = cfg["data"].get("data_fraction", 1.0)

    # Use separate dataset dir for fractional datasets to avoid overwriting
    if data_fraction < 1.0:
        frac_pct = int(data_fraction * 100)
        dataset_dir = f"{dataset_dir}_frac{frac_pct}"
        cfg["data"]["dataset_dir"] = dataset_dir  # Update config for prepare_coco

    # Check if COCO dataset exists; if not, run preparation
    coco_train_ann = Path(dataset_dir) / "train" / "_annotations.coco.json"
    if not coco_train_ann.exists():
        print(f"\nCOCO dataset not found. Running data preparation (fraction={data_fraction})...")
        from src.data.prepare_coco import main as prepare_main
        prepare_main(config_path, data_fraction_override=data_fraction,
                     output_dir_override=dataset_dir)
    else:
        print(f"\nCOCO dataset found at: {dataset_dir}")

    # -------------------------------------------------------------------------
    # Initialize RF-DETR model
    # -------------------------------------------------------------------------
    model_cfg = cfg["model"]
    ModelClass = get_model_class(model_cfg["variant"])

    print(f"\nInitializing {model_cfg['variant']}...")
    model = ModelClass()

    # -------------------------------------------------------------------------
    # Inject SSL backbone and prepare pretrain_weights
    # -------------------------------------------------------------------------
    # IMPORTANT: RF-DETR's .train() re-loads rf-detr-small.pth internally,
    # which overwrites any backbone weights we inject directly. To persist
    # our SSL backbone, we must:
    #   1. Inject SSL weights into the model
    #   2. Save the full modified model to a temp checkpoint
    #   3. Pass it to .train(pretrain_weights=...) so RF-DETR uses OUR weights
    pretrain_weights_path = None

    if is_ssl:
        inject_ssl_backbone(model, ssl_backbone_path)

        # Save modified model as a full checkpoint for .train() to load.
        # RF-DETR's internal hierarchy is non-standard, so we need to
        # discover where the actual nn.Module with state_dict() lives.
        ckpt_cfg = cfg["checkpoint"]
        output_dir_base = Path(ckpt_cfg["output_dir"])
        output_dir_base.mkdir(parents=True, exist_ok=True)

        # Discover the internal nn.Module
        internal_module = None
        search_paths = [
            "model.model.model",  # RFDETRSmall → ModelContext → LWDETR
            "model.model",        # RFDETRSmall → LWDETR
            "model",              # Fallback
        ]

        # Debug: print object types at each level
        print("\n[SSL] Discovering RF-DETR internal structure:")
        obj = model
        depth = 0
        while hasattr(obj, 'model') and depth < 5:
            has_sd = callable(getattr(obj, 'state_dict', None))
            print(f"[SSL]   {'model.' * depth}model → {type(obj).__name__} (state_dict={has_sd})")
            obj = obj.model
            depth += 1
        has_sd = callable(getattr(obj, 'state_dict', None))
        print(f"[SSL]   {'model.' * depth} → {type(obj).__name__} (state_dict={has_sd})")

        for attr_path in search_paths:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                if callable(getattr(obj, 'state_dict', None)):
                    internal_module = obj
                    print(f"[SSL] Found saveable module at: {attr_path} ({type(obj).__name__})")
                    break
            except AttributeError:
                continue

        if internal_module is not None:
            pretrain_weights_path = str(output_dir_base / "rfdetr_with_ssl_backbone.pth")
            torch.save(internal_module.state_dict(), pretrain_weights_path)
            print(f"[SSL] Saved SSL-injected model to: {pretrain_weights_path}")
        else:
            print("[SSL] WARNING: Could not find internal module with state_dict().")
            print("[SSL] Proceeding with direct backbone injection (may be overwritten by .train()).")

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
    print(f"  Devices: {train_cfg.get('devices', 1)}")
    print(f"  Effective batch: {train_cfg['batch_size'] * train_cfg['grad_accum_steps'] * train_cfg.get('devices', 1)}")
    print(f"  Output: {output_dir}")
    print(f"  W&B run: {final_run_name}")
    if pretrain_weights_path:
        print(f"  Pretrain weights: {pretrain_weights_path} (SSL-injected)")
    # Resolve resume path (takes priority over pretrain_weights)
    resume_path = train_cfg.get("resume_from")
    if resume_path:
        print(f"  Resume from: {resume_path}")

    # Build training kwargs for RF-DETR's .train() method
    train_kwargs = {
        "dataset_dir": dataset_dir,
        "epochs": train_cfg["epochs"],
        "batch_size": train_cfg["batch_size"],
        "grad_accum_steps": train_cfg["grad_accum_steps"],
        "output_dir": str(output_dir),
        "devices": train_cfg.get("devices", 1),
    }

    # Resume takes priority: loads model + optimizer + scheduler + epoch
    # pretrain_weights only loads model weights (for fresh start with SSL backbone)
    if resume_path:
        train_kwargs["resume"] = resume_path
    elif pretrain_weights_path:
        train_kwargs["pretrain_weights"] = pretrain_weights_path

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
