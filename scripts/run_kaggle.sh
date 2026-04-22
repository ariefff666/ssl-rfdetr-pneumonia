#!/bin/bash
# ==============================================================================
# run_kaggle.sh — Execute training pipeline on Kaggle with background support
#
# This script is designed to be run inside a Kaggle notebook terminal.
# It handles the full 3-phase pipeline:
#   Phase 1: Data preparation (RSNA CSV → COCO JSON)
#   Phase 2: SSL continual pre-training (DINOv2 on X-rays)
#   Phase 3: RF-DETR fine-tuning (SSL + Baseline)
#
# Usage (from Kaggle notebook cell):
#   !bash scripts/run_kaggle.sh
#
# To run in background (persists after browser close):
#   !nohup bash scripts/run_kaggle.sh > /kaggle/working/pipeline.log 2>&1 &
#   !echo $!  # Print the process ID for monitoring
# ==============================================================================

set -e  # Exit on error

# --- Environment Setup ---
echo "============================================================"
echo "Pipeline Start: $(date)"
echo "============================================================"

# Install dependencies
pip install -q rfdetr albumentations wandb pycocotools scikit-learn tqdm pyyaml seaborn

# Set W&B API key (replace with your key or set in Kaggle secrets)
# export WANDB_API_KEY="your-wandb-api-key-here"

# Navigate to project root
cd /kaggle/working/ssl-rfdetr-pneumonia

# ==============================================================================
# PHASE 1: Data Preparation
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 1: Preparing COCO dataset from RSNA CSV"
echo "============================================================"

python -m src.data.prepare_coco --config configs/finetune_rfdetr.yaml

# ==============================================================================
# PHASE 2: SSL Continual Pre-training
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 2: DINOv2 SSL Continual Pre-training"
echo "============================================================"

# Use torchrun for multi-GPU (T4x2)
torchrun --nproc_per_node=2 \
    -m src.train_ssl \
    --config configs/ssl_pretrain.yaml

# Identify the final backbone checkpoint
SSL_BACKBONE="/kaggle/working/checkpoints/ssl/backbone_epoch_50.pth"
echo "SSL backbone saved to: ${SSL_BACKBONE}"

# ==============================================================================
# PHASE 3A: RF-DETR Fine-tuning WITH SSL backbone
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 3A: RF-DETR Fine-tuning (WITH SSL backbone)"
echo "============================================================"

python -m src.train_rfdetr \
    --config configs/finetune_rfdetr.yaml \
    --ssl-backbone "${SSL_BACKBONE}" \
    --run-name rfdetr-finetune

# ==============================================================================
# PHASE 3B: RF-DETR Fine-tuning WITHOUT SSL (Baseline)
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 3B: RF-DETR Fine-tuning (BASELINE — original DINOv2)"
echo "============================================================"

python -m src.train_rfdetr \
    --config configs/finetune_rfdetr.yaml \
    --run-name rfdetr-finetune

# ==============================================================================
# PHASE 4: Compare Results
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 4: Comparing SSL vs Baseline results"
echo "============================================================"

python -m src.compare_results \
    --ssl-dir /kaggle/working/checkpoints/rfdetr/rfdetr-finetune-ssl \
    --baseline-dir /kaggle/working/checkpoints/rfdetr/rfdetr-finetune-baseline \
    --output-dir /kaggle/working/comparison

echo ""
echo "============================================================"
echo "Pipeline Complete: $(date)"
echo "Results at: /kaggle/working/comparison/"
echo "============================================================"
