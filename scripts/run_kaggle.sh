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

# --- NCCL Fix for Multi-GPU on Kaggle T4x2 ---
# Disable P2P to prevent NCCL deadlock/timeout
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --- Environment Setup ---
echo "============================================================"
echo "Pipeline Start: $(date)"
echo "============================================================"

# Install dependencies
pip install -q rfdetr albumentations wandb pycocotools scikit-learn tqdm pyyaml seaborn

# Set W&B API key (replace with your key or set in Kaggle secrets)
# export WANDB_API_KEY="your-wandb-api-key-here"

# Navigate to project root and set Python path
cd /kaggle/working/ssl-rfdetr-pneumonia
export PYTHONPATH="/kaggle/working/ssl-rfdetr-pneumonia:$PYTHONPATH"

# Ensure all package directories and __init__.py files exist
mkdir -p src/data src/models src/utils
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py

# Debug: verify package structure
echo "--- Verifying package structure ---"
find src -name "*.py" | head -20
echo "-----------------------------------"

# ==============================================================================
# PHASE 1: Data Preparation
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 1: Preparing COCO dataset from RSNA CSV"
echo "============================================================"

# Force rebuild COCO dataset to apply empty image filter (DDP fix)
# Empty images cause uneven data distribution → NCCL deadlock
rm -rf /kaggle/working/dataset_coco

python3 src/data/prepare_coco.py --config configs/finetune_rfdetr.yaml

# ==============================================================================
# PHASE 2: SSL Continual Pre-training
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 2: DINOv2 SSL Continual Pre-training"
echo "============================================================"

BACKBONE_INPUT_PATH="/kaggle/input/datasets/arief666/rfdetr-final-backbone/backbone_epoch_50.pth"

# Mengecek apakah file backbone sudah tersedia di Kaggle Input
if [ -f "$BACKBONE_INPUT_PATH" ]; then
    echo "=> Backbone sudah ditemukan di: $BACKBONE_INPUT_PATH"
    echo "=> Melewati Phase 2 dan langsung menuju Phase 3..."
    
    # Simpan alamat backbone untuk dipakai di Phase 3
    FINAL_BACKBONE="$BACKBONE_INPUT_PATH"
else
    echo "=> Backbone tidak ditemukan. Memulai Phase 2 dari awal/resume..."
    
    # Fix DINOv2 download conflict (only needed when actually running SSL training)
    rm -rf /root/.cache/torch/hub/  
    python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')"
    
    # Jalankan training SSL
    torchrun --nproc_per_node=2 src/train_ssl.py \
        --config configs/ssl_pretrain.yaml
        
    if [ $? -ne 0 ]; then echo "Error di Phase 2! Menghentikan pipeline."; exit 1; fi
    
    # Jika run Phase 2 sukses, ini adalah lokasi output defaultnya
    FINAL_BACKBONE="/kaggle/working/ssl-rfdetr-pneumonia/checkpoints/ssl/backbone_epoch_50.pth"
fi

# ==============================================================================
# PHASE 3A: RF-DETR Fine-tuning WITH SSL backbone
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 3A: RF-DETR Fine-tuning (WITH SSL backbone)"
echo "============================================================"

pip install faster-coco-eval

# Check for resume checkpoint (if previous SSL run was interrupted)
SSL_RESUME=""
SSL_CKPT_DIR="/kaggle/working/checkpoints/rfdetr/rfdetr-finetune-ssl"
SSL_RESUME_INPUT_PTH="/kaggle/input/datasets/arief666/rfdetr-ssl-checkpoint/checkpoint.pth"
SSL_RESUME_INPUT_CKPT="/kaggle/input/datasets/arief666/rfdetr-ssl-checkpoint/last.ckpt"

if [ -f "${SSL_RESUME_INPUT_PTH}" ]; then
    SSL_RESUME="--resume ${SSL_RESUME_INPUT_PTH}"
    echo "Resuming from uploaded checkpoint: ${SSL_RESUME_INPUT_PTH}"
elif [ -f "${SSL_RESUME_INPUT_CKPT}" ]; then
    SSL_RESUME="--resume ${SSL_RESUME_INPUT_CKPT}"
    echo "Resuming from uploaded checkpoint: ${SSL_RESUME_INPUT_CKPT}"
elif [ -f "${SSL_CKPT_DIR}/checkpoint.pth" ]; then
    SSL_RESUME="--resume ${SSL_CKPT_DIR}/checkpoint.pth"
elif [ -f "${SSL_CKPT_DIR}/last.ckpt" ]; then
    SSL_RESUME="--resume ${SSL_CKPT_DIR}/last.ckpt"
fi

# DDP handled internally by Lightning (devices=2 in config)
python3 src/train_rfdetr.py \
    --config configs/finetune_rfdetr.yaml \
    --ssl-backbone "${FINAL_BACKBONE}" \
    --run-name rfdetr-finetune \
    ${SSL_RESUME}

# ==============================================================================
# PHASE 3B: RF-DETR Fine-tuning WITHOUT SSL (Baseline)
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 3B: RF-DETR Fine-tuning (BASELINE — original DINOv2)"
echo "============================================================"

python3 src/train_rfdetr.py \
    --config configs/finetune_rfdetr.yaml \
    --run-name rfdetr-finetune

# ==============================================================================
# PHASE 4: Compare Results
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 4: Comparing SSL vs Baseline results"
echo "============================================================"

python3 src/compare_results.py \
    --ssl-dir /kaggle/working/checkpoints/rfdetr/rfdetr-finetune-ssl \
    --baseline-dir /kaggle/working/checkpoints/rfdetr/rfdetr-finetune-baseline \
    --output-dir /kaggle/working/comparison

echo ""
echo "============================================================"
echo "Pipeline Complete: $(date)"
echo "Results at: /kaggle/working/comparison/"
echo "============================================================"
