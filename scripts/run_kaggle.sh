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
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --- Environment Setup ---
echo "============================================================"
echo "Pipeline Start: $(date)"
echo "============================================================"

# Install dependencies
pip install -q rfdetr albumentations wandb pycocotools scikit-learn tqdm pyyaml seaborn

# Navigate to project root and set Python path
cd /kaggle/working/ssl-rfdetr-pneumonia
export PYTHONPATH="/kaggle/working/ssl-rfdetr-pneumonia:$PYTHONPATH"

# Ensure all package directories
mkdir -p src/data src/models src/utils
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/utils/__init__.py

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

# Force rebuild COCO dataset
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

if [ -f "$BACKBONE_INPUT_PATH" ]; then
    echo "=> Backbone sudah ditemukan di: $BACKBONE_INPUT_PATH"
    echo "=> Melewati Phase 2 dan langsung menuju Phase 3..."
    FINAL_BACKBONE="$BACKBONE_INPUT_PATH"
else
    echo "=> Backbone tidak ditemukan. Memulai Phase 2 dari awal/resume..."
    rm -rf /root/.cache/torch/hub/
    python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')"
    torchrun --nproc_per_node=2 src/train_ssl.py --config configs/ssl_pretrain.yaml
    if [ $? -ne 0 ]; then echo "Error di Phase 2!"; exit 1; fi
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

# Detect GPU count
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected ${NUM_GPUS} GPU(s)"

# Jalankan torchrun — train_rfdetr.py sekarang sudah handle dist.init_process_group()
# secara benar sehingga barrier() berfungsi dan race condition tidak terjadi
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Using DDP with ${NUM_GPUS} GPUs via torchrun"
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29500 \
        src/train_rfdetr.py \
        --config configs/finetune_rfdetr.yaml \
        --ssl-backbone "${FINAL_BACKBONE}" \
        --run-name rfdetr-finetune
else
    echo "Using single GPU"
    python3 src/train_rfdetr.py \
        --config configs/finetune_rfdetr.yaml \
        --ssl-backbone "${FINAL_BACKBONE}" \
        --run-name rfdetr-finetune
fi

# ==============================================================================
# PHASE 3B: RF-DETR Fine-tuning WITHOUT SSL (Baseline)
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 3B: RF-DETR Fine-tuning (BASELINE — original DINOv2)"
echo "============================================================"

if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29501 \
        src/train_rfdetr.py \
        --config configs/finetune_rfdetr.yaml \
        --run-name rfdetr-finetune
else
    python3 src/train_rfdetr.py \
        --config configs/finetune_rfdetr.yaml \
        --run-name rfdetr-finetune
fi

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
