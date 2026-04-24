#!/bin/bash
# ==============================================================================
# run_kaggle.sh — Run SSL + Baseline experiments for one data fraction
#
# Features:
#   - Auto-resume: detects last.ckpt and resumes training if found
#   - Skips completed runs (if best checkpoint already exists)
#   - Runs both SSL + Baseline sequentially, then generates visualizations
#
# Usage:
#   bash scripts/run_kaggle.sh 0.1     # Run 10% fraction
#   bash scripts/run_kaggle.sh 0.25    # Run 25% fraction
#   bash scripts/run_kaggle.sh 0.5     # Run 50% fraction
#   bash scripts/run_kaggle.sh 1.0     # Run 100% fraction
#
# Background:
#   nohup bash scripts/run_kaggle.sh 0.1 > /kaggle/working/pipeline.log 2>&1 &
# ==============================================================================

set -e

# --- Parse fraction argument ---
FRACTION="${1:-0.1}"
FRAC_PCT=$(python3 -c "print(int(float('${FRACTION}') * 100))")
echo "============================================================"
echo "Pipeline: Fraction ${FRAC_PCT}% — $(date)"
echo "============================================================"

# --- NCCL Fix ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --- Setup ---
pip install -q rfdetr albumentations wandb pycocotools scikit-learn tqdm pyyaml seaborn faster-coco-eval

cd /kaggle/working/ssl-rfdetr-pneumonia
export PYTHONPATH="/kaggle/working/ssl-rfdetr-pneumonia:$PYTHONPATH"

mkdir -p src/data src/models src/utils
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/utils/__init__.py

# Hapus cache dinov2 agar tidak konflik
rm -rf /root/.cache/torch/hub/
python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')"

# --- GPU Detection ---
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected ${NUM_GPUS} GPU(s), Fraction=${FRAC_PCT}%"

# --- Backbone ---
BACKBONE_INPUT_PATH="/kaggle/input/datasets/arief666/rfdetr-final-backbonep16/backbone_epoch_50 (p16).pth"
if [ ! -f "$BACKBONE_INPUT_PATH" ]; then
    echo "ERROR: Backbone not found at $BACKBONE_INPUT_PATH"
    exit 1
fi
FINAL_BACKBONE="$BACKBONE_INPUT_PATH"

# --- Output directories ---
SSL_RUN_NAME="rfdetr-frac${FRAC_PCT}"
BASELINE_RUN_NAME="rfdetr-frac${FRAC_PCT}"
SSL_DIR="/kaggle/working/checkpoints/rfdetr/${SSL_RUN_NAME}-ssl"
BASELINE_DIR="/kaggle/working/checkpoints/rfdetr/${BASELINE_RUN_NAME}-baseline"

# --- Dataset dir per fraction ---
if [ "$FRAC_PCT" -eq 100 ]; then
    DATASET_DIR="/kaggle/working/dataset_coco"
else
    DATASET_DIR="/kaggle/working/dataset_coco_frac${FRAC_PCT}"
fi

# --- Temp config ---
TEMP_CONFIG="/tmp/finetune_frac${FRAC_PCT}.yaml"

# ==============================================================================
# PHASE 1: Prepare COCO dataset (skip if already exists)
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 1: Preparing COCO dataset (${FRAC_PCT}% fraction)"
echo "============================================================"

if [ -f "${DATASET_DIR}/train/_annotations.coco.json" ]; then
    echo "=> Dataset sudah ada di ${DATASET_DIR}, skip preparation."
else
    rm -rf "${DATASET_DIR}"

    python3 -c "
import yaml
with open('configs/finetune_rfdetr.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['data_fraction'] = ${FRACTION}
cfg['data']['dataset_dir'] = '${DATASET_DIR}'
with open('${TEMP_CONFIG}', 'w') as f:
    yaml.dump(cfg, f)
print('Config written to ${TEMP_CONFIG}')
print(f'  data_fraction: {cfg[\"data\"][\"data_fraction\"]}')
print(f'  dataset_dir: {cfg[\"data\"][\"dataset_dir\"]}')
"

    python3 src/data/prepare_coco.py --config "${TEMP_CONFIG}"
fi

# Pastikan config selalu ada (meski dataset di-skip)
if [ ! -f "${TEMP_CONFIG}" ]; then
    python3 -c "
import yaml
with open('configs/finetune_rfdetr.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['data']['data_fraction'] = ${FRACTION}
cfg['data']['dataset_dir'] = '${DATASET_DIR}'
with open('${TEMP_CONFIG}', 'w') as f:
    yaml.dump(cfg, f)
"
fi

# ==============================================================================
# Helper function: run training with auto-resume
# ==============================================================================
run_train() {
    local MODE="$1"          # "ssl" atau "baseline"
    local RUN_NAME="$2"
    local OUTPUT_DIR="$3"
    local EXTRA_ARGS="$4"    # e.g., --ssl-backbone path

    echo ""
    echo "------------------------------------------------------------"
    echo "  Mode: ${MODE} | Run: ${RUN_NAME}"
    echo "------------------------------------------------------------"

    # Check: apakah training sudah selesai? (best checkpoint exists + no last.ckpt)
    if [ -f "${OUTPUT_DIR}/checkpoint_best_regular.pth" ] && [ ! -f "${OUTPUT_DIR}/last.ckpt" ]; then
        echo "=> Training ${MODE} sudah SELESAI (best checkpoint ditemukan, no last.ckpt)."
        echo "=> Skip training. Checkpoint di: ${OUTPUT_DIR}"
        return 0
    fi

    # Check: ada last.ckpt? → resume
    RESUME_FLAG=""
    if [ -f "${OUTPUT_DIR}/last.ckpt" ]; then
        echo "=> last.ckpt DITEMUKAN di ${OUTPUT_DIR}/last.ckpt"
        echo "=> Melanjutkan (RESUME) training dari checkpoint..."
        RESUME_FLAG="--resume ${OUTPUT_DIR}/last.ckpt"
    else
        echo "=> Tidak ada checkpoint. Memulai training dari AWAL..."
    fi

    if [ "${NUM_GPUS}" -gt 1 ]; then
        # Gunakan port berbeda untuk SSL vs Baseline agar tidak konflik
        local PORT=29500
        if [ "${MODE}" = "baseline" ]; then
            PORT=29501
        fi

        torchrun --nproc_per_node=${NUM_GPUS} --master_port=${PORT} \
            src/train_rfdetr.py \
            --config "${TEMP_CONFIG}" \
            --run-name "${RUN_NAME}" \
            ${EXTRA_ARGS} \
            ${RESUME_FLAG}
    else
        python3 src/train_rfdetr.py \
            --config "${TEMP_CONFIG}" \
            --run-name "${RUN_NAME}" \
            ${EXTRA_ARGS} \
            ${RESUME_FLAG}
    fi
}

# ==============================================================================
# PHASE 2A: RF-DETR Fine-tune — SSL backbone
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 2A: RF-DETR Fine-tune — SSL backbone (${FRAC_PCT}%)"
echo "============================================================"

run_train "ssl" "${SSL_RUN_NAME}" "${SSL_DIR}" "--ssl-backbone \"${FINAL_BACKBONE}\""

# ==============================================================================
# PHASE 2B: RF-DETR Fine-tune — Baseline
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 2B: RF-DETR Fine-tune — Baseline (${FRAC_PCT}%)"
echo "============================================================"

run_train "baseline" "${BASELINE_RUN_NAME}" "${BASELINE_DIR}" ""

# ==============================================================================
# PHASE 3: Visualization & Comparison
# ==============================================================================
echo ""
echo "============================================================"
echo "PHASE 3: Generating Visualizations (${FRAC_PCT}%)"
echo "============================================================"

VIZ_DIR="/kaggle/working/visualizations/frac${FRAC_PCT}"

python3 src/visualize.py \
    --dataset-dir "${DATASET_DIR}" \
    --ssl-dir "${SSL_DIR}" \
    --baseline-dir "${BASELINE_DIR}" \
    --output-dir "${VIZ_DIR}" \
    --fraction ${FRAC_PCT}

echo ""
echo "============================================================"
echo "DONE: Fraction ${FRAC_PCT}% — $(date)"
echo "  SSL checkpoints:       ${SSL_DIR}"
echo "  Baseline checkpoints:  ${BASELINE_DIR}"
echo "  Visualizations:        ${VIZ_DIR}"
echo "============================================================"
