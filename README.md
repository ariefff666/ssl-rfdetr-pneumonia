# SSL RF-DETR: Domain-Adaptive Self-Supervised Learning for Pneumonia Detection

> **Skripsi Project** — *Domain-Adaptive Continual Pre-training using Self-Supervised Learning on DINOv2 Backbone for RF-DETR Pneumonia Detection on Chest X-Ray Images*

## Overview

This project investigates whether **continually pre-training** the DINOv2 backbone (used inside RF-DETR) on **unlabeled medical chest X-ray images** improves pneumonia detection accuracy compared to using the original DINOv2 weights pre-trained on natural images.

### Pipeline

```
┌──────────────────────────────────┐
│  Phase 1: Data Preparation       │   RSNA CSV → COCO JSON
│  (prepare_coco.py)               │   Stratified train/valid/test split
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Phase 2: SSL Pre-training       │   DINOv2 self-distillation
│  (train_ssl.py)                  │   on ALL X-ray images (no labels)
│  Output: backbone_epoch_50.pth   │   → Domain-adapted features
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Phase 3A: RF-DETR + SSL         │   Fine-tune with SSL backbone
│  (train_rfdetr.py --ssl-backbone)│   on labeled RSNA data
├──────────────────────────────────┤
│  Phase 3B: RF-DETR Baseline      │   Fine-tune with original DINOv2
│  (train_rfdetr.py)               │   on same labeled data
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Phase 4: Comparison             │   mAP, AP50, AP75 side-by-side
│  (compare_results.py)            │   SSL vs Baseline
└──────────────────────────────────┘
```

## Project Structure

```
├── configs/
│   ├── ssl_pretrain.yaml          # SSL hyperparameters
│   └── finetune_rfdetr.yaml       # RF-DETR fine-tuning hyperparameters
├── src/
│   ├── data/
│   │   ├── prepare_coco.py        # RSNA CSV → COCO JSON converter
│   │   ├── dataset_ssl.py         # PyTorch Dataset for SSL (multi-crop views)
│   │   └── transforms.py          # Medical X-ray augmentations (Albumentations)
│   ├── models/
│   │   └── ssl_dinov2.py          # DINOv2 student-teacher self-distillation
│   ├── utils/
│   │   ├── logger.py              # W&B initialization
│   │   └── metrics.py             # mAP computation & metric tracking
│   ├── train_ssl.py               # Entry: SSL continual pre-training
│   ├── train_rfdetr.py            # Entry: RF-DETR fine-tuning
│   └── compare_results.py         # Entry: Results comparison
├── scripts/
│   └── run_kaggle.sh              # Full pipeline execution script
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quick Start (Kaggle)

### Prerequisites

1. A Kaggle account with GPU quota enabled
2. [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset added to your notebook
3. A [Weights & Biases](https://wandb.ai) account (free tier is sufficient)

### Step 1: Setup Kaggle Notebook

1. Create a new Kaggle Notebook with **GPU T4 x2** accelerator
2. Add the RSNA Pneumonia Detection Challenge dataset as input
3. Add this GitHub repository as a utility script:
   - Click **"Add data"** → **"Add utility script"** → paste your repo URL

   Or clone manually in a notebook cell:
   ```python
   !git clone https://github.com/ariefff666/ssl-rfdetr-pneumonia.git /kaggle/working/ssl-rfdetr-pneumonia
   ```

### Step 2: Set W&B API Key

```python
import os
os.environ["WANDB_API_KEY"] = "your-wandb-api-key"  # Get from wandb.ai/authorize
```

### Step 3: Run the Full Pipeline

**Option A — Interactive (see output in real-time):**
```python
!cd /kaggle/working/ssl-rfdetr-pneumonia && bash scripts/run_kaggle.sh
```

**Option B — Background (keeps running after closing browser):**
```python
!cd /kaggle/working/ssl-rfdetr-pneumonia && nohup bash scripts/run_kaggle.sh > /kaggle/working/pipeline.log 2>&1 &
!echo "Pipeline started! PID: $(cat /proc/sys/kernel/ns_last_pid)"
```

Monitor progress:
```python
!tail -f /kaggle/working/pipeline.log  # Live log
```

### Step 4: Run Phases Individually (if needed)

```bash
# Phase 1: Prepare COCO dataset
python -m src.data.prepare_coco --config configs/finetune_rfdetr.yaml

# Phase 2: SSL pre-training (multi-GPU)
torchrun --nproc_per_node=2 -m src.train_ssl --config configs/ssl_pretrain.yaml

# Phase 3A: Fine-tune WITH SSL backbone
python -m src.train_rfdetr --config configs/finetune_rfdetr.yaml \
    --ssl-backbone /kaggle/working/checkpoints/ssl/backbone_epoch_50.pth \
    --run-name rfdetr-with-ssl

# Phase 3B: Fine-tune BASELINE (no SSL)
python -m src.train_rfdetr --config configs/finetune_rfdetr.yaml \
    --run-name rfdetr-baseline

# Phase 4: Compare results
python -m src.compare_results \
    --ssl-dir /kaggle/working/checkpoints/rfdetr/rfdetr-with-ssl-ssl \
    --baseline-dir /kaggle/working/checkpoints/rfdetr/rfdetr-baseline-baseline \
    --output-dir /kaggle/working/comparison
```

## Configuration

All hyperparameters are in YAML config files under `configs/`. Key parameters:

| Parameter | SSL Config | RF-DETR Config |
|-----------|-----------|----------------|
| Epochs | 50 | 50 |
| Batch size (per GPU) | 64 | 4 |
| Gradient accumulation | — | 4 (effective: 16) |
| Learning rate | 3e-4 | 1e-4 |
| Image size | 512×512 | 512×512 (RF-DETR-S) |
| Multi-GPU | torchrun DDP | rfdetr built-in |
| W&B logging | ✅ | ✅ |

## Dataset

**RSNA Pneumonia Detection Challenge**
- 26,684 chest X-ray images (1024×1024 pixels)
- 9,555 pneumonia bounding box annotations
- Split: 80% train / 10% valid / 10% test (stratified)
- [Kaggle Source](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

## Technology Stack

- **Model**: [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow Detection Transformer)
- **Backbone**: [DINOv2](https://github.com/facebookresearch/dinov2) (ViT-S/14 with registers)
- **SSL Method**: Self-distillation (student-teacher with EMA)
- **Training**: PyTorch + torchrun (DDP) + Mixed Precision (FP16)
- **Augmentation**: [Albumentations](https://albumentations.ai/)
- **Tracking**: [Weights & Biases](https://wandb.ai/)
- **Platform**: Kaggle (T4 x2 GPU)

## License

MIT
