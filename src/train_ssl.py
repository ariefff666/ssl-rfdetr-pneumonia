"""
train_ssl.py — Entry point for DINOv2 Domain-Adaptive Continual Pre-training.

This script trains the DINOv2 backbone using self-distillation (student-teacher)
on ALL RSNA chest X-ray images WITHOUT using any labels. The resulting backbone
weights capture medical-domain visual features.

Output:
    checkpoints/ssl/backbone_epoch_{N}.pth — Adapted DINOv2 backbone weights
    (to be loaded into RF-DETR for downstream pneumonia detection)

Usage (single GPU):
    python -m src.train_ssl --config configs/ssl_pretrain.yaml

Usage (multi-GPU via torchrun on Kaggle T4x2):
    torchrun --nproc_per_node=2 -m src.train_ssl --config configs/ssl_pretrain.yaml
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

from src.data.dataset_ssl import SSLChestXrayDataset
from src.models.ssl_dinov2 import SSLDINOv2
from src.utils.logger import init_wandb, log_metrics, finish_wandb
from src.utils.metrics import MetricTracker


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    warmup_epochs: int = 0,
    warmup_start_value: float = 0.0,
) -> list[float]:
    """Create a cosine annealing schedule with linear warmup."""
    warmup_schedule = []
    if warmup_epochs > 0:
        warmup_schedule = [
            warmup_start_value + (base_value - warmup_start_value) * i / warmup_epochs
            for i in range(warmup_epochs)
        ]

    cos_epochs = epochs - warmup_epochs
    cosine_schedule = [
        final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / cos_epochs))
        for i in range(cos_epochs)
    ]

    return warmup_schedule + cosine_schedule


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed processes."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def main(config_path: str) -> None:
    """Main SSL pre-training loop."""
    cfg = load_config(config_path)
    rank, local_rank, world_size = setup_distributed()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print("=" * 70)
        print("DINOv2 Domain-Adaptive Continual Pre-training")
        print(f"Device: {device} | World size: {world_size}")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # Dataset & DataLoader
    # -------------------------------------------------------------------------
    dataset = SSLChestXrayDataset(
        image_dir=cfg["data"]["image_dir"],
        image_size=cfg["data"]["image_size"],
        num_local_crops=cfg["augmentation"]["num_local_crops"],
        config=cfg["augmentation"],
    )

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    if is_main_process(rank):
        print(f"Dataset: {len(dataset)} images")
        print(f"Batches per epoch: {len(dataloader)}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = SSLDINOv2(
        backbone_name=cfg["model"]["backbone"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)

    if world_size > 1:
        # Only wrap student parts in DDP (teacher is updated via EMA, not gradients)
        model.student_backbone = DDP(
            model.student_backbone, device_ids=[local_rank]
        )
        model.student_head = DDP(
            model.student_head, device_ids=[local_rank]
        )

    # -------------------------------------------------------------------------
    # Optimizer & Schedulers
    # -------------------------------------------------------------------------
    train_cfg = cfg["training"]
    params = list(model.student_backbone.parameters()) + list(model.student_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    lr_schedule = cosine_scheduler(
        base_value=train_cfg["learning_rate"],
        final_value=train_cfg["min_lr"],
        epochs=train_cfg["epochs"],
        warmup_epochs=train_cfg["warmup_epochs"],
    )

    momentum_schedule = cosine_scheduler(
        base_value=train_cfg["momentum_teacher"],
        final_value=1.0,
        epochs=train_cfg["epochs"],
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda")

    # -------------------------------------------------------------------------
    # Checkpointing — resume if specified
    # -------------------------------------------------------------------------
    ckpt_cfg = cfg["checkpoint"]
    save_dir = Path(ckpt_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    if ckpt_cfg.get("resume_from"):
        ckpt = torch.load(ckpt_cfg["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if is_main_process(rank):
            print(f"Resumed from checkpoint: epoch {start_epoch}")

    # -------------------------------------------------------------------------
    # W&B Logging
    # -------------------------------------------------------------------------
    if is_main_process(rank) and cfg["logging"]["use_wandb"]:
        init_wandb(
            project=cfg["logging"]["wandb_project"],
            run_name=cfg["logging"]["wandb_run_name"],
            config=cfg,
            tags=["ssl", "dinov2", "continual-pretraining", "chest-xray"],
        )

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    log_every = cfg["logging"]["log_every_steps"]
    grad_clip = train_cfg["gradient_clip_norm"]

    for epoch in range(start_epoch, train_cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)

        tracker = MetricTracker()
        epoch_start = time.time()

        # Set learning rate for this epoch
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[epoch]

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}",
            disable=not is_main_process(rank),
        )

        for step, batch in enumerate(progress):
            global_step = epoch * len(dataloader) + step

            # Move data to device
            batch = {
                "global_1": batch["global_1"].to(device, non_blocking=True),
                "global_2": batch["global_2"].to(device, non_blocking=True),
                "local_crops": batch["local_crops"].to(device, non_blocking=True),
            }

            # Forward + backward with mixed precision
            with torch.amp.autocast("cuda"):
                loss = model.compute_loss(batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Update teacher EMA
            model.update_teacher(momentum=momentum_schedule[epoch])

            # Track metrics
            tracker.update("loss", loss.item())
            progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_schedule[epoch]:.6f}"})

            # Log to W&B periodically
            if is_main_process(rank) and (global_step + 1) % log_every == 0:
                log_metrics({
                    "ssl/loss": loss.item(),
                    "ssl/learning_rate": lr_schedule[epoch],
                    "ssl/teacher_momentum": momentum_schedule[epoch],
                    "ssl/epoch": epoch + 1,
                }, step=global_step)

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_loss = tracker.average("loss")

        if is_main_process(rank):
            print(
                f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | "
                f"LR: {lr_schedule[epoch]:.6f} | Time: {epoch_time:.1f}s"
            )

            log_metrics({
                "ssl/epoch_loss": avg_loss,
                "ssl/epoch_time_sec": epoch_time,
            }, step=(epoch + 1) * len(dataloader))

            # Save checkpoint
            save_every = ckpt_cfg["save_every_epochs"]
            is_last = (epoch + 1) == train_cfg["epochs"]

            if (epoch + 1) % save_every == 0 or is_last:
                # Save full checkpoint (for resume)
                full_ckpt_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                # Handle DDP module unwrapping
                student_bb = model.student_backbone
                if hasattr(student_bb, "module"):
                    student_bb = student_bb.module
                student_hd = model.student_head
                if hasattr(student_hd, "module"):
                    student_hd = student_hd.module

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": {
                        "student_backbone": student_bb.state_dict(),
                        "student_head": student_hd.state_dict(),
                        "teacher_backbone": model.teacher_backbone.state_dict(),
                        "teacher_head": model.teacher_head.state_dict(),
                        "center": model.criterion.center,
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": cfg,
                }, full_ckpt_path)

                # Save standalone backbone weights (for RF-DETR integration)
                backbone_path = save_dir / f"backbone_epoch_{epoch + 1}.pth"
                torch.save(student_bb.state_dict(), backbone_path)

                print(f"  Saved checkpoint: {full_ckpt_path}")
                print(f"  Saved backbone: {backbone_path}")

    # Finish
    if is_main_process(rank):
        finish_wandb()
        print("\nSSL Pre-training complete!")
        print(f"Final backbone saved to: {save_dir / f'backbone_epoch_{train_cfg['epochs']}.pth'}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 SSL Continual Pre-training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ssl_pretrain.yaml",
        help="Path to SSL config YAML",
    )
    args = parser.parse_args()
    main(args.config)
