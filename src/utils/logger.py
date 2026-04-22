"""
logger.py — Weights & Biases (W&B) initialization and logging utilities.
"""

import os
from typing import Any

import wandb


def init_wandb(
    project: str,
    run_name: str,
    config: dict | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> wandb.sdk.wandb_run.Run | None:
    """
    Initialize a W&B run for experiment tracking.

    Args:
        project: W&B project name.
        run_name: Descriptive name for this run.
        config: Hyperparameters dict to log.
        tags: Optional tags for filtering runs.
        notes: Optional description of this run.

    Returns:
        The wandb Run object, or None if WANDB_API_KEY is not set.
    """
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("[W&B] WANDB_API_KEY not found. Logging disabled.")
        print("[W&B] Set it with: export WANDB_API_KEY='your-key-here'")
        return None

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=tags or [],
        notes=notes,
        save_code=True,
    )

    print(f"[W&B] Initialized run: {run.name} ({run.url})")
    return run


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to W&B if a run is active."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_summary(summary: dict[str, Any]) -> None:
    """Log summary metrics (final results) to W&B."""
    if wandb.run is not None:
        for key, value in summary.items():
            wandb.run.summary[key] = value


def finish_wandb() -> None:
    """Cleanly finish the W&B run."""
    if wandb.run is not None:
        wandb.finish()
        print("[W&B] Run finished.")
