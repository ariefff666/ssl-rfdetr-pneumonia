"""
metrics.py — Evaluation metric computation helpers.

Provides wrappers around pycocotools for mAP calculation,
plus lightweight metric tracking utilities.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MetricTracker:
    """
    Track running averages of metrics across batches/epochs.

    Usage:
        tracker = MetricTracker()
        tracker.update("loss", 0.5)
        tracker.update("loss", 0.3)
        print(tracker.average("loss"))  # 0.4
    """

    _values: dict[str, list[float]] = field(default_factory=dict)

    def update(self, name: str, value: float, count: int = 1) -> None:
        """Add a metric value (optionally weighted by count)."""
        if name not in self._values:
            self._values[name] = []
        self._values[name].extend([value] * count)

    def average(self, name: str) -> float:
        """Get the running average of a metric."""
        values = self._values.get(name, [])
        return float(np.mean(values)) if values else 0.0

    def latest(self, name: str) -> float:
        """Get the most recent value of a metric."""
        values = self._values.get(name, [])
        return values[-1] if values else 0.0

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._values.clear()

    def summary(self) -> dict[str, float]:
        """Return averages of all tracked metrics."""
        return {name: self.average(name) for name in self._values}


def compute_coco_metrics(coco_gt, coco_dt) -> dict[str, float]:
    """
    Compute COCO detection metrics (mAP, AP50, AP75, etc.).

    Args:
        coco_gt: Ground truth COCO object (pycocotools.coco.COCO).
        coco_dt: Detection results COCO object.

    Returns:
        Dictionary of metric names to values.
    """
    from pycocotools.cocoeval import COCOeval

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "mAP": coco_eval.stats[0],          # AP @ IoU=0.50:0.95
        "AP50": coco_eval.stats[1],          # AP @ IoU=0.50
        "AP75": coco_eval.stats[2],          # AP @ IoU=0.75
        "AP_small": coco_eval.stats[3],      # AP for small objects
        "AP_medium": coco_eval.stats[4],     # AP for medium objects
        "AP_large": coco_eval.stats[5],      # AP for large objects
        "AR_max1": coco_eval.stats[6],       # AR given 1 detection per image
        "AR_max10": coco_eval.stats[7],      # AR given 10 detections per image
        "AR_max100": coco_eval.stats[8],     # AR given 100 detections per image
    }

    return metrics
