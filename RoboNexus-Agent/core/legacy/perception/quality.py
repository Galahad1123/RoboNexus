"""Legacy temporal mask quality checks for perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class QualityConfig:
    min_area_ratio: float = 0.0005
    max_area_ratio: float = 0.70
    min_temporal_iou: float = 0.15
    max_centroid_jump_ratio: float = 0.25
    min_score: float = 0.45
    consecutive_failures: int = 2


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def evaluate_mask(
    mask: np.ndarray,
    previous: np.ndarray | None,
    tracker_confidence: float,
    config: QualityConfig,
) -> tuple[float, dict[str, float]]:
    height, width = mask.shape
    area_ratio = float(mask.mean())
    area_score = (
        1.0
        if config.min_area_ratio <= area_ratio <= config.max_area_ratio
        else 0.0
    )
    temporal_iou = 1.0 if previous is None else mask_iou(mask, previous)
    motion_score = 1.0
    if previous is not None and mask.any() and previous.any():
        cy, cx = np.argwhere(mask).mean(axis=0)
        py, px = np.argwhere(previous).mean(axis=0)
        jump = float(np.hypot(cx - px, cy - py) / np.hypot(width, height))
        motion_score = max(0.0, 1.0 - jump / config.max_centroid_jump_ratio)
    temporal_score = min(1.0, temporal_iou / config.min_temporal_iou)
    score = (
        0.35 * float(np.clip(tracker_confidence, 0, 1))
        + 0.25 * temporal_score
        + 0.20 * motion_score
        + 0.20 * area_score
    )
    return score, {
        "tracker_confidence": float(tracker_confidence),
        "area_ratio": area_ratio,
        "temporal_iou": temporal_iou,
        "motion_score": motion_score,
    }
