"""Semantic segmentation metrics.

Metrics are computed from a confusion matrix whose rows are target classes
and columns are predicted classes.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch


def segmentation_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Return a ``num_classes x num_classes`` segmentation confusion matrix."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}."
        )

    pred = pred.detach().reshape(-1).to(torch.int64)
    target = target.detach().reshape(-1).to(torch.int64)

    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index

    pred = pred[valid]
    target = target[valid]
    valid_pred = (pred >= 0) & (pred < num_classes)
    pred = pred[valid_pred]
    target = target[valid_pred]

    encoded = target * num_classes + pred
    confusion = torch.bincount(
        encoded.cpu(),
        minlength=num_classes * num_classes,
    )
    return confusion.reshape(num_classes, num_classes)


def pixel_accuracy(confusion_matrix: torch.Tensor) -> float:
    """Pixel Accuracy: correctly classified pixels divided by valid pixels."""
    confusion = confusion_matrix.to(torch.float64)
    total = confusion.sum()
    if total.item() == 0:
        return 0.0
    return (torch.diag(confusion).sum() / total).item()


def dice_per_class(confusion_matrix: torch.Tensor) -> np.ndarray:
    """Dice score for each class."""
    confusion = confusion_matrix.to(torch.float64)
    intersection = torch.diag(confusion)
    pred_count = confusion.sum(dim=0)
    target_count = confusion.sum(dim=1)
    denominator = pred_count + target_count
    dice = torch.full_like(intersection, torch.nan)
    valid = denominator > 0
    dice[valid] = (2.0 * intersection[valid]) / denominator[valid]
    return dice.cpu().numpy()


def iou_per_class(confusion_matrix: torch.Tensor) -> np.ndarray:
    """Intersection over Union for each class."""
    confusion = confusion_matrix.to(torch.float64)
    intersection = torch.diag(confusion)
    pred_count = confusion.sum(dim=0)
    target_count = confusion.sum(dim=1)
    union = pred_count + target_count - intersection
    iou = torch.full_like(intersection, torch.nan)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou.cpu().numpy()


def mean_dice(confusion_matrix: torch.Tensor) -> float:
    """Mean Dice over classes present in prediction or target."""
    scores = dice_per_class(confusion_matrix)
    if np.all(np.isnan(scores)):
        return 0.0
    return float(np.nanmean(scores))


def mean_iou(confusion_matrix: torch.Tensor) -> float:
    """Mean IoU over classes present in prediction or target."""
    scores = iou_per_class(confusion_matrix)
    if np.all(np.isnan(scores)):
        return 0.0
    return float(np.nanmean(scores))


def segmentation_metrics(confusion_matrix: torch.Tensor) -> Dict[str, object]:
    """Compute Pixel Accuracy, Dice, IoU, and Mean IoU."""
    dice_scores = dice_per_class(confusion_matrix)
    iou_scores = iou_per_class(confusion_matrix)
    return {
        "pixel_accuracy": pixel_accuracy(confusion_matrix),
        "dice": 0.0 if np.all(np.isnan(dice_scores)) else float(np.nanmean(dice_scores)),
        "dice_per_class": dice_scores,
        "iou": iou_scores,
        "mean_iou": 0.0 if np.all(np.isnan(iou_scores)) else float(np.nanmean(iou_scores)),
    }
