"""Config-driven loss builder."""
from __future__ import annotations

from .focal import FocalLoss
from .dice import DiceLoss
from .hybrid import HybridDiceFocalLoss


def build_loss(cfg):
    lc = cfg.loss

    alpha = lc.get("class_weights", None)
    ignore = int(lc.get("ignore_index", -100))
    name = lc.name.lower()

    if name == "focal":
        return FocalLoss(
            gamma=float(lc.focal_gamma),
            alpha=alpha,
            ignore_index=ignore,
        )

    if name == "dice":
        return DiceLoss(
            smooth=float(lc.dice_smooth),
            ignore_index=ignore,
        )

    if name == "hybrid":
        return HybridDiceFocalLoss(
            dice_weight=float(lc.hybrid_dice_weight),
            focal_weight=float(lc.hybrid_focal_weight),
            gamma=float(lc.focal_gamma),
            alpha=alpha,
            dice_smooth=float(lc.dice_smooth),
            ignore_index=ignore,
        )

    raise ValueError(f"Unknown loss '{name}'")