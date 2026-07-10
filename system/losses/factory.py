"""Config-driven loss builder."""
from __future__ import annotations

from .focal import FocalLoss
from .dice import DiceLoss
from .hybrid import HybridDiceFocalLoss


def build_loss(cfg):
    lc = cfg.loss

    alpha = lc.get("class_weights", None)
    ignore = int(lc.get("ignore_index", -100))
    include_background = bool(lc.get("include_background", False))
    name = lc.name.lower()

    if alpha is None:
        print("[factory.build_loss] WARNING: no class_weights set in config — "
              "rare classes will be unweighted in the loss.")

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
            include_background=include_background,
        )

    if name == "hybrid":
        return HybridDiceFocalLoss(
            dice_weight=float(lc.hybrid_dice_weight),
            focal_weight=float(lc.hybrid_focal_weight),
            gamma=float(lc.focal_gamma),
            alpha=alpha,
            dice_smooth=float(lc.dice_smooth),
            ignore_index=ignore,
            include_background=include_background,
        )

    raise ValueError(f"Unknown loss '{name}'")