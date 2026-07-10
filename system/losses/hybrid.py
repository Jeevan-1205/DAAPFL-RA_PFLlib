"""Hybrid Dice + Focal loss."""
from __future__ import annotations
from typing import Optional, Sequence
import torch
import torch.nn as nn
from .focal import FocalLoss
from .dice import DiceLoss


class HybridDiceFocalLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5,
                 gamma: float = 2.0, alpha: Optional[Sequence[float]] = None,
                 dice_smooth: float = 1.0, ignore_index: int = -100,
                 include_background: bool = False):
        super().__init__()
        print(f"[Hybrid] gamma={gamma}, alpha={alpha}, include_background={include_background}")
        self.dw, self.fw = dice_weight, focal_weight
        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
        )
        self.dice = DiceLoss(
            smooth=dice_smooth,
            include_background=include_background,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(logits, target) + self.dw * self.dice(logits, target)