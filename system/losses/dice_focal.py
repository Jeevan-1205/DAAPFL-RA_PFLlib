import torch.nn as nn

from .dice import DiceLoss
from .focal import FocalLoss


class DiceFocalLoss(nn.Module):

    def __init__(
        self,
        num_classes,
        dice_weight=1.0,
        focal_weight=1.0,
        gamma=2.0,
        alpha=None,
        include_background=True,
    ):
        super().__init__()

        self.dice = DiceLoss(
            num_classes=num_classes,
            include_background=include_background,
        )

        self.focal = FocalLoss(
            gamma=gamma,
            alpha=alpha,
        )

        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, logits, target):

        dice_loss = self.dice(
            logits,
            target,
        )

        focal_loss = self.focal(
            logits,
            target,
        )

        return (
            self.dw * dice_loss
            +
            self.fw * focal_loss
        )