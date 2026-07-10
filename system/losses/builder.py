import torch
import torch.nn as nn

from .dice import DiceLoss
from .focal import FocalLoss
from .hybrid import HybridDiceFocalLoss


def build_loss(
    name="dice_focal",
    num_classes=5,
    alpha=None,
    include_background=False,
    **kwargs,
):
    name = name.lower()

    if alpha is None:
        print("[build_loss] WARNING: no class weights (alpha) provided — "
              "rare classes will be unweighted in the loss.")

    if name == "cross_entropy":
        weight = None
        if alpha is not None:
            weight = torch.as_tensor(alpha, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)

    if name == "dice":
        return DiceLoss(include_background=include_background, **kwargs)

    if name == "focal":
        return FocalLoss(alpha=alpha, **kwargs)

    if name == "dice_focal":
        return HybridDiceFocalLoss(
            alpha=alpha,
            include_background=include_background,
            **kwargs,
        )

    raise ValueError(f"Unknown loss: {name}")