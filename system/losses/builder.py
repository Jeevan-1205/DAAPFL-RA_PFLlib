import torch.nn as nn

from .dice import DiceLoss
from .focal import FocalLoss
from .hybrid import HybridDiceFocalLoss


def build_loss(
    name="dice_focal",
    num_classes=5,
    **kwargs,
):
    name = name.lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if name == "dice":
        return DiceLoss(**kwargs)

    if name == "focal":
        return FocalLoss(**kwargs)

    if name == "dice_focal":
        return HybridDiceFocalLoss(**kwargs)

    raise ValueError(f"Unknown loss: {name}")