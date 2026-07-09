import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        alpha: Tensor/list of class weights or None.
        gamma: Focusing parameter.
    """

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean",
    ):
        super().__init__()

        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32)

        print(f"[Focal] gamma={gamma}, alpha={alpha}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):

        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
        )

        pt = torch.exp(-ce)

        focal = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:

            alpha = self.alpha.to(logits.device)

            weight = alpha[target]

            focal = weight * focal

        if self.reduction == "mean":
            return focal.mean()

        if self.reduction == "sum":
            return focal.sum()

        return focal