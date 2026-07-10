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
        ignore_index=-100,
    ):
        super().__init__()

        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32)

        print(f"[Focal] gamma={gamma}, alpha={alpha}, ignore_index={ignore_index}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, target):

        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        valid = target != self.ignore_index

        pt = torch.exp(-ce)

        focal = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:

            alpha = self.alpha.to(logits.device)

            safe_target = target.clone()
            safe_target[~valid] = 0
            weight = alpha[safe_target]

            focal = weight * focal

        # zero out ignored pixels before reducing, since ce is 0 there
        # but we don't want them counted in a mean over all pixels
        focal = focal * valid.float()

        if self.reduction == "mean":
            denom = valid.float().sum().clamp_min(1.0)
            return focal.sum() / denom

        if self.reduction == "sum":
            return focal.sum()

        return focal