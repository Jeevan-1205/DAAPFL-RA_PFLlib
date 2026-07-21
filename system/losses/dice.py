import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1e-6,
        include_background=True,
        ignore_index=-100
    ):
        super().__init__()

        self.smooth = smooth
        self.ignore_index = ignore_index
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        num_classes = logits.shape[1]

        probs = F.softmax(logits, dim=1)

        valid = target != self.ignore_index


        tgt = target.clone()
        tgt[~valid] = 0

        onehot = F.one_hot(
            tgt,
            num_classes,
        ).permute(0,3,1,2).float()

        mask = valid.unsqueeze(1).float()

        probs = probs * mask
        onehot = onehot * mask

        dims = (0,2,3)

        intersection = (probs * onehot).sum(dims)

        denominator = (
            probs.sum(dims)
            +
            onehot.sum(dims)
        )

        dice = (
            2*intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        # Only compute Dice for classes that are actually present in the GT of
        # this batch (standard present-class masking).  When a class has no GT
        # pixels, there is no meaningful Dice signal and computing it produces a
        # suppressive gradient that pushes the model to predict zero probability
        # for that class — harming rare classes in non-IID federated settings.
        present = onehot.sum(dim=(0, 2, 3)) > 0
        if not self.include_background:
            present[0] = False

        dice = dice[present]

        if dice.numel() == 0:
            return logits.new_tensor(0.)

        return 1 - dice.mean()