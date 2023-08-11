"""
    Loss modules are implemented analogously to losses in PyTorch. All losses inherit from a common Loss class.
    All loss classes need to implemented a forward method. In contrast to torch losses (where the loss is typically
    only applied to the target neurons), the loss implemented here is composed of two parts: one applied to the
    target neuron (loss_target) and one applied to the non-target neurons (loss_nontarget). The total loss is a
    weighted sum of the two: loss = loss_target + alpha * loss_nontarget
    This helps to silence the non-target neurons since we do not use a softmax at the output of the model.
    Thus the forward methods implements here return a list of three losses: [loss, loss_target, loss_nontarget]
"""

from torch import nn
from typing import List
from torch import Tensor

__all__ = ["_Loss"]


class _Loss(nn.Module):
    reduction: str

    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> List[Tensor]:
        pass
