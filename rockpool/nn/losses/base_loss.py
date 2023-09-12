"""
    Loss modules are implemented analogously to losses in PyTorch. All losses inherit from a common Loss class.
    All loss classes need to implement a forward method which returns a scalar.
"""

from torch import nn

__all__ = ["_Loss"]


class _Loss(nn.Module):
    reduction: str

    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> float:
        pass
