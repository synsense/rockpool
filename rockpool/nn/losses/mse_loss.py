from torch import nn
import inspect
from typing import List
from torch import Tensor

from .base_loss import _Loss

__all__ = ["MSELoss"]


class MSELoss(_Loss):
    """
    This is a wrapper around the PyTorch MSE loss.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.loss = 0

    def log_params(self, parser) -> None:
        params = [
            i
            for i in inspect.getmembers(self)
            if not i[0].startswith("_") and not inspect.ismethod(i[1])
        ]
        for key, value in params:
            parser.add_argument("--" + key, type=type(value), default=value)

    def forward(self, prediction: Tensor, target: Tensor) -> List[Tensor]:
        self.loss = self.mse(prediction, target)
        return self.loss
