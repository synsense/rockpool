"""
Implement a LIF Module with bit-shift decay, using a Torch backend
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

from typing import Union, List, Tuple, Callable, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from rockpool.nn.modules.torch.lif_torch import LIFTorch, StepPWL, PeriodicExponential
import rockpool.parameters as rp
from rockpool.typehints import *

__all__ = ["LIFBitshiftTorch"]


class Bitshift(torch.autograd.Function):
    """
    Subtract from membrane potential on reaching threshold
    """

    @staticmethod
    def forward(ctx, data, dash, tau):
        v = data - (data / (2 ** dash))
        ctx.save_for_backward(tau)

        return v

    @staticmethod
    def backward(ctx, grad_output):
        [tau] = ctx.saved_tensors
        grad_input = grad_output * tau

        return grad_input, None, None


# helper functions
def calc_bitshift_decay(tau, dt):
    bitsh = torch.round(torch.log2(tau / dt)).int()
    bitsh[bitsh < 0] = 0
    return bitsh


def inv_calc_bitshift_decay(dash, dt):
    return dt * torch.exp2(dash)


class LIFBitshiftTorch(LIFTorch):
    @property
    def alpha(self):
        return 1 - 1 / (
            2 ** calc_bitshift_decay(self.tau_mem, self.dt).to(self.tau_mem.device)
        )

    @property
    def beta(self):
        return 1 - 1 / (
            2 ** calc_bitshift_decay(self.tau_syn, self.dt).to(self.tau_syn.device)
        )
