"""
Implement a LIF Module with bit-shift decay, using a Torch backend
"""

import torch
from rockpool.nn.modules.torch.lif_torch import LIFTorch
from rockpool.typehints import *

__all__ = ["LIFBitshiftTorch"]


class Bitshift(torch.autograd.Function):
    """
    Subtract from membrane potential on reaching threshold
    """

    @staticmethod
    def forward(ctx, data, dash, tau):
        v = data - (data / (2**dash))
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
    def __init__(
        self,
        *args,
        max_spikes_per_dt: P_int = 31,
        **kwargs,
    ):

        # - Initialise superclass
        super().__init__(
            max_spikes_per_dt=max_spikes_per_dt,
            *args,
            **kwargs,
        )

        ## make sure the tau mem and tau syn are representable by bitshift decay
        alpha = self.alpha
        beta = self.beta

        self.tau_mem.data = -self.dt / torch.log(alpha)
        self.tau_syn.data = -self.dt / torch.log(beta)

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
