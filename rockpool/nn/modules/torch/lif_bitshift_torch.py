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
    def __init__(
        self,
        tau_mem: Optional[Union[FloatVector, P_float]] = None,
        tau_syn: Optional[Union[FloatVector, P_float]] = None,
        dt: P_float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Args:
            tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 10ms will be used by default.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
        """

        # - Initialise superclass
        super().__init__(
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            dt=dt,
            *args, 
            **kwargs,
        )

        #dash_mem = calc_bitshift_decay(self.tau_mem, self.dt)
        #dash_syn = calc_bitshift_decay(self.tau_syn, self.dt)

        ## make sure the tau mem and tau syn are representable by bitshift decay
        #self.tau_mem.data = inv_calc_bitshift_decay(dash_mem, self.dt) 
        #self.tau_syn.data = inv_calc_bitshift_decay(dash_syn, self.dt) 

        alpha = self.alpha
        beta = self.beta


        self.tau_mem.data = -dt / torch.log(alpha)
        self.tau_syn.data = -dt / torch.log(beta)


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
