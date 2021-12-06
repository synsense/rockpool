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
    """
    Instantiate an LIF module with Bitshift decay.

    Args:
        shape (tuple): Either a single dimension ``(Nout,)``, which defines a feed-forward layer of LIF modules with equal amounts of synapses and neurons, or two dimensions ``(Nin, Nout)``, which defines a layer of ``Nin`` synapses and ``Nout`` LIF neurons.
        tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
        tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
        has_bias (bool): When ``True`` the module provides a trainable bias. Default: ``True``
        bias (Optional[FloatVector]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, ``0.0`` will be used by default.
        threshold (FloatVector): An optional array specifying the firing threshold of each neuron. If not provided, ``0.`` will be used by default.
        has_rec (bool): When ``True`` the module provides a trainable recurrent weight matrix. Default ``False``, module is feed-forward.
        w_rec (torch.Tensor): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a matrix with shape ``(Nout, Nin)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
        noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: ``0.0``
        spike_generation_fn (Callable): Function to call for spike production. Usually simple threshold crossing. Implements the suroogate gradient function in the backward call. (StepPWL or PeriodicExponential).
        learning_window (float): Cutoff value for the surrogate gradient.
        weight_init_func (Optional[Callable[[Tuple], torch.tensor]): The initialisation function to use when generating weights. Default: ``None`` (Kaiming initialisation)
        dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
        device: Defines the device on which the model will be processed.
    """

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
