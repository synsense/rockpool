"""
Implement a LIF Module with bit-shift decay, using a Torch backend
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

from typing import Union, List, Tuple
import numpy as np

from rockpool.nn.modules.torch.lif_torch import LIFTorch, StepPWL, PeriodicExponential

import torch

import rockpool.parameters as rp

from typing import Tuple, Any

from rockpool.typehints import P_int, P_float, P_tensor, FloatVector, P_bool, P_str

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
    bitsh = torch.log2(tau / dt)
    bitsh[bitsh < 0] = 0
    return bitsh


class LIFBitshiftTorch(LIFTorch):
    def __init__(
        self,
        shape: tuple,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        has_bias: P_bool = True,
        bias: FloatVector = 0.0,
        threshold: FloatVector = 0.0,
        has_rec: P_bool = False,
        w_rec: torch.Tensor = None,
        noise_std: P_float = 0.0,
        gradient_fn=StepPWL,
        learning_window: P_float = 1.0,
        dt: P_float = 1e-3,
        device: P_str = "cuda",
        *args,
        **kwargs,
    ):

        """

        Parameters
        ----------
        shape: tuple
            Input and output dimensions. (n_neurons * n_synapses, n_neurons)
        tau_mem: float / (n_neurons)
            Decay time constant in units of simulation time steps
        tau_syn: float / (n_synapses, n_neurons)
            Decay time constant in units of simulation time steps
        threshold: float
            Spiking threshold
        learning_window: float
            Learning window around spike threshold for surrogate gradient calculation
        has_bias: bool
            Bias / current injection to the membrane
        bias: FloatVector
            Inital values for the bias
        dt: float
            Resolution of the simulation in seconds.
        device:
            Device. Either 'cuda' or 'cpu'.
        """

        super().__init__(
            shape=shape,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            has_bias=has_bias,
            bias=bias,
            threshold=threshold,
            has_rec=has_rec,
            w_rec=w_rec,
            noise_std=noise_std,
            gradient_fn=gradient_fn,
            learning_window=learning_window,
            dt=dt,
            device=device,
            *args,
            **kwargs,
        )

        self.dash_mem: P_tensor = rp.SimulationParameter(
            calc_bitshift_decay(self.tau_mem, self.dt).unsqueeze(1).to(device)
        )
        self.dash_syn: P_tensor = rp.SimulationParameter(
            calc_bitshift_decay(self.tau_syn, self.dt).unsqueeze(1).to(device)
        )

        self.bitshift_decay = Bitshift().apply

    def decay_isyn(self, v):
        return self.bitshift_decay(v, self.dash_syn, self.beta)

    def decay_vmem(self, v):
        return self.bitshift_decay(v, self.dash_mem, self.alpha)
