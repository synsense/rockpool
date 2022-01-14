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
import torch.functional as F
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        ----------
        data: Tensor
            Data takes the shape of (batch, time_steps, n_synapses)

        Returns
        -------
        out: Tensor
            Out of spikes with the shape (batch, time_steps, n_neurons)

        """
        (n_batches, time_steps, n_connections) = data.shape
        if n_connections != self.size_in:
            raise ValueError(
                "Input has wrong neuron dimension. It is {}, must be {}".format(
                    n_connections, self.size_in
                )
            )

        data = data.reshape(n_batches, time_steps, self.n_neurons, self.n_synapses)

        # - Replicate states out by batches
        vmem = torch.ones(n_batches, self.n_neurons).to(data.device) * self.vmem
        isyn = (
            torch.ones(n_batches, self.n_neurons, self.n_synapses).to(data.device)
            * self.isyn
        )
        bias = torch.ones(n_batches, self.n_neurons).to(data.device) * self.bias
        spikes = torch.zeros(n_batches, self.n_neurons).to(data.device) * self.spikes

        # - Set up state record and output
        if self._record:
            self._record_Vmem = torch.zeros(n_batches, time_steps, self.n_neurons)
            self._record_Isyn = torch.zeros(
                n_batches, time_steps, self.n_neurons, self.n_synapses
            )

        self._record_spikes = torch.zeros(
            n_batches, time_steps, self.n_neurons, device=data.device
        )

        # - Calculate and cache updated values for decay factors
        alpha = self.alpha
        beta = self.beta

        # - Loop over time
        for t in range(time_steps):

            # Integrate synaptic input
            isyn = isyn + data[:, t]

            # Decay synaptic and membrane state
            vmem *= alpha
            isyn *= beta

            # - Apply spikes over the recurrent weights
            if hasattr(self, "w_rec"):
                rec_inp = F.linear(spikes, self.w_rec.T).reshape(
                    n_batches, self.n_neurons, self.n_synapses
                )
                isyn = isyn + rec_inp

            # Integrate membrane state and apply noise
            if self.noise_std > 0:
                vmem = (
                    vmem
                    + isyn.sum(2)
                    + bias
                    + torch.randn(vmem.shape, device=vmem.device) * self.noise_std
                )
            else:
                vmem = vmem + isyn.sum(2) + bias

            # - Spike generation
            spikes = self.spike_generation_fn(
                vmem, self.threshold, self.learning_window
            )

            # - Membrane reset
            vmem = vmem - spikes * self.threshold

            if self._record:
                # recording
                self._record_Vmem[:, t] = vmem
                self._record_Isyn[:, t] = isyn

            self._record_spikes[:, t] = spikes

        self.vmem = vmem[0].detach()
        self.isyn = isyn[0].detach()
        self.spikes = spikes[0].detach()

        return self._record_spikes
