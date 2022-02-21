""""
Implements an exponential low-pass synapse using Torch backend
"""
import warnings

import torch
import numpy as np
from typing import Union, List, Tuple
from rockpool.nn.modules.torch.torch_module import TorchModule

# Data shape convention (time, batch, synapses, neurons)

__all__ = ["LowPass"]


class LowPass(TorchModule):
    def __init__(
        self, n_neurons: int, dt: float, tau_mem: Union[float, List], *args, **kwargs
    ):
        """

        Parameters
        ----------
        n_neurons: int
            number of neurons
        dt: float
            resolution of the simulation
        tau_mem: float / (n_neurons)
            decay time constant in units of simulation time steps
        """
        warnings.warn(
            DeprecationWarning("`LowPass` is deprecated. Use `ExpSynTorch` instead.")
        )

        super().__init__(*args, **kwargs)

        # Initialize class variables
        self.n_neurons = n_neurons
        self.tau_mem = np.array(tau_mem)
        self.dt = dt

        # Initialize states
        self.register_buffer("vmem", torch.zeros((1, self.n_neurons)))

        # Calculate decay rates
        self.register_buffer(
            "alpha_mem", torch.Tensor([np.exp(-self.dt / self.tau_mem)])
        )

    def init_states(self, batch_size: int = 1):
        """
        Initialize the state variables with a given batch size

        Parameters
        ----------
        batch_size: int
            Number of batches presented

        Returns
        -------

        """
        self.vmem.data = torch.zeros(
            (batch_size, self.n_neurons), device=self.vmem.device
        )

    def detach(self):
        """
        Detach gradients of the state variables
        Returns
        -------

        """
        self.vmem = self.vmem.detach()

    def forward(self, data: torch.Tensor):
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics dynamics Leaky Integrate and Fire dynamics

        Parameters
        ----------
        data: Tensor
            Data takes the shape of (time_steps, batch, n_neurons)

        Returns
        -------
        out: Tensor
            Tensor of spikes with the shape (time_steps, batch, n_neurons)

        """
        (time_steps, n_batches, n_neurons) = data.shape
        vmem = self.vmem
        alpha_mem = self.alpha_mem
        out_states = torch.zeros((time_steps, n_batches, n_neurons), device=data.device)

        for t in range(time_steps):
            # Leak
            vmem = vmem * alpha_mem

            # State propagatoin
            vmem = vmem + data[t]  # shape (batch, neuron)

            # Spike generation
            out_states[t] = vmem

        self.vmem = vmem

        return out_states

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        return input_shape
