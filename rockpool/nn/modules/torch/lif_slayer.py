"""
Implement a LIF Module with Slayer backend 
"""

from importlib import util

# if util.find_spec("sinabs.slayer") is None:
#    raise ModuleNotFoundWarning(
#        "'Sinabs-Slayer' backend not found. Modules that rely on Sinabs-Slayer will not be available."
#    )

from typing import Union, List, Tuple
import numpy as np

from rockpool.nn.modules import LIFTorch
from rockpool.nn.modules.torch.linear_torch import LinearTorch

import torch
from sinabs.slayer.layers import LIF
from sinabs.layers import ExpLeak

import rockpool.parameters as rp

from typing import Tuple, Any

from rockpool.typehints import FloatVector, P_int, P_float, P_tensor, P_bool, P_str

__all__ = ["LIFSlayer"]


class LIFSlayer(LIFTorch):
    def __init__(
        self,
        shape: tuple,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        threshold: FloatVector = 0.0,
        learning_window: P_float = 1.0,
        scale_grads: P_float = 1.0,
        dt: P_float = 0.001,
        device: P_str = "cuda",
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        n_neurons: int
            number of neurons
        n_synapses: int
            number of synapses
        tau_mem: float / (n_neurons)
            decay time constant in units of simulation time steps
        tau_syn: float / (n_synapses, n_neurons)
            decay time constant in units of simulation time steps
        threshold: float
            Spiking threshold
        learning_window: float
            Learning window around spike threshold for surrogate gradient calculation
        dt: float
            Resolution of the simulation in seconds.
        """
        # Initialize class variables

        super().__init__(
            shape=shape,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            threshold=threshold,
            learning_window=learning_window,
            dt=dt,
            device=device,
            *args,
            **kwargs,
        )

        self.scale_grads: P_float = rp.SimulationParameter(scale_grads)
        self.membrane_subtract: P_bool = rp.SimulationParameter(self.threshold)

        alpha = torch.exp(-self.dt / self.tau_syn).to(device)
        self._exp_decay = ExpLeak(alpha=alpha).to(device)

        self._lif_slayer = LIF(
            tau_mem=(self.tau_mem / self.dt).float(),
            threshold=float(self.threshold[0]),
            threshold_low=None,
            membrane_subtract=float(self.membrane_subtract[0]),
            window=self.learning_window,
            scale_grads=self.scale_grads,
        ).to(device)

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:

        self._record = record

        output_data, _, _ = super().evolve(input_data, record)

        # states = {
        #    "Isyn": self.isyn,
        #    "Vmem": self.vmem,
        # }

        # record_dict = (
        #    {
        #        "Isyn": self._record_Isyn,
        #        "Vmem": self._record_Vmem,
        #    }
        #    if record
        #    else {}
        # )

        return output_data, _, _

    def detach(self):
        """
        Detach gradients of the state variables
        Returns
        -------

        """
        self.vmem = self.vmem.detach()
        self.isyn = self.isyn.detach()
        if hasattr(self, "n_spikes_out"):
            self.n_spikes_out = self.n_spikes_out.detach()

    def forward(self, data: torch.Tensor):
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics dynamics Leaky Integrate and Fire dynamics

        Parameters
        ----------
        data: Tensor
            Data takes the shape of (time_steps, batch, n_synapses, n_neurons)

        Returns
        -------
        out: Tensor
            Tensor of spikes with the shape (time_steps, batch, n_neurons)

        """
        (n_batches, time_steps, n_connections) = data.shape
        # if n_connections != self.size_in:
        #    raise ValueError(
        #        "Input has wrong neuron dimension. It is {}, must be {}".format(
        #            self.size_in, self.size_out
        #        )
        #    )

        # data = data.reshape(n_batches, time_steps, self.n_synapses, self.n_neurons)

        out_spikes = self._exp_decay(data)
        out_spikes = self._lif_slayer(out_spikes.sum(-1))

        return out_spikes
