"""
Implement a LIF Module, using a Slayer backend
"""

from importlib import util

if util.find_spec("sinabs.slayer") is None:
    raise ModuleNotFoundError(
        "'Slayer' backend not found. Modules that rely on Slayer will not be available."
    )

from typing import Union, List, Tuple, Callable, Optional, Any
import numpy as np
from rockpool.nn.modules.torch.lif_torch import LIFBaseTorch 
import torch
import torch.nn.functional as F
import torch.nn.init as init
import rockpool.parameters as rp

from rockpool.typehints import *

from rockpool.graph import (
    GraphModuleBase,
    as_GraphHolder,
    LIFNeuronWithSynsRealValue,
    LinearWeights,
)

from sinabs.slayer.spike import SpikeFunctionIterForward
from sinabs.slayer.leaky import LeakyIntegrator

__all__ = ["LIFSlayer"]



class LIFSlayer(LIFBaseTorch):

    def __init__(
        self,
        tau_mem: P_float = 0.02,
        threshold: P_float = 1.,
        has_bias: bool = False,
        has_rec: bool = False,
        noise_std: P_float = 0.0,
        device: P_str = None,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module using the Slayer backend

        Args:
            tau_mem (float): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            threshold (float): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            has_bias (bool): Must be False
            has_rec (bool): Must be False
            noise_std (float): Must be 0
            device (str): Must be cuda 
        """

        assert device == "cuda"
        assert isinstance(tau_mem, float)
        assert isinstance(threshold, float)
        assert has_bias == False
        assert has_rec == False
        assert noise_std == 0.

        # - Initialise superclass
        super().__init__(
            tau_mem=tau_mem,
            threshold=threshold,
            has_bias=has_bias,
            has_rec=has_rec,
            noise_std=noise_std,
            device=device,
            *args, **kwargs,
        )

    
    def forward_leak(self, inp: torch.Tensor, alpha, state):

        out_state = LeakyIntegrator.apply(
            inp, state.flatten().contiguous(), alpha
        )
    
        return out_state 


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
                    self.size_in, self.size_out
                )
            )


        # - Replicate states out by batches
        vmem = torch.ones(n_batches, self.n_neurons).to(data.device) * self.vmem
        vmem = vmem.squeeze().repeat(1, time_steps)

        isyn = (
            torch.ones(n_batches, self.n_synapses, self.n_neurons).to(data.device)
            * self.isyn
        )

        bias = torch.ones(n_batches, self.n_neurons).to(data.device) * self.bias
        spikes = torch.zeros(n_batches, self.n_neurons).to(data.device) * self.spikes
        spikes = spikes.squeeze()

        # - Set up state record and output
        if self._record:
            self._record_Vmem = torch.zeros(n_batches, time_steps, self.n_neurons)
            self._record_Isyn = torch.zeros(
                n_batches, time_steps, self.n_synapses, self.n_neurons
            )

        self._record_spikes = torch.zeros(
            n_batches, time_steps, self.n_neurons, device=data.device
        )


        # leak
        data = data.reshape(n_batches, time_steps, self.n_synapses, self.n_neurons)
#        .movedim(-1, 1).reshape(n_batches * n_connections, time_steps)
        inp = torch.zeros(n_batches * self.n_neurons, self.n_synapses, time_steps).to(data.device)

        for syn in range(self.n_synapses):
            inp[:, syn] = self.forward_leak(data[:, :, syn].movedim(-1, 1).reshape(n_batches * self.n_neurons, time_steps), 
                                            self.beta[syn][0], 
                                            isyn[:, syn])

        output_spikes, vmem = SpikeFunctionIterForward.apply(
                inp.sum(1), # input
                self.threshold[0], #membrane subtract
                self.alpha[0].item(), # alpha
                vmem, # init state
                spikes, # last activations
                self.threshold[0], # threshold
                None, # threshold low
                self.learning_window, #learning window
                1.0, #scale grads
                )
        
        vmem = vmem.reshape(n_batches, self.n_neurons, time_steps).movedim(1, -1)
        spikes = output_spikes.reshape(n_batches, self.n_neurons, time_steps).movedim(1, -1)

        vmem = vmem - spikes * self.threshold[0]
        inp = inp.reshape(n_batches, self.n_neurons, self.n_synapses, time_steps).movedim(1, 2).movedim(-1, 1)

        if self._record:
            # recording
            self._record_Vmem = vmem.detach()
            self._record_Isyn = inp.detach()

        self._record_spikes = spikes

        self.vmem = vmem[0, -1].detach()
        self.isyn = inp[0, -1].detach()
        self.spikes = spikes[0, -1].detach()

        self._record_spikes

        return self._record_spikes

