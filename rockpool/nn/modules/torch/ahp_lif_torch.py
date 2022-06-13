"""
Implement a modified version of LIF Module (ahp, after hyperpolarization feedback,  is added)), using a Torch backend
"""

from tkinter import E
from typing import Union, Tuple, Callable, Optional, Any
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
# from rockpool.nn.modules.torch.torch_module import LIFBaseTorch

from rockpool.nn.modules.torch.lif_torch import StepPWL, PeriodicExponential, sigmoid, LIFBaseTorch

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

__all__ = ["aLIFTorch"]

class aLIFTorch(LIFBaseTorch):
    def __init__(
        self,
        shape: tuple,
        tau_ahp: Optional[Union[FloatVector, P_float]] = None,
        w_ahp: torch.Tensor = None,
        weight_init_func: Optional[
            Callable[[Tuple], torch.tensor]
        ] = lambda s: init.kaiming_uniform_(torch.empty(s)),
        *args,
        **kwargs,
    ):

        """
        A variant of leaky integrate-and-fire spiking neuron model with a Torch backend
        It is built based on LIFTorch with an added inhibitory recurrent connection called ahp (after hyperpolarization) feedback. This connection includes wahp and tau_ahp which currently are set to a constant negative scalar and traianble vectors, respectively. The role of this feedback is to pull down the membrane voltage and reduce the firing rate 

        This module implements the update equations:

        .. math ::

            I_{ahp} += S_{ahp} \\cdot W_{ahp}
            I_{syn} += S_{in}(t) + S_{rec} \\cdot W_{rec}

            I_{ahp} *= \exp(-dt / \tau_{ahp})
            I_{syn} *= \exp(-dt / \tau_{syn})

            I_{syn} +=  I_{ahp}

            V_{mem} *= \exp(-dt / \tau_{mem})
            V_{mem} += I_{syn} + b + \sigma \zeta(t)

        where :math:`S_{in}(t)` is a vector containing ``1`` (or a weighed spike) for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a Wiener noise process with standard deviation :math:`\\sigma` after 1s; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively. :math:`S_{rec}(t)` is a vector containing ``1`` for each neuron that emitted a spike in the last time-step. :math:`W_{rec}` is a recurrent weight matrix, if recurrent weights are used. :math:`b` is an optional bias current per neuron (default 0.).
        and :math `S_{ahp}(t)` is a vector containing ``1`` for each neuron that emitted a spike in the last time-step. :math:`W_{ahp}` is a  weight vector coresponding to inhibitory recurrent self-connections, if ahp mode is used are used. \tau_{ahp} is the time constant of the ahp current

        :On spiking:

        When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr}`, then the neuron emits a spike. The spiking neuron subtracts its own threshold on reset.

        .. math ::

            V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

            V_{mem, j} = V_{mem, j} - V_{thr}

        Neurons therefore share a common resting potential of ``0``, have individual firing thresholds, and perform subtractive reset of ``-V_{thr}``.

        added args to LIFBaseTorch:
        tau_ahp (Optional[FloatVector]): An optional array with concrete initialisation data for the time constants of ahp (after hyperpolarization) currents. If not provided, 20ms will be used by default.
        w_ahp (torch.Tensor): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the ahp (after hyperpolarization) feedback weights, which must be a matrix with shape ``(Nout, Nin)``. If the model is not initialised in ahp mode, then you may not provide ``w_ahp``.

        """

        super().__init__(
            shape=shape,
            *args,
            **kwargs,
        )

        w_ahp_shape = (self.size_out,)

        self.w_ahp: P_tensor = rp.Parameter(
            w_ahp,
            shape=w_ahp_shape,
            init_func= lambda s:torch.ones(s)* (-9e-1),
            # init_func=weight_init_func,
            family="weights",
            cast_fn=self.to_float_tensor,
        )
        """ (Tensor) ahp (after hyperpolarization feedback) weights `(Nout, Nin)` """

        self.tau_ahp: P_tensor = rp.Parameter(
            tau_ahp,
            family="taus",
            shape=[
                (
                    self.size_out,
                ),
                (
                    1,
                ),
                (),
            ],
            init_func=lambda s: torch.ones(s) * 20e-3,
            cast_fn=self.to_float_tensor,
        )
        """ (Tensor) Synaptic time constants `(Nin,)` or `()` """  

        self.iahp: P_tensor = rp.State(
            shape=(self.size_out),
            init_func=torch.zeros,
            cast_fn=self.to_float_tensor,
        )
        """ (Tensor)  currents `(Nin,)` """

        self._record_iahp = None

    def evolve(self, input_data: torch.Tensor, record: bool = False
        ) -> Tuple[Any, Any, Any]:
        self._record = record

        # - Evolve with superclass evolution
        output_data, _, _ = super().evolve(input_data, record)

        # - Build state record

        record_dict = (
            {
                "vmem": self._record_vmem,
                "isyn": self._record_isyn,
                "spikes": self._record_spikes,
                "irec": self._record_irec,
                "iahp": self._record_iahp,
                "U": self._record_U,
            }
            if record
            else {}
        )

        return output_data, self.state(), record_dict


    def _syn_integration(self):
        tau_syn = self.tau_syn.broadcast_to((self.size_out, self.n_synapses))
        # if self.has_ahp:
        tau_syn = torch.cat((tau_syn, self.tau_ahp.reshape(self.size_out,1)),1)
        return tau_syn.flatten().detach().numpy()

    def  _wahp_reshape(self):
        # - to match the shape of w_ahp with the shape of w_rec for mapper
        # w_ahp is a vector while traning but for mapper we build matrix out of that (size: n_neourons)
        w_ahp = torch.zeros((self.size_out, self.size_out))
        for i in range(self.size_out):
            w_ahp[i,i] += self.w_ahp[i] 
        return w_ahp    

    def as_graph(self) -> GraphModuleBase:
        tau_mem = self.tau_mem.broadcast_to((self.size_out,)).flatten().detach().numpy()
        # - to integrate tau_ahp (if present) with tau_syn        
        tau_syn = self._syn_integration()
        w_ahp = self._wahp_reshape()

        threshold = (
            self.threshold.broadcast_to((self.size_out,)).flatten().detach().numpy()
        )
        bias = self.bias.broadcast_to((self.size_out,)).flatten().detach().numpy()

        # - Generate a GraphModule for the neurons
        neurons = LIFNeuronWithSynsRealValue._factory(
            self.size_in,
            self.size_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
            tau_mem,
            tau_syn,
            # tau_ahp,
            threshold,
            bias,
            self.dt,
        )

        # - Include recurrent weights if present and combine them with ahp weights
        # - Weights are connected over the existing input and output nodes
        all_wrec = torch.cat((self.w_rec, w_ahp), 1)  if len(self.attributes_named("w_rec")) > 0 else w_ahp
        
        w_rec_graph = LinearWeights(
            neurons.output_nodes,
            neurons.input_nodes,
            f"{type(self).__name__}_recurrent_{self.name}_{id(self)}",
            self,
            all_wrec.detach().numpy(),
        )
        
        # - Return a graph containing neurons and optional weights
        return as_GraphHolder(neurons)


    @property
    def gamma(self):
        return torch.exp(-self.dt / self.tau_ahp).to(self.tau_ahp.device)    


    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        ----------
        data: Tensor
            Data takes the shape of (batch, time_steps, n_synapses)

        Returns
        -------
        out: Tensor
            Out of spikes with the shape (batch, time_steps, Nout)

        """
        # - Auto-batch over input data
        input_data, (vmem, spikes, isyn, iahp) = self._auto_batch(
            input_data,
            (self.vmem, self.spikes, self.isyn, self.iahp),
            (
                (self.size_out,),
                (self.size_out,),
                (self.size_out, self.n_synapses),
                (self.size_out,),

            ),
        )
        n_batches, n_timesteps, _ = input_data.shape

        # - Reshape data over separate input synapses
        input_data = input_data.reshape(
            n_batches, n_timesteps, self.size_out, self.n_synapses
        )

        # - Set up state record and output
        if self._record:
            self._record_vmem = torch.zeros(n_batches, n_timesteps, self.size_out)
            self._record_isyn = torch.zeros(
                n_batches, n_timesteps, self.size_out, self.n_synapses+1
            )
            self._record_irec = torch.zeros(
                n_batches, n_timesteps, self.size_out, self.n_synapses
            )

            self._record_iahp = torch.zeros(
                n_batches, n_timesteps, self.size_out
            )

            self._record_U = torch.zeros(n_batches, n_timesteps, self.size_out)

        self._record_spikes = torch.zeros(
            n_batches, n_timesteps, self.size_out, device=input_data.device
        )

        # - Calculate and cache updated values for decay factors
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        noise_zeta = self.noise_std * torch.sqrt(torch.tensor(self.dt))

        # - Generate membrane noise trace
        noise_ts = noise_zeta * torch.randn(
            (n_batches, n_timesteps, self.size_out), device=vmem.device
        )

        # - Loop over time
        for t in range(n_timesteps):
            # Integrate synaptic input
            isyn = isyn + input_data[:, t]

            # - Apply spikes over the recurrent weights
            if hasattr(self, "w_rec"):
                irec = F.linear(spikes, self.w_rec.T).reshape(
                    n_batches, self.size_out, self.n_synapses
                )
                isyn = isyn + irec

            isyn *= beta
 # - Apply spikes over the ahp weights
            if hasattr(self, "w_ahp"):
                
                iahp = iahp + torch.mul(spikes, 
                self.w_ahp.repeat(n_batches,1).reshape(n_batches, self.size_out))

                iahp *= gamma  
                # isyn = isyn + iahp.reshape(n_batches, self.size_out,1)
                isyn_ = torch.cat((isyn, iahp.reshape(n_batches, self.size_out,1)), 2)

            # Decay synaptic and membrane state
            vmem *= alpha

            # Integrate membrane state and apply noise
            vmem = vmem + isyn_.sum(2) + noise_ts[:, t, :] + self.bias

            # - Spike generation
            spikes = self.spike_generation_fn(
                vmem, self.threshold, self.learning_window, self.max_spikes_per_dt
            )

            # - Apply subtractive membrane reset
            vmem = vmem - spikes * self.threshold

            # - Maintain state record
            if self._record:
                self._record_vmem[:, t] = vmem
                self._record_isyn[:, t] = isyn_
                if hasattr(self, "w_rec"):
                    self._record_irec[:, t] = irec

                if hasattr(self, "w_ahp"):
                    self._record_iahp[:, t] = iahp 
                    

                self._record_U[:, t] = sigmoid(vmem * 20.0, self.threshold)

            # - Maintain output spike record
            self._record_spikes[:, t] = spikes

        # - Update states
        self.vmem = vmem[0].detach()
        self.isyn = isyn[0].detach()
        self.iahp = iahp[0].detach()

        self.spikes = spikes[0].detach()

        # - Return output
        return self._record_spikes
