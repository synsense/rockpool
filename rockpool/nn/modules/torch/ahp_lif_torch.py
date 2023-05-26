"""
Implement a modified version of LIF Module (ahp, after hyperpolarization feedback,  is added)), using a Torch backend

Provides :py:class:`.aLIFTorch` module.
"""

from typing import Union, Tuple, Callable, Optional, Any
from rockpool.nn.modules.torch.lif_torch import (
    sigmoid,
    LIFBaseTorch,
)

import torch
import torch.nn.functional as F
import rockpool.parameters as rp

from rockpool.typehints import *

from rockpool.graph import (
    GraphModuleBase,
    GraphHolder,
    LIFNeuronWithSynsRealValue,
    LinearWeights,
)

__all__ = ["aLIFTorch"]


class aLIFTorch(LIFBaseTorch):
    """
    A leaky integrate-and-fire spiking neuron model with adaptive hyperpolarisation, with a Torch backend
    """

    def __init__(
        self,
        shape: Union[Tuple, int],
        tau_ahp: Optional[Union[FloatVector, P_float]] = None,
        w_ahp: torch.Tensor = None,
        w_ahp_init_func: Optional[Callable[[Tuple], torch.tensor]] = lambda s: -0.9
        * torch.ones(s),
        *args,
        **kwargs,
    ):
        """
        Instantiate an aLIFTorch module

        It is built based on LIFTorch with an added inhibitory recurrent connection called ahp (after hyperpolarization) feedback. This connection includes wahp and tau_ahp which currently are set to a constant negative scalar and trainable vectors, respectively. The role of this feedback is to pull down the membrane voltage and reduce the firing rate

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

        Args:
            shape (tuple): Either a single dimension ``(Nout,)``, which defines a feed-forward layer of LIF modules with equal amounts of synapses and neurons, or two dimensions ``(Nin, Nout)``, which defines a layer of ``Nin`` synapses and ``Nout`` LIF neurons.
            tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 20ms will be used by default.
            bias (Optional[FloatVector]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, ``0.0`` will be used by default.
            threshold (FloatVector): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            has_rec (bool): When ``True`` the module provides a trainable recurrent weight matrix. Default ``False``, module is feed-forward.
            w_rec (torch.Tensor): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a matrix with shape ``(Nout, Nin)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: ``0.0`` (no noise)
            spike_generation_fn (Callable): Function to call for spike production. Usually simple threshold crossing. Implements the surrogate gradient function in the backward call. (StepPWL or PeriodicExponential).
            learning_window (float): Cutoff value for the surrogate gradient.
            max_spikes_per_dt (int): The maximum number of events that will be produced in a single time-step. Default: ``np.inf``; do not clamp spiking.
            weight_init_func (Optional[Callable[[Tuple], torch.tensor]): The initialisation function to use when generating weights. Default: ``None`` (Kaiming initialisation)
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            tau_ahp (Optional[FloatVector]): An optional array with concrete initialisation data for the time constants of ahp (after hyperpolarization) currents. If not provided, 20ms will be used by default.
            w_ahp (torch.Tensor): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the ahp (after hyperpolarization) feedback weights, which must be a matrix with shape ``(Nout, Nin)``. If the model is not initialised in ahp mode, then you may not provide ``w_ahp``.
        """

        # - Initialise superclass
        super().__init__(
            shape=shape,
            *args,
            **kwargs,
        )

        # - To-float-tensor conversion utility
        to_float_tensor = lambda x: torch.as_tensor(x, dtype=torch.float)

        self.w_ahp: P_tensor = rp.Parameter(
            w_ahp,
            shape=[(self.size_out,), ()],
            init_func=w_ahp_init_func,
            family="weights",
            cast_fn=to_float_tensor,
        )
        """ (Tensor) ahp (after hyperpolarization feedback) weights `(Nout, Nin)` """

        self.tau_ahp: P_tensor = rp.Parameter(
            tau_ahp,
            family="taus",
            shape=[
                (self.size_out,),
                (1,),
                (),
            ],
            init_func=lambda s: torch.ones(s) * 20e-3,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Synaptic time constants `(Nin,)` or `()` """

        self.iahp: P_tensor = rp.State(
            shape=(self.size_out),
            init_func=torch.zeros,
            cast_fn=to_float_tensor,
        )
        """ (Tensor)  currents `(Nin,)` """

    @property
    def gamma(self) -> torch.Tensor:
        """
        Decay factor for AHP synapses :py:attr:`.aLIFTorch.tau_ahp`
        """
        return torch.exp(-self.dt / self.tau_ahp).to(self.tau_ahp.device)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        Args:
            input_data (torch.Tensor): Data takes the shape of (batch, time_steps, n_synapses)

        Returns:
            torch.Tensor: Out of spikes with the shape (batch, time_steps, Nout)
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
            self._record_dict["vmem"] = torch.zeros(
                n_batches, n_timesteps, self.size_out, device=vmem.device
            )
            self._record_dict["isyn"] = torch.zeros(
                n_batches,
                n_timesteps,
                self.size_out,
                self.n_synapses,
                device=vmem.device,
            )
            self._record_dict["irec"] = torch.zeros(
                n_batches,
                n_timesteps,
                self.size_out,
                self.n_synapses,
                device=vmem.device,
            )

            self._record_dict["iahp"] = torch.zeros(
                n_batches, n_timesteps, self.size_out, device=vmem.device
            )

        self._record_dict["spikes"] = torch.zeros(
            n_batches, n_timesteps, self.size_out, device=vmem.device
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
            # - Integrate synaptic input
            isyn = isyn + input_data[:, t]

            # - Apply spikes over the recurrent weights
            if hasattr(self, "w_rec"):
                irec = F.linear(spikes, self.w_rec.T).reshape(
                    n_batches, self.size_out, self.n_synapses
                )
                isyn = isyn + irec

            # - Decay isyn
            isyn *= beta

            # - Apply spikes over the ahp weights
            if hasattr(self, "w_ahp"):
                iahp = iahp + torch.mul(
                    spikes,
                    self.w_ahp.repeat(n_batches, 1).reshape(n_batches, self.size_out),
                )

                # - Decay iahp
                iahp *= gamma

                # - Build augmented isyn
                isyn_ = torch.cat((isyn, iahp.reshape(n_batches, self.size_out, 1)), 2)
            else:
                isyn_ = isyn

            # - Decay membrane state
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
                self._record_dict["vmem"][:, t] = vmem
                self._record_dict["isyn"][:, t] = isyn
                if hasattr(self, "w_rec"):
                    self._record_dict["irec"][:, t] = irec

                if hasattr(self, "w_ahp"):
                    self._record_dict["iahp"][:, t] = iahp

            # - Maintain output spike record
            self._record_dict["spikes"][:, t] = spikes

        # - Update states
        self.vmem = vmem[0].detach()
        self.isyn = isyn[0].detach()
        self.iahp = iahp[0].detach()
        self.spikes = spikes[0].detach()

        # - Return output
        return self._record_dict["spikes"]

    def as_graph(self) -> GraphModuleBase:
        def syn_integration(self):
            """
            Create a tau_syn matrix including tau_ahp, for :py:meth`.as_graph` export

            Returns:
                np.array: ``tau_syn``
            """
            tau_syn = self.tau_syn.expand((self.size_out, self.n_synapses))
            tau_ahp = self.tau_ahp.reshape((-1, 1)).expand((self.size_out, 1))
            tau_syn_ahp = torch.cat((tau_syn, tau_ahp), 1)
            return tau_syn_ahp.flatten().detach().numpy()

        def w_ahp_reshape(self):
            """
            Create and reshape a ``w_ahp`` matrix for :py:meth`.as_graph` export

            Returns:
                np.array: ``w_ahp``
            """
            # - to match the shape of w_ahp with the shape of w_rec for mapper
            # w_ahp is a vector while training but for mapper we build matrix out of that of size: (n_neurons, n_neurons)
            w_ahp = torch.zeros((self.size_out, self.size_out))
            for i in range(self.size_out):
                w_ahp[i, i] += self.w_ahp[i]
            return w_ahp

        # - Get tau_mem for export
        tau_mem = self.tau_mem.expand((self.size_out,)).flatten().detach().numpy()

        # - Get tau_syn and w_ahp for export
        tau_syn_ahp = syn_integration(self)
        w_ahp = w_ahp_reshape(self)

        # - Get threshold and bias parameters for export
        threshold = self.threshold.expand((self.size_out,)).flatten().detach().numpy()
        bias = self.bias.expand((self.size_out,)).flatten().detach().numpy()

        # - Generate a GraphModule for the neurons
        neurons = LIFNeuronWithSynsRealValue._factory(
            self.size_in + self.size_out,  # Including AHP synapses
            self.size_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
            tau_mem,
            tau_syn_ahp,  # Including AHP synapses
            threshold,
            bias,
            self.dt,
        )

        # - Include recurrent weights if present and combine them with ahp weights
        # - Weights are connected over the existing input and output nodes
        w_rec = (
            self.w_rec
            if hasattr(self, "w_rec")
            else torch.zeros(self.size_out, self.size_in)
        )

        all_wrec = torch.cat((w_rec, w_ahp), 1)

        w_rec_graph = LinearWeights(
            neurons.output_nodes,
            neurons.input_nodes,
            f"{type(self).__name__}_recurrent_{self.name}_{id(self)}",
            self,
            all_wrec.detach().numpy(),
        )

        # - Return a graph containing neurons and weights, but trimming off the AHP input nodes
        return GraphHolder(
            neurons.input_nodes[: self.size_in],
            neurons.output_nodes,
            neurons.name,
            None,
        )
