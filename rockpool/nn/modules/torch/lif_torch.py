"""
Implement a LIF Module, using a Torch backend
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

from typing import Union, List, Tuple, Callable, Optional, Any
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
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

__all__ = ["LIFTorch"]


class StepPWL(torch.autograd.Function):
    """
    Heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param torch.Tensor x: Input value

    :return torch.Tensor: output value and gradient function
    """

    @staticmethod
    def forward(ctx, data, threshold=0.0, window=0.5):
        ctx.save_for_backward(data.clone())
        ctx.threshold = threshold
        ctx.window = window
        return ((data > 0) * torch.floor(data / threshold)).float()
        # return torch.clip(torch.floor(data + 1.0 - threshold), 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        (data,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[data < -ctx.window] = 0
        return grad_input, None, None


class PeriodicExponential(torch.autograd.Function):
    """
    Subtract from membrane potential on reaching threshold
    """

    @staticmethod
    def forward(ctx, data, threshold=1, window=0.5):
        ctx.save_for_backward(data.clone())
        ctx.threshold = threshold
        ctx.window = window
        return ((data > 0) * torch.floor(data / threshold)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (membranePotential,) = ctx.saved_tensors

        vmem_shifted = membranePotential - ctx.threshold / 2
        vmem_periodic = vmem_shifted - torch.div(
            vmem_shifted, ctx.threshold, rounding_mode="floor"
        )
        vmem_below = vmem_shifted * (membranePotential < ctx.threshold)
        vmem_above = vmem_periodic * (membranePotential >= ctx.threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = (
            torch.exp(-torch.abs(vmem_new - ctx.threshold / 2) / ctx.window)
            / ctx.threshold
        )

        return grad_output * spikePdf, None, None


class LIFBaseTorch(TorchModule):
    """
    A leaky integrate-and-fire spiking neuron model

    This module implements the dynamics:

    .. math ::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t)

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``0.``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{tanh}(V_j + 1) / 2 + .5
    """

    def __init__(
        self,
        shape: tuple,
        tau_mem: Optional[Union[FloatVector, P_float]] = None,
        tau_syn: Optional[Union[FloatVector, P_float]] = None,
        bias: Optional[FloatVector] = None,
        threshold: Optional[FloatVector] = None,
        has_rec: P_bool = False,
        w_rec: torch.Tensor = None,
        noise_std: P_float = 0.0,
        spike_generation_fn: torch.autograd.Function = StepPWL,
        learning_window: P_float = 0.5,
        weight_init_func: Optional[Callable[[Tuple], torch.tensor]] = None,
        dt: P_float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Args:
            shape (tuple): Either a single dimension ``(Nout,)``, which defines a feed-forward layer of LIF modules with equal amounts of synapses and neurons, or two dimensions ``(Nin, Nout)``, which defines a layer of ``Nin`` synapses and ``Nout`` LIF neurons.
            tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 10ms will be used by default.
            bias (Optional[FloatVector]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, ``0.0`` will be used by default.
            threshold (FloatVector): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            has_rec (bool): When ``True`` the module provides a trainable recurrent weight matrix. Default ``False``, module is feed-forward.
            w_rec (torch.Tensor): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a matrix with shape ``(Nout, Nin)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: ``0.0``
            spike_generation_fn (Callable): Function to call for spike production. Usually simple threshold crossing. Implements the surrogate gradient function in the backward call. (StepPWL or PeriodicExponential).
            learning_window (float): Cutoff value for the surrogate gradient.
            weight_init_func (Optional[Callable[[Tuple], torch.tensor]): The initialisation function to use when generating weights. Default: ``None`` (Kaiming initialisation)
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            device: Defines the device on which the model will be processed.
        """
        # - Check shape argument
        if np.size(shape) == 1:
            shape = (np.array(shape).item(), np.array(shape).item())

        if np.size(shape) > 2:
            raise ValueError(
                "`shape` must be a one- or two-element tuple `(Nin, Nout)`."
            )

        # - Initialise superclass
        super().__init__(
            shape=shape,
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        self.n_synapses: P_int = rp.SimulationParameter(shape[0] // shape[1])
        """ (int) Number of input synapses per neuron """

        self.n_neurons: P_int = rp.SimulationParameter(shape[1])
        """ (int) Number of neurons """

        self.dt: P_float = rp.SimulationParameter(dt)
        """ (float) Euler simulator time-step in seconds"""

        # - Initialise recurrent weights
        if weight_init_func is None:
            weight_init_func = lambda s: init.kaiming_uniform_(torch.empty(s))

        w_rec_shape = (self.size_out, self.size_in)
        if has_rec:
            self.w_rec: P_tensor = rp.Parameter(
                w_rec,
                shape=w_rec_shape,
                init_func=weight_init_func,
                family="weights",
            )
            """ (Tensor) Recurrent weights `(Nout, Nin)` """
        else:
            if w_rec is not None:
                raise ValueError("`w_rec` may not be provided if `has_rec` is `False`")

        self.noise_std: P_float = rp.SimulationParameter(noise_std)
        """ (float) Noise std.dev. injected onto the membrane of each neuron during evolution """

        to_float_tensor = lambda x: torch.tensor(x).float()

        self.tau_mem: P_tensor = rp.Parameter(
            tau_mem,
            family="taus",
            shape=[(self.n_neurons,), ()],
            init_func=lambda s: torch.ones(s) * 100e-3,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Membrane time constants `(Nout,)` or `()` """

        self.tau_syn: P_tensor = rp.Parameter(
            tau_syn,
            family="taus",
            shape=[
                (
                    self.n_neurons,
                    self.n_synapses,
                ),
                (
                    1,
                    self.n_synapses,
                ),
                (),
            ],
            init_func=lambda s: torch.ones(s) * 50e-3,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Synaptic time constants `(Nin,)` or `()` """

        self.bias: P_tensor = rp.Parameter(
            bias,
            shape=[(self.size_out,), ()],
            family="bias",
            init_func=torch.zeros,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Neuron biases `(Nout,)` or `()` """

        self.threshold: P_tensor = rp.Parameter(
            threshold,
            shape=[(self.n_neurons,), ()],
            family="thresholds",
            init_func=torch.ones,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Firing threshold for each neuron `(Nout,)` """

        self.learning_window: P_tensor = rp.SimulationParameter(
            learning_window,
            cast_fn=to_float_tensor,
        )
        """ (float) Learning window cutoff for surrogate gradient function """

        self.vmem: P_tensor = rp.State(
            shape=self.n_neurons, init_func=torch.zeros, cast_fn=to_float_tensor
        )
        """ (Tensor) Membrane potentials `(Nout,)` """

        self.isyn: P_tensor = rp.State(
            shape=(self.n_neurons, self.n_synapses),
            init_func=torch.zeros,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Synaptic currents `(Nin,)` """

        self.spikes: P_tensor = rp.State(
            shape=self.n_neurons, init_func=torch.zeros, cast_fn=to_float_tensor
        )
        """ (Tensor) Spikes `(Nin,)` """

        self.spike_generation_fn: P_Callable = rp.SimulationParameter(
            spike_generation_fn().apply
        )
        """ (Callable) Spike generation function with surrograte gradient """

        # placeholders for recordings
        self._record_Vmem = None
        self._record_Isyn = None
        self._record_spikes = None

        self._record_dict = {}
        self._record = False

    def evolve(
        self, input_data: torch.Tensor, record: bool = False
    ) -> Tuple[Any, Any, Any]:

        self._record = record

        # - Evolve with superclass evolution
        output_data, _, _ = super().evolve(input_data, record)

        # - Build state record
        record_dict = (
            {
                "vmem": self._record_Vmem,
                "isyn": self._record_Isyn,
                "spikes": self._record_spikes,
            }
            if record
            else {}
        )

        return output_data, self.state(), record_dict

    def as_graph(self) -> GraphModuleBase:
        tau_mem = self.tau_mem.broadcast_to((self.size_out,)).flatten().detach().numpy()
        tau_syn = (
            self.tau_syn.broadcast_to((self.n_neurons, self.n_synapses))
            .flatten()
            .detach()
            .numpy()
        )
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
            threshold,
            bias,
            self.dt,
        )

        # - Include recurrent weights if present
        if len(self.attributes_named("w_rec")) > 0:
            # - Weights are connected over the existing input and output nodes
            w_rec_graph = LinearWeights(
                neurons.output_nodes,
                neurons.input_nodes,
                f"{type(self).__name__}_recurrent_{self.name}_{id(self)}",
                self,
                self.w_rec.detach().numpy(),
            )

        # - Return a graph containing neurons and optional weights
        return as_GraphHolder(neurons)

    @property
    def alpha(self):
        return torch.exp(-self.dt / self.tau_mem).to(self.tau_mem.device)

    @alpha.setter
    def alpha(self, val):
        self.tau_mem = (-self.dt / torch.log(val)).to(self.tau_mem.device)

    @property
    def beta(self):
        return torch.exp(-self.dt / self.tau_syn).to(self.tau_syn.device)

    @beta.setter
    def beta(self, val):
        self.tau_syn = (-self.dt / torch.log(val)).to(self.tau_syn.device)


class LIFTorch(LIFBaseTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            vmem = vmem - spikes

            if self._record:
                # recording
                self._record_Vmem[:, t] = vmem
                self._record_Isyn[:, t] = isyn

            self._record_spikes[:, t] = spikes

        self.vmem = vmem[0].detach()
        self.isyn = isyn[0].detach()
        self.spikes = spikes[0].detach()

        return self._record_spikes
