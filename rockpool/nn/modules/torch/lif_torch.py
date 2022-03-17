"""
Implement a LIF Module, using a Torch backend
"""

from typing import Union, Tuple, Callable, Optional, Any
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
    Heaviside step function with piece-wise linear surrogate to use as spike-generation surrogate
    """

    @staticmethod
    def forward(
        ctx,
        x,
        threshold=torch.tensor(1.0),
        window=torch.tensor(0.5),
        max_spikes_per_dt=torch.tensor(float("inf")),
    ):
        ctx.save_for_backward(x, threshold)
        ctx.window = window
        nr_spikes = ((x >= threshold) * torch.floor(x / threshold)).float()
        nr_spikes[nr_spikes > max_spikes_per_dt] = max_spikes_per_dt.float()
        return nr_spikes

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_threshold = grad_window = grad_max_spikes_per_dt = None

        mask = x >= (threshold - ctx.window)
        if ctx.needs_input_grad[0]:
            grad_x = grad_output / threshold * mask

        if ctx.needs_input_grad[1]:
            grad_threshold = -x * grad_output / (threshold**2) * mask

        return grad_x, grad_threshold, grad_window, grad_max_spikes_per_dt


class PeriodicExponential(torch.autograd.Function):
    """
    Subtract from membrane potential on reaching threshold
    """

    @staticmethod
    def forward(
        ctx,
        data,
        threshold=1.0,
        window=0.5,
        max_spikes_per_dt=torch.tensor(float("inf")),
    ):
        ctx.save_for_backward(data.clone())
        ctx.threshold = threshold
        ctx.window = window
        nr_spikes = ((data >= threshold) * torch.floor(data / threshold)).float()
        nr_spikes[nr_spikes > max_spikes_per_dt] = max_spikes_per_dt.float()
        return nr_spikes

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

        return grad_output * spikePdf, None, None, None


# - Surrogate functions to use in learning
def sigmoid(x: FloatVector, threshold: FloatVector) -> FloatVector:
    """
    Sigmoid function

    :param FloatVector x: Input value

    :return FloatVector: Output value
    """
    return torch.tanh(x + 1 - threshold) / 2 + 0.5


class LIFBaseTorch(TorchModule):
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
        max_spikes_per_dt: P_int = torch.tensor(float("inf")),
        weight_init_func: Optional[
            Callable[[Tuple], torch.tensor]
        ] = lambda s: init.kaiming_uniform_(torch.empty(s)),
        dt: P_float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

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

        self.n_neurons = self.size_out
        self.n_synapses: P_int = shape[0] // shape[1]
        """ (int) Number of input synapses per neuron """

        self.dt: P_float = rp.SimulationParameter(dt)
        """ (float) Euler simulator time-step in seconds"""

        # - To-float-tensor conversion utility
        to_float_tensor = lambda x: torch.tensor(x).float()

        # - Initialise recurrent weights
        w_rec_shape = (self.size_out, self.size_in)
        if has_rec:
            self.w_rec: P_tensor = rp.Parameter(
                w_rec,
                shape=w_rec_shape,
                init_func=weight_init_func,
                family="weights",
                cast_fn=to_float_tensor,
            )
            """ (Tensor) Recurrent weights `(Nout, Nin)` """
        else:
            if w_rec is not None:
                raise ValueError("`w_rec` may not be provided if `has_rec` is `False`")

        self.noise_std: P_float = rp.SimulationParameter(noise_std)
        """ (float) Noise std.dev. injected onto the membrane of each neuron during evolution """

        self.tau_mem: P_tensor = rp.Parameter(
            tau_mem,
            family="taus",
            shape=[(self.size_out,), ()],
            init_func=lambda s: torch.ones(s) * 20e-3,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Membrane time constants `(Nout,)` or `()` """

        self.tau_syn: P_tensor = rp.Parameter(
            tau_syn,
            family="taus",
            shape=[
                (
                    self.size_out,
                    self.n_synapses,
                ),
                (
                    1,
                    self.n_synapses,
                ),
                (),
            ],
            init_func=lambda s: torch.ones(s) * 20e-3,
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
            shape=[(self.size_out,), ()],
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
            shape=self.size_out, init_func=torch.zeros, cast_fn=to_float_tensor
        )
        """ (Tensor) Membrane potentials `(Nout,)` """

        self.isyn: P_tensor = rp.State(
            shape=(self.size_out, self.n_synapses),
            init_func=torch.zeros,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Synaptic currents `(Nin,)` """

        self.spikes: P_tensor = rp.State(
            shape=self.size_out, init_func=torch.zeros, cast_fn=to_float_tensor
        )
        """ (Tensor) Spikes `(Nin,)` """

        self.spike_generation_fn: P_Callable = rp.SimulationParameter(
            spike_generation_fn.apply
        )
        """ (Callable) Spike generation function with surrograte gradient """

        self.max_spikes_per_dt: P_int = rp.SimulationParameter(
            max_spikes_per_dt, cast_fn=to_float_tensor
        )
        """ (int) Maximum number of events that can be produced in each time-step """

        # placeholders for recordings
        self._record_vmem = None
        self._record_isyn = None
        self._record_irec = None
        self._record_U = None
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
                "vmem": self._record_vmem,
                "isyn": self._record_isyn,
                "spikes": self._record_spikes,
                "irec": self._record_irec,
                "U": self._record_U,
            }
            if record
            else {}
        )

        return output_data, self.state(), record_dict

    def as_graph(self) -> GraphModuleBase:
        tau_mem = self.tau_mem.broadcast_to((self.size_out,)).flatten().detach().numpy()
        tau_syn = (
            self.tau_syn.broadcast_to((self.size_out, self.n_synapses))
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

    @property
    def beta(self):
        return torch.exp(-self.dt / self.tau_syn).to(self.tau_syn.device)


class LIFTorch(LIFBaseTorch):
    """
    A leaky integrate-and-fire spiking neuron model with a Torch backend

    This module implements the update equations:

    .. math ::

        I_{syn} += S_{in}(t) + S_{rec} \\cdot W_{rec}
        I_{syn} *= \exp(-dt / \tau_{syn})
        V_{mem} *= \exp(-dt / \tau_{mem})
        V_{mem} += I_{syn} + b + \sigma \zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` (or a weighed spike) for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a Wiener noise process with standard deviation :math:`\\sigma` after 1s; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively. :math:`S_{rec}(t)` is a vector containing ``1`` for each neuron that emitted a spike in the last time-step. :math:`W_{rec}` is a recurrent weight matrix, if recurrent weights are used. :math:`b` is an optional bias current per neuron (default 0.).

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr}`, then the neuron emits a spike. The spiking neuron subtracts its own threshold on reset.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        V_{mem, j} = V_{mem, j} - V_{thr}

    Neurons therefore share a common resting potential of ``0``, have individual firing thresholds, and perform subtractive reset of ``-V_{thr}``.
    """

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
        input_data, (vmem, spikes, isyn) = self._auto_batch(
            input_data,
            (self.vmem, self.spikes, self.isyn),
            (
                (self.size_out,),
                (self.size_out,),
                (self.size_out, self.n_synapses),
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
                n_batches, n_timesteps, self.size_out, self.n_synapses
            )
            self._record_irec = torch.zeros(
                n_batches, n_timesteps, self.size_out, self.n_synapses
            )

            self._record_U = torch.zeros(n_batches, n_timesteps, self.size_out)

        self._record_spikes = torch.zeros(
            n_batches, n_timesteps, self.size_out, device=input_data.device
        )

        # - Calculate and cache updated values for decay factors
        alpha = self.alpha
        beta = self.beta
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

            # Decay synaptic and membrane state
            vmem *= alpha
            isyn *= beta

            # Integrate membrane state and apply noise
            vmem = vmem + isyn.sum(2) + noise_ts[:, t, :] + self.bias

            # - Spike generation
            spikes = self.spike_generation_fn(
                vmem, self.threshold, self.learning_window, self.max_spikes_per_dt
            )

            # - Apply subtractive membrane reset
            vmem = vmem - spikes * self.threshold

            # - Maintain state record
            if self._record:
                self._record_vmem[:, t] = vmem
                self._record_isyn[:, t] = isyn

                if hasattr(self, "w_rec"):
                    self._record_irec[:, t] = irec

                self._record_U[:, t] = sigmoid(vmem * 20.0, self.threshold)

            # - Maintain output spike record
            self._record_spikes[:, t] = spikes

        # - Update states
        self.vmem = vmem[0].detach()
        self.isyn = isyn[0].detach()
        self.spikes = spikes[0].detach()

        # - Return output
        return self._record_spikes
