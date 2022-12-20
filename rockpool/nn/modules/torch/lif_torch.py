"""
Implement a LIF Module, using a Torch backend

Provides :py:class:`.LIFBaseTorch` base class for LIF torch modules, and :py:class:`.LIFTorch` module.
"""

from tempfile import gettempprefix
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
        ctx.max_spikes_per_dt = max_spikes_per_dt
        nr_spikes = ((data >= threshold) * torch.floor(data / threshold)).float()
        nr_spikes[nr_spikes > max_spikes_per_dt] = max_spikes_per_dt.float()
        return nr_spikes

    @staticmethod
    def backward(ctx, grad_output):
        (membranePotential,) = ctx.saved_tensors

        vmem_shifted = membranePotential - ctx.threshold / 2
        nr_spikes_shifted = torch.clamp(
            torch.div(vmem_shifted, ctx.threshold, rounding_mode="floor"),
            max=ctx.max_spikes_per_dt - 1,
        )

        vmem_periodic = vmem_shifted - nr_spikes_shifted * ctx.threshold
        vmem_below = vmem_shifted * (membranePotential < ctx.threshold)
        vmem_above = vmem_periodic * (membranePotential >= ctx.threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = (
            torch.exp(-torch.abs(vmem_new - ctx.threshold / 2) / ctx.window)
            / ctx.threshold
        )

        return (
            grad_output * spikePdf,
            grad_output * -spikePdf * membranePotential / ctx.threshold,
            None,
            None,
        )


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
        leak_mode: str = "taus",
        tau_mem: Optional[Union[FloatVector, P_float]] = None,
        tau_syn: Optional[Union[FloatVector, P_float]] = None,
        alpha: Optional[Union[FloatVector, P_float]] = None,
        beta: Optional[Union[FloatVector, P_float]] = None,
        dash_mem: Optional[Union[IntVector, P_int]] = None,
        dash_syn: Optional[Union[IntVector, P_int]] = None,
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
            leak_mode (str): sets the training mode of time constants:    Default: 'taus'

            if 'taus' tau_mem and tau_syn are set to trainable parameters
            if 'decays' alpha and beta are set to trainable parameters (alpha and beta are:  \exp(-dt / \tau_{mem}) and  \exp(-dt / \tau_{syn}) respectively)
            if 'bitshifts' dash_mem and dash_syn are set to traianble parameters. dash_mem and dash_syn are bitshift equivalent of decays alpha = 1-(1/(2**dash_mem))

            Note:
            if time constants (taus) are not passed as Constant in the instantiation of module they will be set to traianble parameters (default leak_mode). The user can choose to train time constants in different parameter spaces as well (decays and bitshifts)

            tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 20ms will be used by default.

            alpha (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane decays. If not provided, 0.5 will be used by default.
            beta (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic decays. If not provided, 0.5 will be used by default.

            dash_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane bitshifts. If not provided, 1 will be used by default.
            dash_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic bitshifts. If not provided, 1 will be used by default.

            bias (Optional[FloatVector]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, ``0.0`` will be used by default.
            threshold (FloatVector): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            has_rec (bool): When ``True`` the module provides a trainable recurrent weight matrix. Default ``False``, module is feed-forward.
            w_rec (torch.Tensor): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a matrix with shape ``(Nout, Nin)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: ``0.0`` (no noise)
            spike_generation_fn (Callable): Function to call for spike production. Usually simple threshold crossing. Implements the surrogate gradient function in the backward call. (StepPWL or PeriodicExponential).
            learning_window (float): Cutoff value for the surrogate gradient.
            max_spikes_per_dt (int): The maximum number of events that will be produced in a single time-step. Default: ``np.inf``; do not clamp spiking.
            weight_init_func (Optional[Callable[[Tuple], torch.tensor]): The initialisation function to use when generating recurrent weights. Default: ``None`` (Kaiming initialisation)
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms

        """

        # - Check training mode
        assert leak_mode in [
            "taus",
            "decays",
            "bitshifts",
        ], "Training of time constants in LIFTorch neurons can be done only in one of the following modes: taus, decays, bitshifts"

        self.leak_mode = leak_mode

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
        to_float_tensor = lambda x: torch.as_tensor(x, dtype=torch.float)

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

        if self.leak_mode == "taus":
            self._tau_mem: P_tensor = rp.Parameter(
                tau_mem,
                family="taus",
                shape=[(self.size_out,), ()],
                init_func=lambda s: torch.ones(s) * 20e-3,
                cast_fn=to_float_tensor,
            )
            """ (Tensor) Membrane time constants `(Nout,)` or `()` """

            self._tau_syn: P_tensor = rp.Parameter(
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

        elif self.leak_mode == "decays":
            self._alpha: P_tensor = rp.Parameter(
                alpha,
                family="decays",
                shape=[(self.size_out,), ()],
                init_func=lambda s: torch.ones(s) * 0.5,
                cast_fn=to_float_tensor,
            )
            """ (Tensor) Membrane decay factor `(Nout,)` or `()` """

            self._beta: P_tensor = rp.Parameter(
                beta,
                family="decays",
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
                init_func=lambda s: torch.ones(s) * 0.5,
                cast_fn=to_float_tensor,
            )
            """ (Tensor) Synaptic decay factor `(Nin,)` or `()` """

        elif self.leak_mode == "bitshifts":
            self._dash_mem: P_tensor = rp.Parameter(
                dash_mem,
                family="bitshifts",
                shape=[(self.size_out,), ()],
                init_func=lambda s: torch.ones(s),
                cast_fn=to_float_tensor,
            )
            """ (Tensor) membrane bitshift in xylo `(Nout,)` or `()` """

            self._dash_syn: P_tensor = rp.Parameter(
                dash_syn,
                family="bitshifts",
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
                init_func=lambda s: torch.ones(s),
                cast_fn=to_float_tensor,
            )
            """ (Tensor) synaptic bitshift in xylo `(Nout,)` or `()` """

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

        # - Placeholders for state recordings
        self._record_dict = {}
        self._record = False

    def evolve(
        self, input_data: torch.Tensor, record: bool = False
    ) -> Tuple[Any, Any, Any]:
        # - Keep track of "record" flag for use by `forward` method
        self._record = record

        # - Evolve with superclass evolution
        output_data, _, _ = super().evolve(input_data, record)

        # - Obtain state record dictionary
        record_dict = self._record_dict if record else {}

        # - Clear record in order to avoid non-leaf tensors hanging around
        self._record_dict = {}

        return output_data, self.state(), record_dict

    def as_graph(self) -> GraphModuleBase:
        # - Get neuron parameters for export

        tau_mem = self.tau_mem.expand((self.size_out,)).flatten().detach().cpu().numpy()
        tau_syn = (
            self.tau_syn.expand((self.size_out, self.n_synapses))
            .flatten()
            .detach()
            .cpu()
            .numpy()
        )
        threshold = (
            self.threshold.expand((self.size_out,)).flatten().detach().cpu().numpy()
        )
        bias = self.bias.expand((self.size_out,)).flatten().detach().cpu().numpy()

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
                self.w_rec.detach().cpu().numpy(),
            )

        # - Return a graph containing neurons and optional weights
        return as_GraphHolder(neurons)

    def calc_alpha(self) -> torch.Tensor:
        """
        Decay factor for membrane time constants :py:attr:`.LIFTorch.tau_mem`
        """
        return torch.exp(-self.dt / self.tau_mem).to(self.tau_mem.device)

    def calc_beta(self) -> torch.Tensor:
        """
        Decay factor for synaptic time constants :py:attr:`.LIFTorch.tau_syn`
        """
        return torch.exp(-self.dt / self.tau_syn).to(self.tau_syn.device)

    @property
    def get_all_params(self):

        if self.leak_mode == "taus":
            tau_mem, tau_syn = self._tau_mem, self._tau_syn
            alpha, beta = torch.exp(-self.dt / self._tau_mem).to(
                self._tau_mem.device
            ), torch.exp(-self.dt / self._tau_syn).to(self._tau_syn.device)
            dash_mem = -torch.log2(
                1 - torch.exp(-self.dt / self._tau_mem).to(self._tau_mem.device)
            )
            dash_syn = -torch.log2(
                1 - torch.exp(-self.dt / self._tau_syn).to(self._tau_syn.device)
            )

        elif self.leak_mode == "decays":

            tau_mem, tau_syn = -(self.dt / torch.log(self._alpha)), -(
                self.dt / torch.log(self._beta)
            )
            alpha, beta = self._alpha, self._beta
            dash_mem, dash_syn = -torch.log2(1 - self._alpha), -torch.log2(
                1 - self._beta
            )

        elif self.leak_mode == "bitshifts":
            tau_mem, tau_syn = -self.dt / torch.log(
                1 - 1 / (2**self._dash_mem)
            ), -self.dt / torch.log(1 - 1 / (2**self._dash_syn))
            alpha, beta = 1 - 1 / (2**self._dash_mem), 1 - 1 / (2**self._dash_syn)
            dash_mem, dash_syn = self._dash_mem, self._dash_syn

        return tau_mem, tau_syn, alpha, beta, dash_mem, dash_syn

    @property
    def tau_mem(self) -> torch.Tensor:
        tau_mem, *_ = self.get_all_params
        return tau_mem

    @property
    def tau_syn(self) -> torch.Tensor:
        _, tau_syn, *_ = self.get_all_params
        return tau_syn

    @property
    def alpha(self) -> torch.Tensor:
        _, _, alpha, *_ = self.get_all_params
        return alpha

    @property
    def beta(self) -> torch.Tensor:
        _, _, _, beta, *_ = self.get_all_params
        return beta

    @property
    def dash_mem(self) -> torch.Tensor:
        *_, dash_mem, _ = self.get_all_params
        return dash_mem

    @property
    def dash_syn(self) -> torch.Tensor:
        *_, dash_syn = self.get_all_params
        return dash_syn

    @property
    def beta(self) -> torch.Tensor:
        _, _, _, beta, *_ = self.get_all_params
        return beta

    # @property
    # def beta(self) -> torch.Tensor:
    #     alpha, *_ = self._get_all_TCs()
    #     self._set_all_TCs(beta = new_value)


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
            self._record_dict["vmem"] = torch.zeros(
                n_batches, n_timesteps, self.size_out
            )
            self._record_dict["isyn"] = torch.zeros(
                n_batches, n_timesteps, self.size_out, self.n_synapses
            )
            self._record_dict["irec"] = torch.zeros(
                n_batches, n_timesteps, self.size_out, self.n_synapses
            )

            self._record_dict["U"] = torch.zeros(n_batches, n_timesteps, self.size_out)

        self._record_dict["spikes"] = torch.zeros(
            n_batches, n_timesteps, self.size_out, device=input_data.device
        )
        # if self.leak_mode == 'decays':
        #     alpha, beta = self.alpha, self.beta

        # elif self.leak_mode == 'bitshifts':
        #     alpha, beta = 1 - 1 / (2**self.dash_mem), 1 - 1 / (2**self.dash_syn)
        # else:
        #     alpha, beta = self.calc_alpha(), self.calc_beta()

        # alpha, beta = _()
        # dash_mem, dash_syn = _()

        # tau_mem, tau_syn = -()

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
            vmem *= self.alpha.to(vmem.device)
            isyn *= self.beta.to(isyn.device)

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
                self._record_dict["vmem"][:, t] = vmem
                self._record_dict["isyn"][:, t] = isyn

                if hasattr(self, "w_rec"):
                    self._record_dict["irec"][:, t] = irec

                self._record_dict["U"][:, t] = sigmoid(vmem * 20.0, self.threshold)

            # - Maintain output spike record
            self._record_dict["spikes"][:, t] = spikes

        # - Update states
        self.vmem = vmem[0].detach()
        self.isyn = isyn[0].detach()
        self.spikes = spikes[0].detach()

        # - Return output
        return self._record_dict["spikes"]
