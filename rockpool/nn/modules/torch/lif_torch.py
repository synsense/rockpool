"""
Implement a LIF Module, using a Torch backend

Provides :py:class:`.LIFBaseTorch` base class for LIF torch modules, and :py:class:`.LIFTorch` module.
"""

from tempfile import gettempprefix
from typing import Union, Tuple, Callable, Optional, Any
import warnings
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
        max_spikes_per_dt=torch.tensor(2.0**16),
    ):
        ctx.save_for_backward(x, threshold)
        ctx.window = window
        # .to(x.dtype), not .float(): this is the default spike-generation
        # surrogate for LIFTorch, so an unconditional cast to float32 here
        # silently discarded precision for a module cast to float64, and
        # would silently upcast float16/bfloat16 back to float32 too.
        nr_spikes = ((x >= threshold) * torch.floor(x / threshold)).to(x.dtype)
        clamp_bool = (nr_spikes > max_spikes_per_dt).to(x.dtype)
        nr_spikes -= (nr_spikes - max_spikes_per_dt.to(x.dtype)) * clamp_bool
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
        max_spikes_per_dt=torch.tensor(2.0**16),
    ):
        ctx.save_for_backward(data.clone())
        ctx.threshold = threshold
        ctx.window = window
        ctx.max_spikes_per_dt = max_spikes_per_dt
        # .to(data.dtype), not .float() -- see the matching fix + comment
        # in StepPWL.forward above.
        nr_spikes = ((data >= threshold) * torch.floor(data / threshold)).to(
            data.dtype
        )
        clamp_bool = (nr_spikes > max_spikes_per_dt).to(data.dtype)
        nr_spikes -= (nr_spikes - max_spikes_per_dt.to(data.dtype)) * clamp_bool
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


def decay_to_tau(dt, *decays):
    return tuple([-(dt / torch.log(decay).to(decay.device)) for decay in decays])


def tau_to_decay(dt, *taus):
    return tuple([torch.exp(-dt / tau).to(tau.device) for tau in taus])


def tau_to_bitshift(dt, *taus):
    return tuple([-torch.log2(1 - torch.exp(-dt / tau)).to(tau.device) for tau in taus])


def bitshift_to_tau(dt, *dashes):
    return tuple(
        [-dt / torch.log(1 - 1 / (2**dash)).to(dash.device) for dash in dashes]
    )


def decay_to_bitshift(*decays):
    return tuple([-torch.log2(1 - decay).to(decay.device) for decay in decays])


def bitshift_to_decay(*dashes):
    return tuple([(1 - 1 / (2**dash)).to(dash.device) for dash in dashes])


class LIFBaseTorch(TorchModule):
    def __init__(
        self,
        shape: tuple,
        leak_mode: str = "taus",
        tau_mem: Optional[Union[FloatVector, P_float]] = None,
        tau_syn: Optional[Union[FloatVector, P_float]] = None,
        alpha: Optional[Union[FloatVector, P_float]] = None,
        beta: Optional[Union[FloatVector, P_float]] = None,
        dash_mem: Optional[Union[IntVector, P_float]] = None,
        dash_syn: Optional[Union[IntVector, P_float]] = None,
        bias: Optional[FloatVector] = None,
        threshold: Optional[FloatVector] = None,
        has_rec: P_bool = False,
        w_rec: torch.Tensor = None,
        noise_std: P_float = 0.0,
        spike_generation_fn: torch.autograd.Function = StepPWL,
        learning_window: P_float = 0.5,
        max_spikes_per_dt: P_int = torch.tensor(2.0**16),
        weight_init_func: Optional[
            Callable[[Tuple], torch.tensor]
        ] = lambda s: init.kaiming_uniform_(torch.empty(s)),
        dt: P_float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Note:
            On instantiation, the user can specify how the decay parameters of the module are defined; either as time constant values (:py:attr:`.tau_mem` and :py:attr:`.tau_syn`), as decay factors (:py:attr:`.alpha` and :py:attr:`.beta`) or as bitshift values (:py:attr:`.dash_mem` and `.dash_syn`).
            This is specifed using the ``leak_mode`` argument on initialisation.
            By default, this is set to ``'taus'``, in which the time constants are direct parameters, which are trainable by default.

            if ``'taus'``,  :py:attr:`.tau_mem` and :py:attr:`.tau_syn` are used as model parameters
            if ``'decays'``,  :py:attr:`.alpha` and :py:attr:`.beta` are used as model parameters (:py:attr:`.alpha` and :py:attr:`.beta` are:  :math:`\exp(-dt / \\tau_{mem}`) and  :math:`\\exp(-dt / \\tau_{syn}`) respectively)
            if ``'bitshifts'``, :py:attr:`.dash_mem` and `.dash_syn` are used as model parameters. :py:attr:`.dash_mem` and :py:attr:`.dash_syn` are the bitshift equivalent of decays, such that :math:`.alpha = 1-(1/(2**dash_mem))`

            If decay parameters are passed as :py:func:`.Constant` in the instantiation of module they will be set to non-traianble parameters.

        Args:
            shape (tuple): Either a single dimension ``(Nout,)``, which defines a feed-forward layer of LIF modules with equal amounts of synapses and neurons, or two dimensions ``(Nin, Nout)``, which defines a layer of ``Nin`` synapses and ``Nout`` LIF neurons.
            leak_mode (str): sets the training mode of time constants:    Default: ``'taus'``. Must be one of ``{'taus', 'decays', 'bitshifts'}``
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
            max_spikes_per_dt (float): The maximum number of events that will be produced in a single time-step. Default: ``2**16``.
            weight_init_func (Optional[Callable[[Tuple], torch.tensor]): The initialisation function to use when generating recurrent weights. Default: ``None`` (Kaiming initialisation)
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms

        """

        # - Check training mode
        if leak_mode not in [
            "taus",
            "decays",
            "bitshifts",
        ]:
            raise ValueError(
                "Training of time constants in `LIFTorch` neurons can be done only in one of the following modes: 'taus', 'decays', 'bitshifts'. `leak_mode` must be one of these values."
            )

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

        # - Initialise dummy parameters list
        self._dummy_params = ()

        self.leak_mode = rp.SimulationParameter(leak_mode)
        """ (str) The mode by which leaks are determined for this module. """

        self.n_neurons = self.size_out
        self.n_synapses: P_int = shape[0] // shape[1]
        """ (int) Number of input synapses per neuron """

        self.dt: P_float = rp.SimulationParameter(dt)
        """ (float) Euler simulator time-step in seconds"""

        # - To-float-tensor conversion utility
        to_float_tensor = lambda x: torch.as_tensor(x, dtype=torch.float)

        # - Initialise recurrent weights
        w_rec_shape = (self.size_out, self.size_in)
        self._has_rec: bool = rp.SimulationParameter(has_rec)
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

            self.w_rec: P_ndarray = rp.SimulationParameter(
                torch.zeros((self.size_out, self.size_in))
            )

        self.noise_std: P_float = rp.SimulationParameter(noise_std)
        """ (float) Noise std.dev. injected onto the membrane of each neuron during evolution """

        if self.leak_mode == "taus":
            if any([alpha, beta, dash_mem, dash_syn]):
                raise ValueError(
                    "current leak mode is set to taus and only parameters from this family can be directly initilized (eg: tau_mem and tau_syn)"
                )

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

            self._dummy_params = ("alpha", "beta", "dash_syn", "dash_mem")

        elif self.leak_mode == "decays":
            if any([tau_mem, tau_syn, dash_mem, dash_syn]):
                raise ValueError(
                    "current leak mode is set to decays and only parameters from this family can be directly initilized (eg: alpha and beta)"
                )

            self.alpha: P_tensor = rp.Parameter(
                alpha,
                family="decays",
                shape=[(self.size_out,), ()],
                init_func=lambda s: torch.ones(s) * 0.5,
                cast_fn=to_float_tensor,
            )
            """ (Tensor) Membrane decay factor `(Nout,)` or `()` """

            self.beta: P_tensor = rp.Parameter(
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

            self._dummy_params = ("tau_syn", "tau_mem", "dash_syn", "dash_mem")

        elif self.leak_mode == "bitshifts":
            if any([alpha, beta, tau_mem, tau_syn]):
                raise ValueError(
                    "current leak mode is set to bitshifts and only parameters from this family can be directly initilized (eg: dash_mem and dash_syn)"
                )

            self.dash_mem: P_tensor = rp.Parameter(
                dash_mem,
                family="bitshifts",
                shape=[(self.size_out,), ()],
                init_func=lambda s: torch.ones(s),
                cast_fn=to_float_tensor,
            )
            """ (Tensor) membrane bitshift in xylo `(Nout,)` or `()` """

            self.dash_syn: P_tensor = rp.Parameter(
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

            self._dummy_params = ("alpha", "beta", "tau_syn", "tau_mem")

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

        # dtype/device-aware init_func for state: reset_state() calls
        # init_func(shape) directly via _reset_attribute() and does NOT
        # re-apply cast_fn (cast_fn only runs once, at construction -- see
        # ParameterBase.__init__ / the State docstring: "Will only be
        # called once on initialisation."). A bare `torch.zeros` here
        # always reset vmem/isyn/spikes back to float32 on every
        # reset_state(), even after the module had been correctly cast
        # with .double() / .half() beforehand -- .to() itself works fine
        # (it casts the existing buffer in place), so the mismatch only
        # ever surfaced on the *next* reset_state() call. `self.threshold`
        # is used as the dtype/device anchor: it's a plain Parameter (not
        # regenerated by reset_state()) and, unlike tau_mem/alpha/dash_mem,
        # it's set in every leak_mode, so it's always a safe reference by
        # the time these State descriptors are constructed.
        def _state_init_func(shape):
            return torch.zeros(
                shape, dtype=self.threshold.dtype, device=self.threshold.device
            )

        self.vmem: P_tensor = rp.State(
            shape=self.size_out, init_func=_state_init_func, cast_fn=to_float_tensor
        )
        """ (Tensor) Membrane potentials `(Nout,)` """

        self.isyn: P_tensor = rp.State(
            shape=(self.size_out, self.n_synapses),
            init_func=_state_init_func,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Synaptic currents `(Nin,)` """

        self.spikes: P_tensor = rp.State(
            shape=self.size_out, init_func=_state_init_func, cast_fn=to_float_tensor
        )
        """ (Tensor) Spikes `(Nin,)` """

        self.spike_generation_fn: P_Callable = rp.SimulationParameter(
            spike_generation_fn.apply
        )
        """ (Callable) Spike generation function with surrograte gradient """

        self.max_spikes_per_dt: P_float = rp.SimulationParameter(
            max_spikes_per_dt, cast_fn=to_float_tensor
        )
        """ (float) Maximum number of events that can be produced in each time-step """

        # - Placeholders for state recordings
        self._record_dict = {}
        self._record = False

    def __getattr__(self, name: str) -> Union[torch.Tensor, TorchModule]:
        """
        Overridden __getattr__ to manage access to decay parameters
        """
        if name in object.__getattribute__(self, "_dummy_params"):
            all_TCs = self._get_all_leak_params()
            return all_TCs[name]

        return super().__getattr__(name)

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
        if self._has_rec:
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

    def _get_all_leak_params(self):
        """
        Calculate and return all decay parameters, depending on leak mode
        """
        if self.leak_mode == "taus":
            # - Compute decay parameters based on taus
            tau_mem = getattr(
                self, "tau_mem", torch.tensor(torch.nan).repeat(self.size_out)
            )
            tau_syn = getattr(
                self, "tau_syn", torch.tensor(torch.nan).repeat(self.size_out)
            )
            alpha, beta = tau_to_decay(self.dt, tau_mem, tau_syn)
            dash_mem, dash_syn = tau_to_bitshift(self.dt, tau_mem, tau_syn)

        elif self.leak_mode == "decays":
            # - Compute decay parameters based on decay constants
            alpha = getattr(
                self, "alpha", torch.tensor(torch.nan).repeat(self.size_out)
            )
            beta = getattr(self, "beta", torch.tensor(torch.nan).repeat(self.size_out))
            tau_mem, tau_syn = decay_to_tau(self.dt, alpha, beta)
            dash_mem, dash_syn = decay_to_bitshift(alpha, beta)

        elif self.leak_mode == "bitshifts":
            # - Compute decay parameters based on bitshift values
            dash_mem = getattr(
                self, "dash_mem", torch.tensor(torch.nan).repeat(self.size_out)
            )
            dash_syn = getattr(
                self, "dash_syn", torch.tensor(torch.nan).repeat(self.size_out)
            )
            tau_mem, tau_syn = bitshift_to_tau(self.dt, dash_mem, dash_syn)
            alpha, beta = bitshift_to_decay(dash_mem, dash_syn)

        # - Return all parameters
        return {
            "tau_mem": tau_mem,
            "tau_syn": tau_syn,
            "alpha": alpha,
            "beta": beta,
            "dash_mem": dash_mem,
            "dash_syn": dash_syn,
        }

    def __setattr__(self, key, value: Any):
        """
        Overridden __setattr__ to manage access to decay parameters
        """
        if hasattr(self, "_dummy_params") and key in self._dummy_params:
            self._set_leak_param(key, value)
        return super().__setattr__(key, value)

    def _set_leak_param(self, name, value):
        """
        Set the value of a named decay parameter, depending on leak mode
        """
        if self.leak_mode == "taus":
            # - Compute tau from `name`
            if name == "alpha":
                return setattr(self, "tau_mem", tau_to_decay(self.dt, value)[0])
            elif name == "beta":
                return setattr(self, "tau_syn", tau_to_decay(self.dt, value)[0])
            elif name == "dash_mem":
                return setattr(self, "tau_mem", tau_to_bitshift(self.dt, value)[0])
            elif name == "dash_syn":
                return setattr(self, "tau_syn", tau_to_bitshift(self.dt, value)[0])

        elif self.leak_mode == "decays":
            if name == "tau_mem":
                return setattr(self, "alpha", decay_to_tau(self.dt, value)[0])
            elif name == "tau_syn":
                return setattr(self, "beta", decay_to_tau(self.dt, value)[0])
            elif name == "dash_mem":
                return setattr(self, "alpha", decay_to_bitshift(value)[0])
            elif name == "dash_syn":
                return setattr(self, "beta", decay_to_bitshift(value)[0])

        elif self.leak_mode == "bitshifts":
            if name == "tau_mem":
                return setattr(self, "dash_mem", tau_to_bitshift(self.dt, value)[0])
            elif name == "tau_syn":
                return setattr(self, "dash_syn", tau_to_bitshift(self.dt, value)[0])
            elif name == "alpha":
                return setattr(self, "dash_mem", decay_to_bitshift(value)[0])
            elif name == "beta":
                return setattr(self, "dash_syn", decay_to_bitshift(value)[0])


class LIFTorch(LIFBaseTorch):
    """
    A leaky integrate-and-fire spiking neuron model with a Torch backend

    This module implements the update equations:

    .. math ::

        I_{syn} += S_{in}(t) + S_{rec} \\cdot W_{rec}

        I_{syn} *= \\exp(-dt / \\tau_{syn})

        V_{mem} *= \\exp(-dt / \\tau_{mem})

        V_{mem} += I_{syn} + b + \\sigma \\zeta(t)

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
        forward  method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        Args:
            input_data (torch.Tensor): Data takes the shape of (batch, time_steps, n_synapses)

        Returns:
            torch.Tensor: Out of spikes with the shape (batch, time_steps, Nout)
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
        # dtype=input_data.dtype on every one of these: they are the
        # buffers written into per-timestep below via indexed assignment
        # (`self._record_dict["spikes"][:, t] = spikes`), which silently
        # DOWNCASTS the right-hand side to the destination's dtype. Without
        # this, a module correctly cast to float64/float16 still returned
        # float32 output, because the output itself was accumulated into
        # one of these pre-allocated (implicitly float32) buffers.
        if self._record:
            self._record_dict["vmem"] = torch.zeros(
                n_batches, n_timesteps, self.size_out, dtype=input_data.dtype
            )
            self._record_dict["isyn"] = torch.zeros(
                n_batches,
                n_timesteps,
                self.size_out,
                self.n_synapses,
                dtype=input_data.dtype,
            )
            self._record_dict["irec"] = torch.zeros(
                n_batches,
                n_timesteps,
                self.size_out,
                self.n_synapses,
                dtype=input_data.dtype,
            )

        self._record_dict["spikes"] = torch.zeros(
            n_batches,
            n_timesteps,
            self.size_out,
            device=input_data.device,
            dtype=input_data.dtype,
        )

        noise_zeta = self.noise_std * torch.sqrt(
            torch.tensor(self.dt, dtype=vmem.dtype)
        )

        # - Generate membrane noise trace
        # dtype=vmem.dtype: an omitted dtype on torch.randn defaulted to
        # float32, so `vmem = vmem + ... + noise_ts[:, t, :] + ...` (line
        # ~747 below) silently upcast the whole membrane update to float32
        # from the very first timestep noise participates in -- regardless
        # of what dtype the module was actually cast to.
        noise_ts = noise_zeta * torch.randn(
            (n_batches, n_timesteps, self.size_out),
            device=vmem.device,
            dtype=vmem.dtype,
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

            # - Maintain output spike record
            self._record_dict["spikes"][:, t] = spikes

        # - Update states
        # Registered state attributes are shape-locked to (self.size_out,) --
        # there's no architectural support for persisting a batched state
        # here, so for n_batches > 1 this keeps ONLY batch index 0's final
        # state, discarding the rest. That used to happen completely
        # silently: a later forward()/evolve() call meant to continue the
        # same batch of sequences would incorrectly resume ALL of them from
        # sample 0's state, with no indication anything was lost. Now at
        # least visible, not silent. Same defect, same fix as
        # rockpool/nn/modules/native/lif.py's LIF.evolve().
        if n_batches > 1:
            warnings.warn(
                f"LIFTorch forward()/evolve() was called with batch size {n_batches}, "
                f"but only batch index 0's final state is retained for the next call -- "
                f"state continuation is NOT correct per-sample for batches > 1. Call "
                f"reset_state() between independent batched calls if you don't need "
                f"cross-call continuation.",
                stacklevel=2,
            )
        self.vmem = vmem[0].detach()
        self.isyn = isyn[0].detach()
        self.spikes = spikes[0].detach()

        # - Return output
        return self._record_dict["spikes"]
