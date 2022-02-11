"""
Rate dynamics module with torch backend
"""

from rockpool.nn.modules.torch.torch_module import TorchModule
from ..native.linear import unit_eigs, kaiming
import rockpool.typehints as rt
import rockpool.parameters as rp

from rockpool.graph import (
    RateNeuronWithSynsRealValue,
    LinearWeights,
    GraphModuleBase,
    as_GraphHolder,
)

import torch
import torch.nn.functional as F
import torch.nn.init as init

from typing import Optional, Union, Callable, Tuple, Any

__all__ = ["RateTorch"]

relu = lambda x, t: torch.clip(x - t, 0, None)


class RateTorch(TorchModule):
    """
    Encapsulates a population of rate neurons, supporting feed-forward and recurrent modules, with a Toch backend

    Examples:
        Instantiate a feed-forward module with 8 neurons:

        >>> mod = RateTorch(8,)
        RateEulerJax 'None' with shape (8,)

        Instantiate a recurrent module with 12 neurons:

        >>> mod_rec = RateTorch(12, has_rec = True)
        RateEulerJax 'None' with shape (12,)

        Instantiate a feed-forward module with defined time constants:

        >>> mod = RateTorch(7, tau = torch.arange(7,) * 10e-3)
        RateEulerJax 'None' with shape (7,)

    This module implements the update equations:

    .. math::

        \dot{X} = -X + i(t) + W_{rec} H(X) + bias + \sigma \zeta_t
        X = X + \dot{x} * dt / \tau

        H(x, t) = relu(x, t) = (x - t) * ((x - t) > 0)
    """

    def __init__(
        self,
        shape: Union[tuple, int],
        tau: Optional[rt.FloatVector] = None,
        bias: Optional[rt.FloatVector] = None,
        threshold: Optional[rt.FloatVector] = None,
        has_rec: bool = False,
        w_rec: Optional[rt.FloatVector] = None,
        weight_init_func: Callable = unit_eigs,
        activation_func: Callable = relu,
        noise_std: float = 0.0,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate a module with rate dynamics

        Args:
            shape (Union[tuple, int]): The number of units in this module
            tau (Tensor): Time constant of each unit ``(N,)``. Default: 20ms for each unit
            bias (Tensor): Bias current for each neuron ``(N,)``. Default: 0. for each unit
            threshold (Tensor): Threshold for each neuron ``(N,)``. Default: 0. for each unit
            has_rec (bool): Iff ``True``, module includes recurrent connectivity. Default: ``False``, module is feed-forward
            w_rec (Tensor): If ``has_rec``, can be used to provide concrete initialisation data for recurrent weights.
            weight_init_func (Callable): A function used to initialise the recurrent weights, if used. Default: :py:func:`.unit_eigs`; initialise such that recurrent feedback has eigenvalues distributed within the unit circle.
            activation_func (Callable): Actiavtion function. Default: ReLU
            noise_std (float): Std. dev of noise after 1s, added to neuron state. Defualt: ``0.``, no noise.
            dt (float): Simulation time constant in seconds
        """
        # - Call super-class init
        super().__init__(
            shape=shape, spiking_input=False, spiking_output=False, *args, **kwargs
        )

        self.dt: rt.P_float = rp.SimulationParameter(dt)
        """ (float) Euler simulator time-step in seconds"""

        # - To-float-tensor conversion utility
        to_float_tensor = lambda x: torch.tensor(x).float()

        # - Initialise recurrent weights
        w_rec_shape = (self.size_out, self.size_in)
        if has_rec:
            self.w_rec: rt.P_tensor = rp.Parameter(
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

        self.noise_std: rt.P_float = rp.SimulationParameter(noise_std)
        """ (float) Noise std.dev. injected onto the membrane of each unit during evolution """

        self.tau: rt.P_tensor = rp.Parameter(
            tau,
            family="taus",
            shape=[(self.size_out,), ()],
            init_func=lambda s: torch.ones(s) * 20e-3,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Unit time constants `(Nout,)` or `()` """

        self.bias: rt.P_tensor = rp.Parameter(
            bias,
            family="biases",
            shape=[(self.size_out,), ()],
            init_func=lambda s: torch.zeros(*s),
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Unit biases `(Nout,)` or `()` """

        self.threshold: rt.P_tensor = rp.Parameter(
            threshold,
            family="thresholds",
            shape=[(self.size_out,), ()],
            init_func=lambda s: torch.zeros(*s),
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Unit thresholds `(Nout,)` or `()` """

        self.act_fn: rt.P_Callable = rp.SimulationParameter(activation_func)
        """ (Callable) Activation function for the units """

        self.x: rt.P_tensor = rp.State(
            shape=self.size_out, init_func=torch.zeros, cast_fn=to_float_tensor
        )
        """ (Tensor) Unit state `(Nout,)` """

        self._record = False

    def evolve(self, data, record: bool = False) -> Tuple[Any, Any, Any]:
        self._record = record
        out, state, _ = super().evolve(data, record)

        record_dict = {"rec_input": self._rec_input, "x": self._state} if record else {}

        return out, state, record_dict

    def forward(self, data, *args, **kwargs) -> torch.Tensor:
        # - Perform auto-batching
        data, (neur_state,) = self._auto_batch(data, (self.x,))
        (n_batches, time_steps, _) = data.shape

        act = self.act_fn(neur_state, self.threshold)

        # - Set up state record and output
        if self._record:
            self._rec_input = torch.zeros(
                n_batches, time_steps, self.size_out, device=data.device
            )

        self._state = torch.zeros(
            n_batches, time_steps, self.size_out, device=data.device
        )

        alpha = self.dt / self.tau
        noise_zeta = self.noise_std * torch.sqrt(torch.tensor(self.dt))

        # - Loop over time
        for t in range(time_steps):
            # - Integrate input, bias, noise
            dstate = -neur_state + data[:, t] + self.bias

            if self.noise_std > 0.0:
                dstate = dstate + noise_zeta * torch.randn(
                    self.size_out, device=data.device
                )

            # - Recurrent input
            if hasattr(self, "w_rec"):
                rec_inputs = F.linear(act, self.w_rec.T)
                dstate = dstate + rec_inputs
            else:
                rec_inputs = 0.0

            # - Accumulate state
            neur_state = neur_state + dstate * alpha

            # - Record state
            if self._record:
                self._rec_input[:, t, :] = rec_inputs

            self._state[:, t, :] = neur_state

            # - Compute unit activation
            act = self.act_fn(neur_state, self.threshold)

        # - Update states
        self.x = neur_state[0].detach()

        # - Return activations
        return self.act_fn(self._state, self.threshold)

    def as_graph(self) -> GraphModuleBase:
        # - Generate a GraphModule for the neurons
        neurons = RateNeuronWithSynsRealValue._factory(
            self.size_in,
            self.size_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
            self.tau,
            self.bias,
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
                self.w_rec,
            )

        # - Return a graph containing neurons and optional weights
        return as_GraphHolder(neurons)
