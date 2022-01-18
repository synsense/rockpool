"""
Rate dynamics module with torch backend
"""

from rockpool.nn.modules.torch.torch_module import TorchModule
import rockpool.typehints as rt
import rockpool.parameters as rp

import torch
import torch.nn.functional as F
import torch.nn.init as init

from typing import Optional, Union, Callable, Tuple, Any

__all__ = ["RateTorch"]

relu = lambda x, t: torch.clip(x - t, 0, torch.inf)


class RateTorch(TorchModule):
    """
    Rate dynamic neuron, with a torch backend

    This module implements the update equations:

    .. math::

        X = X + i(t) + W_{rec} H(X) + \sigma \zeta_t
        X = X * \exp(-dt / \tau)

        H(x, t) = (x - t) * ((x - t) > 0)
    
    """

    def __init__(
        self,
        shape: Union[tuple, int],
        tau: Optional[rt.FloatVector] = None,
        bias: Optional[rt.FloatVector] = None,
        threshold: Optional[rt.FloatVector] = None,
        has_rec: bool = False,
        w_rec: Optional[rt.FloatVector] = None,
        weight_init_func: Optional[Callable] = lambda s: init.kaiming_uniform_(
            torch.empty(s)
        ),
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
            tau (Tensor): Time constant of each unit ``(N,)``. Default: 10ms for each unit
            bias (Tensor): Bias current for each neuron ``(N,)``. Default: 0. for each unit
            threshold (Tensor): Threshold for each neuron ``(N,)``. Default: 0. for each unit
            has_rec (bool): Iff ``True``, module includes recurrent connectivity. Default: ``False``, module is feed-forward
            w_rec (Tensor): If ``has_rec``, can be used to provide concrete initialisation data for recurrent weights.
            weight_init_func (Callable): Weight initialisation function, if ``has_rec``. Default: Kaiming initialisation
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

        # - Initialise recurrent weights
        w_rec_shape = (self.size_out, self.size_in)
        if has_rec:
            self.w_rec: rt.P_tensor = rp.Parameter(
                w_rec, shape=w_rec_shape, init_func=weight_init_func, family="weights",
            )
            """ (Tensor) Recurrent weights `(Nout, Nin)` """
        else:
            if w_rec is not None:
                raise ValueError("`w_rec` may not be provided if `has_rec` is `False`")

        self.noise_std: rt.P_float = rp.SimulationParameter(noise_std)
        """ (float) Noise std.dev. injected onto the membrane of each unit during evolution """

        # - To-float-tensor conversion utility
        to_float_tensor = lambda x: torch.tensor(x).float()

        self.tau: rt.P_tensor = rp.Parameter(
            tau,
            family="taus",
            shape=[(self.size_out,), ()],
            init_func=lambda s: torch.ones(s) * 10e-3,
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

        self.activation_func: rt.P_Callable = rp.SimulationParameter(activation_func)
        """ (Callable) Activation function for the units """

        self.x: rt.P_tensor = rp.State(
            shape=self.size_out, init_func=torch.zeros, cast_fn=to_float_tensor
        )
        """ (Tensor) Unit state `(Nout,)` """

        self.acts: rt.P_tensor = rp.State(
            shape=self.size_out, init_func=torch.zeros, cast_fn=to_float_tensor
        )
        """ (Tensor) Activations `(Nout,)` """

        self._record = False

    def evolve(self, data, record: bool = False) -> Tuple[Any, Any, Any]:
        self._record = record
        out, state, _ = super().evolve(data, record)

        record_dict = (
            {"rec_input": self._rec_input, "x": self._state, "act": self._act,}
            if record
            else {}
        )

        return out, state, record_dict

    def forward(self, data, *args, **kwargs) -> torch.Tensor:
        # - Verify input data shape
        if len(data.shape) == 2:
            data = torch.unsqueeze(data, 0)
        (n_batches, time_steps, n_inputs) = data.shape

        if n_inputs != self.size_in:
            raise ValueError(
                "Input has wrong input dimension. It is {}, must be {}".format(
                    n_inputs, self.size_in
                )
            )

        # - Replicate states out by batches
        neur_state = torch.ones(n_batches, self.size_out).to(data.device) * self.x
        bias = torch.ones(n_batches, self.size_out).to(data.device) * self.bias
        acts = torch.ones(n_batches, self.size_out).to(data.device) * self.acts
        threshold = (
            torch.ones(n_batches, self.size_out).to(data.device) * self.threshold
        )

        # - Set up state record and output
        if self._record:
            self._state = torch.zeros(
                n_batches, time_steps, self.size_out, device=data.device
            )
            self._rec_input = torch.zeros(
                n_batches, time_steps, self.size_out, device=data.device
            )

        self._act = torch.zeros(
            n_batches, time_steps, self.size_out, device=data.device
        )

        alpha = torch.exp(-self.dt / self.tau)
        noise_zeta = self.noise_std * torch.sqrt(torch.tensor(self.dt))

        # - Loop over time
        for t in range(time_steps):
            # - Decay state
            neur_state *= alpha

            # - Integrate input, bias, noise
            neur_state += data[:, t] + bias

            if self.noise_std > 0.0:
                neur_state += noise_zeta * torch.randn(
                    self.size_out, device=data.device
                )

            # - Recurrent input
            if hasattr(self, "w_rec"):
                rec_inputs = F.linear(acts, self.w_rec.T)
                neur_state = neur_state + rec_inputs
            else:
                rec_inputs = 0.0

            # - Record state
            if self._record:
                self._state[:, t, :] = neur_state
                self._rec_input[:, t, :] = rec_inputs

            # - Compute unit activation
            acts = self.activation_func(neur_state, threshold)
            self._act[:, t, :] = acts

        # - Update states
        self.x = neur_state[0].detach()
        self.acts = acts[0].detach()

        # - Return activations
        return self._act
