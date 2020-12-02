from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter, State, SimulationParameter

# -- Imports
from importlib import util

if (util.find_spec("jax") is None) or (util.find_spec("jaxlib") is None):
    raise ModuleNotFoundError(
        "'Jax' and 'Jaxlib' backend not found. Layers that rely on Jax will not be available."
    )

import jax.numpy as np
import jax
from jax.lax import scan
import jax.random as rand

import numpy as onp

from typing import Optional, Union

FloatVector = Union[float, np.ndarray]


# -- Define useful neuron transfer functions
def H_ReLU(x: FloatVector) -> FloatVector:
    return x


def H_tanh(x: FloatVector) -> FloatVector:
    return np.tanh(x)


def H_sigmoid(x: FloatVector) -> FloatVector:
    return (np.tanh(x) + 1) / 2


# @jax.tree_util.register_pytree_node_class
class RateEulerJax(JaxModule):
    def __init__(
        self,
        shape=None,
        tau: FloatVector = None,
        bias: FloatVector = None,
        w_rec: np.ndarray = None,
        activation_func: str = H_ReLU,
        dt: float = 1e-3,
        noise_std: float = 1e-3,
        rng_key: Optional[int] = None,
        *args,
        **kwargs,
    ):
        # - Work out the shape of this module
        if shape is None:
            assert (
                tau is not None
            ), "You must provide either `shape` or else specify parameters."
            shape = np.array(tau).shape

        # - Call the superclass initialiser
        super().__init__(
            shape=shape, spiking_input=False, spiking_output=False, *args, **kwargs
        )

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        _, rng_key = rand.split(np.array(rng_key, dtype=np.uint32))

        # - Initialise state
        self.activation: np.ndarray = State(shape=self.size_out, init_func=np.zeros)
        self.rng_key: np.ndarray = State(rng_key, init_func=lambda _: rng_key)

        # - Should we be recurrent or FFwd?
        if len(self.shape) == 1:
            # - Feed-forward mode
            if w_rec is not None:
                raise ValueError(
                    "If `shape` is unidimensional, then `w_rec` may not be provided as an argument."
                )

            self.w_rec: float = 0.0

        else:
            # - Recurrent mode
            # - Check that `shape` is correctly specified
            if len(self.shape) > 2:
                raise ValueError("`shape` may not specify more than two dimensions.")

            if self.size_out != self.size_in:
                raise ValueError(
                    "`shape[0]` and `shape[1]` must be equal for a recurrent module."
                )

            self.w_rec: np.ndarray = Parameter(
                w_rec,
                family="weights",
                init_func=lambda s: jax.random.normal(
                    rand.split(self.rng_key)[0], shape=self.shape
                ),
                shape=self.shape,
            )

        # - Set parameters
        self.tau: np.ndarray = Parameter(
            tau,
            "taus",
            init_func=lambda s: np.ones(s) * 100e-3,
            shape=(self.size_out,),
        )
        self.bias: np.ndarray = Parameter(
            bias,
            "bias",
            init_func=lambda s: np.zeros(s),
            shape=(self.size_out,),
        )
        self.dt: float = SimulationParameter(dt)
        self.noise_std: float = SimulationParameter(noise_std)

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
    ):

        dt_tau = self.dt / self.tau
        w_rec = self.w_rec

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inp):
            """
            reservoir_step() - Single step of recurrent reservoir

            :param x:       np.ndarray Current state and activation of reservoir units
            :param inp:    np.ndarray Inputs to each reservoir unit for the current step

            :return:    (new_state, new_activation), (rec_input, activation)
            """
            state, activation = x
            rec_input = np.dot(activation, w_rec)
            state += dt_tau * (-state + inp + self.bias + rec_input)
            activation = H_tanh(state)

            return (state, activation), (rec_input, state, activation)

        # - Evaluate passthrough input layer
        res_inputs = input_data

        # - Compute random numbers for reservoir noise
        key1, subkey = rand.split(self.rng_key)
        noise = self.noise_std * rand.normal(subkey, shape=res_inputs.shape)

        inputs = res_inputs + noise

        # - Use `scan` to evaluate reservoir
        (activation1, _), (rec_inputs, res_state, res_acts) = scan(
            reservoir_step, (self.activation, H_tanh(self.activation)), inputs
        )

        # - Evaluate passthrough output layer
        outputs = res_acts

        new_state = {
            "activation": activation1,
            "rng_key": key1,
        }

        record_dict = {
            "rec_inputs": rec_inputs,
            "res_state": res_state,
            "res_acts": res_acts,
        }

        return outputs, new_state, record_dict
