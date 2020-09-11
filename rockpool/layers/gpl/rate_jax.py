##
# Firing rate reservoir with jax back-end
#
# Includes `RecRateEulerJax` - jax-backed firing rate reservoir
# Includes `RecRateEulerJax_IO` - jax-backed firing rate reservoir with input / output weighting
# Includes `ForceRateEulerJax_IO` - jax-backed firing rate FORCE reservoir with input / output weighting
# Includes `FFRateEulerJax` - jax-backed firing rate feed-forward layer
#
# `RecRateEulerJax` and `RecRateEulerJax_IO` are standard reservoirs, with input, recurrent and output layers
# `ForceRateEulerJax` is a driven reservoir, where "recurrent" inputs are inserted from an external source.
#       Used in reservoir transfer.
##

# -- Imports
from importlib import util

if (util.find_spec("jax") is None) or (util.find_spec("jaxlib") is None):
    raise ModuleNotFoundError(
        "'Jax' and 'Jaxlib' backend not found. Layers that rely on Jax will not be available."
    )

import jax.numpy as np
import jax
from jax import jit
from jax.lax import scan
import jax.random as rand

import numpy as onp

from typing import Optional, Tuple, Callable, Union, Dict, List, Any

from rockpool.layers.layer import Layer
from rockpool.layers.training.gpl.jax_trainer import JaxTrainer
from ...timeseries import TimeSeries, TSContinuous


# -- Define module exports
__all__ = [
    "RecRateEulerJax",
    "RecRateEulerJax_IO",
    "ForceRateEulerJax_IO",
    "FFRateEulerJax",
    "H_ReLU",
    "H_tanh",
    "H_sigmoid",
]

FloatVector = Union[float, np.ndarray]
Params = Dict
State = np.ndarray

# -- Define useful neuron transfer functions
def H_ReLU(x: FloatVector) -> FloatVector:
    return np.clip(x, 0, None)


def H_tanh(x: FloatVector) -> FloatVector:
    return np.tanh(x)


def H_sigmoid(x: FloatVector) -> FloatVector:
    return (np.tanh(x) + 1) / 2


# -- Generators for compiled evolution functions


def _get_rec_evolve_jit(
    H: Callable[[float], float]
) -> Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        List[int],
        float,
    ],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any],
]:
    """
    Return a compiled raw reservoir evolution function

    :param Callable[[float], float] H:   Neuron activation function
    :return Callable:     f(x0, w_in, w_recurrent, w_out, bias, tau, inputs, noise_std, key, dt) -> (x, res_inputs, rec_inputs, res_acts, outputs)
    """

    @jit
    def rec_evolve_jit(
        x0: np.ndarray,
        w_in: FloatVector,
        w_recurrent: FloatVector,
        w_out: FloatVector,
        bias: FloatVector,
        tau: FloatVector,
        inputs: np.ndarray,
        noise_std: float,
        key,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Compiled recurrent evolution function

        :param np.ndarray x0:       Initial state of reservoir layer [N]
        :param np.ndarray w_in:     Input weights [IxN]
        :param np.ndarray w_recurrent:     Recurrent weights [NxN]
        :param np.ndarray w_out:    Output weights [NxO]
        :param np.ndarray bias:     Bias values of reservoir units [N]
        :param np.ndarray tau:      Time constants of reservoir units [N]
        :param np.ndarray inputs:   Input time series [TxN]
        :param np.ndarray force:    Driving time series injected into reservoir units instead of recurrent activity [TxN]
        :param float noise_std:     Standard deviation of noise injected into reservoir units
        :param Any key:             Jax RNG key to use in noise generation
        :param float dt:            Time step for forward Euler solver

        :return:    (x, res_inputs, rec_inputs, res_acts, outputs, key)
                x:          np.ndarray State of
                res_inputs: np.ndarray Time series of weighted external inputs to each reservoir unit [TxN]
                rec_inputs: np.ndarray Time series of recurrent inputs to each reservoir unit [TxN]
                res_acts:   np.ndarray Time series of reservoir unit activities [TxN]
                outputs:    np.ndarray Time series of output layer values [TxO]
                key:        Any        New RNG key
        """
        # - Pre-compute dt/tau
        dt_tau = dt / tau

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inp):
            """
            reservoir_step() - Single step of recurrent reservoir

            :param x:       np.ndarray Current state and activation of reservoir units
            :param inp:    np.ndarray Inputs to each reservoir unit for the current step

            :return:    (new_state, new_activation), (rec_input, activation)
            """
            state, activation = x
            rec_input = np.dot(activation, w_recurrent)
            state += dt_tau * (-state + inp + bias + rec_input)
            activation = H(state)

            return (state, activation), (rec_input, activation)

        # - Evaluate passthrough input layer
        res_inputs = np.dot(inputs, w_in)

        # - Compute random numbers for reservoir noise
        key1, subkey = rand.split(key)
        noise = noise_std * rand.normal(subkey, shape=res_inputs.shape)

        inputs = res_inputs + noise

        # - Use `scan` to evaluate reservoir
        (state, __), (rec_inputs, res_acts) = scan(reservoir_step, (x0, H(x0)), inputs)

        # - Evaluate passthrough output layer
        outputs = np.dot(res_acts, w_out)

        return state, res_inputs, rec_inputs, res_acts, outputs, key1

    return rec_evolve_jit


def _get_rec_evolve_directly_jit(H: Callable[[float], float]):
    @jit
    def rec_evolve_jit(
        x0: np.ndarray,
        w_recurrent: np.ndarray,
        w_out: np.ndarray,
        bias: np.ndarray,
        tau: np.ndarray,
        inputs: np.ndarray,
        noise_std: float,
        key,
        dt: float,
    ):
        # - Pre-compute dt/tau
        dt_tau = dt / tau

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inps):
            inp, rand = inps
            activation = H(x)
            rec_input = np.dot(activation, w_recurrent)
            dx = dt_tau * (-x + inp + bias + rand + rec_input)

            return x + dx, (rec_input, activation)

        # - Compute random numbers for reservoir noise
        __all__, subkey = rand.split(key)
        noise = noise_std * rand.normal(subkey, shape=(inputs.shape[0], np.size(x0)))

        # - Use `scan` to evaluate reservoir
        x, (rec_inputs, res_acts) = scan(reservoir_step, x0, (inputs, noise))

        # - Evaluate passthrough output layer
        outputs = np.dot(res_acts, w_out)

        return x, inputs, rec_inputs, res_acts, outputs

    return rec_evolve_jit


def _get_force_evolve_jit(
    H: Callable[[float], float]
) -> Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        List[int],
        float,
    ],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any],
]:
    """
    Return a compiled raw reservoir evolution function

    :param Callable[[float], float] H:   Neuron activation function
    :return Callable:     f(x0, w_in, w_out, bias, tau, inputs, forces, noise_std, key, dt) -> (x, res_inputs, res_acts, outputs)
    """

    @jit
    def force_evolve_jit(
        x0: np.ndarray,
        w_in: FloatVector,
        w_out: FloatVector,
        bias: FloatVector,
        tau: FloatVector,
        inputs: np.ndarray,
        force: np.ndarray,
        noise_std: float,
        key: Any,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Compiled recurrent evolution function

        :param np.ndarray x0:       Initial state of forced layer [N]
        :param np.ndarray w_in:     Input weights [IxN]
        :param np.ndarray w_out:    Output weights [NxO]
        :param np.ndarray bias:     Bias values of reservoir units [N]
        :param np.ndarray tau:      Time constants of reservoir units [N]
        :param np.ndarray inputs:   Input time series [TxN]
        :param np.ndarray force:    Driving time series injected into reservoir units instead of recurrent activity [TxN]
        :param float noise_std:     Standard deviation of noise injected into reservoir units
        :param List[int] key:       Jax RNG key to use in noise generation
        :param float dt:            Time step for forward Euler solver

        :return:    (x, res_inputs, res_acts, outputs, key)
                x:          np.ndarray State of
                res_inputs: np.ndarray Time series of weighted external inputs to each reservoir unit [TxN]
                res_acts:   np.ndarray Time series of reservoir unit activities [TxN]
                outputs:    np.ndarray Time series of output layer values [TxO]
        """
        # - Pre-compute dt/tau
        dt_tau = dt / tau

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inp):
            x += dt_tau * (-x + inp + bias)
            activation = H(x)

            return x, activation

        # - Evaluate passthrough input layer
        res_inputs = np.dot(inputs, w_in)

        # - Compute random numbers for reservoir noise
        key1, subkey = rand.split(key)
        noise = noise_std * rand.normal(subkey, shape=res_inputs.shape)

        inputs = res_inputs + noise + force

        # - Use `scan` to evaluate reservoir
        x, res_acts = scan(reservoir_step, x0, inputs)

        # - Evaluate passthrough output layer
        outputs = np.dot(res_acts, w_out)

        return x, res_inputs, res_acts, outputs, key1

    return force_evolve_jit


# -- Recurrent reservoir


class RecRateEulerJax(JaxTrainer, Layer):
    """
    ``JAX``-backed firing-rate recurrent layer

    `.RecRateEulerJax` implements a recurrent reservoir with input and output weighting, using a ``JAX``-implemented solver as a back end. The design permits gradient-based learning of weights, biases and time constants using `jax.grad`.

    `.RecRateEulerJax` is compatible with the `.layers.training.jax_trainer` module.

    .. rubric:: Dynamics

    The layer implements the dynamics

    .. math:: \\tau \\cdot \\dot{x} + x = I(t) + W_{rec} \\cdot H(x) + b + \\sigma \\cdot \\zeta(t)

    where :math:`\\tau`` is the neuron time constants; :math:`x` is the N-dimensional state of the neurons; :math:`I(t)` is the N-dimensional input signal; :math:`W_{rec}` is the NxN-dimensional matrix of recurrent synaptic connections; :math:`b` is the N-dimensional vector of bias currents for the neurons; and :math:`\\sigma \\cdot \\zeta(t)` is a white noise process with std. dev :math:`\\sigma`.

    The outputs of the layer are given by :math:`H(x)`, where :math:`H(x)` is a neuron transfer function. By default, :math:`H(x)` is the linear-threshold function

    .. math:: H_{ReLU}(x) = \\text{max}(x, 0)

    .. rubric:: Training

    `.RecRateEulerJax` implements the `.JaxTrainedLayer` training interface. See the documentation for :py:meth:`.JaxTrainedLayer.train_output_target` for information about training these layers.
    """

    def __init__(
        self,
        weights: np.ndarray,
        tau: np.ndarray,
        bias: np.ndarray,
        noise_std: float = 0.0,
        activation_func: Union[str, Callable[[FloatVector], FloatVector]] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        ``JAX``-backed firing rate reservoir, with input and output weighting

        :param np.ndarray weights:                  Recurrent weights [NxN]
        :param np.ndarray tau:                      Time constants [N]
        :param np.ndarray bias:                     Bias values [N]
        :param float noise_std:                     White noise standard deviation applied to reservoir neurons. Default: ``0.0``
        :param Union[str, Callable[[FloatVector], float]] activation_func:   Neuron transfer function f(x: float) -> float. Must be vectorised. Default: H_ReLU. Can be specified as a string: ['relu', 'tanh', 'sigmoid']
        :param Optional[float] dt:                  Reservoir time step. Default: ``np.min(tau) / 10.0``
        :param Optional[str] name:                  Name of the layer. Default: ``None``
        :param Optional[Jax RNG key] rng_key:       Jax RNG key to use for noise. Default: Internally generated
        """

        # - Everything should be 2D
        weights = np.atleast_2d(weights)

        # transform to np.array if necessary
        tau = np.array(tau)
        bias = np.array(bias)

        # - Assign name such that Exceptions in setters work correctly
        if name is None:
            self.name = "unnamed"
        else:
            self.name = name

        # - Get information about network size
        self._size_in = weights.shape[0]
        self._size = weights.shape[1]
        self._size_out = weights.shape[1]

        # -- Set properties
        self.w_recurrent = weights
        self.tau = tau
        self.bias = bias
        self.H = activation_func

        if dt is None:
            dt = np.min(tau) / 10.0

        # - Call super-class initialisation
        super().__init__(weights, dt, noise_std, name, *args, **kwargs)

        # - Get compiled evolution function
        self._evolve_jit = _get_rec_evolve_jit(self._H)
        self._evolve_directly_jit = _get_rec_evolve_directly_jit(self._H)

        # - Reset layer state
        self._state = np.array(self._size)
        self.reset_all()

        # - Reset "last evolution" attributes
        self.res_inputs_last_evolution = TSContinuous()
        self.rec_inputs_last_evolution = TSContinuous()
        self.res_acts_last_evolution = TSContinuous()

        # - Set unit internal input and output weights, for compatibility with later layers
        if not hasattr(self, "_w_in"):
            self._w_in: FloatVector = 1.0
        if not hasattr(self, "_w_out"):
            self._w_out: FloatVector = 1.0

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        _, self._rng_key = rand.split(np.array(rng_key, dtype=np.uint32))

    def _pack(self) -> Params:
        """
        Return a packed form of the tunable parameters for this layer

        :return Params: params: All parameters as a Dict
        """
        return {
            "w_in": self._w_in,
            "w_recurrent": self._weights,
            "w_out": self._w_out,
            "bias": self._bias,
            "tau": self._tau,
        }

    def _unpack(self, params: Params):
        """
        Set the parameters for this layer, given a parameter dictionary

        :param Params params:  Set of parameters for this layer
        """
        (self._w_in, self._weights, self._w_out, self._bias, self._tau,) = (
            params["w_in"],
            params["w_recurrent"],
            params["w_out"],
            params["bias"],
            params["tau"],
        )

    def randomize_state(self):
        """
        Randomize the internal state of the layer.
        """

        # def split_sample(key: Any, shape: Tuple[int]) -> Tuple[Any, np.ndarray]:
        #     key1, subkey = rand.split(key)
        #     sample = rand.normal(subkey, shape=shape)
        #     return key1, sample
        #
        # ss = jit(split_sample, static_argnums=1)
        #
        # # - Randomise the state
        # self._rng_key, self._state = ss(
        #     self._rng_key, (self._size,)
        # )
        self._state = onp.random.randn(
            self._size,
        )

    @property
    def _evolve_functional(self):
        """
        Return a functional form of the evolution function for this layer

        Returns a function ``evol_func`` with the signature::

            def evol_func(params, state, inputs) -> (outputs, new_state):

        :return Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State]]:
        """

        def evol_func(
            params: Params,
            state: State,
            inputs: np.ndarray,
        ):
            # - Call the jitted evolution function for this layer
            (
                new_state,
                res_inputs,
                rec_inputs,
                res_acts,
                outputs,
                key1,
            ) = self._evolve_jit(
                state,
                params["w_in"],
                params["w_recurrent"],
                params["w_out"],
                params["bias"],
                params["tau"],
                inputs,
                self._noise_std,
                self._rng_key,
                self._dt,
            )

            # - Maintain RNG key, if not under compilation
            if not isinstance(key1, jax.core.Tracer):
                self._rng_key = key1

            # - Return the outputs from this layer, and the final layer state
            states_t = {
                "res_inputs": res_inputs,
                "rec_inputs": rec_inputs,
                "res_acts": res_acts,
                "outputs": outputs,
            }
            return outputs, new_state, states_t

        # - Return the evolution function
        return evol_func

    def get_output_from_state(self, state):
        activity = self._H(state)
        output = np.dot(activity, self._w_out)
        rec_input = np.dot(activity, self._weights)

        return output, activity, rec_input

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TSContinuous:
        """
        Evolve the reservoir state

        :param ts_input:        TSContinuous Input time series
        :param duration:        float Duration of evolution in seconds
        :param num_timesteps:   int Number of time steps to evolve (based on self.dt)

        :return: ts_output:     TSContinuous Output time series
        """

        # - Prepare time base and inputs
        time_base_inp, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call raw evolution function
        (
            self._state,
            res_inputs,
            rec_inputs,
            res_acts,
            outputs,
            self._rng_key,
        ) = self._evolve_jit(
            self._state,
            self._w_in,
            self._weights,
            self._w_out,
            self._bias,
            self._tau,
            inps,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        # - Store evolution time series
        self.res_inputs_last_evolution = TSContinuous.from_clocked(
            onp.array(res_inputs), t_start=self.t, dt=self.dt, name="Reservoir inputs"
        )
        self.rec_inputs_last_evolution = TSContinuous.from_clocked(
            onp.array(rec_inputs), t_start=self.t, dt=self.dt, name="Recurrent inputs"
        )
        self.res_acts_last_evolution = TSContinuous.from_clocked(
            onp.array(res_acts), t_start=self.t, dt=self.dt, name="Layer activations"
        )

        # - Wrap outputs as time series
        ts_output = TSContinuous.from_clocked(
            onp.array(outputs), t_start=self.t, dt=self.dt, name="Outputs"
        )

        # - Increment timesteps
        self._timestep += num_timesteps

        return ts_output

    def to_dict(self) -> dict:
        """
        Convert the parameters of this class to a dictionary

        :return dict:
        """
        config = {}
        config["class_name"] = "RecRateEulerJax"
        config["weights"] = onp.array(self.weights).tolist()
        config["tau"] = onp.array(self.tau).tolist()
        config["bias"] = onp.array(self.bias).tolist()
        config["rng_key"] = onp.array(self._rng_key).tolist()
        config["noise_std"] = (
            self.noise_std if type(self.noise_std) is float else self.noise_std.tolist()
        )
        config["dt"] = self.dt
        config["name"] = self.name

        # - Check for a supported activation function
        if not (self._H is H_ReLU or self._H is H_tanh or self._H is H_sigmoid):
            raise RuntimeError(
                self.start_print
                + "Only models using ReLU, tanh or sigmoid activation functions are saveable."
            )

        # - Encode the activation function as a string
        if self._H is H_ReLU:
            config["activation_func"] = "relu"
        elif self._H is H_tanh:
            config["activation_func"] = "tanh"
        elif self._H is H_sigmoid:
            config["activation_func"] = "sigmoid"
        else:
            raise (Exception)
        return config

    @property
    def H(self):
        """ (Callable) Activation function used by the neurons in this layer """
        return self._H

    @H.setter
    def H(self, value):
        if type(value) is str:
            if value in ["relu", "ReLU", "H_ReLU"]:
                self._H = H_ReLU
            elif value in ["tanh", "TANH", "H_tanh"]:
                self._H = H_tanh
            elif value in [
                "sigmoid",
                "sig",
                "H_sigmoid",
                "H_sig",
                "SIGMOID",
                "SIG",
                "H_SIGMOID",
                "H_SIG",
            ]:
                self._H = H_sigmoid
            else:
                raise ValueError(
                    'The activation function must be one of ["relu", "tanh", "sigmoid"]'
                )
        else:
            # - Test the activation function
            try:
                ret = value(np.array([0.1, 0.1]))
            except:
                raise TypeError(
                    "The activation function must be a Callable[[FloatVector], FloatVector]"
                )

            if np.size(ret) != 2:
                raise ValueError(
                    self.start_print
                    + "The activation function must return an array the same size as the input"
                )

            if isinstance(ret, tuple):
                raise TypeError(
                    self.start_print,
                    "The activation function must not return multiple arguments",
                )

            # - Assign the activation function
            self._H = value

    @property
    def w_recurrent(self) -> np.ndarray:
        """ (np.ndarray) [NxN] recurrent weights """
        return onp.array(self._weights)

    @w_recurrent.setter
    def w_recurrent(self, value: np.ndarray):
        if np.ndim(value) != 2:
            raise ValueError(self.start_print + "`w_recurrent` must be 2D")

        if value.shape != (
            self._size,
            self._size,
        ):
            raise ValueError(
                self.start_print
                + "`w_recurrent` must be [{:d}, {:d}]".format(self._size, self._size)
            )

        self._weights = np.array(value).astype("float")

    @property
    def tau(self) -> np.ndarray:
        return onp.array(self._tau)

    @tau.setter
    def tau(self, value: np.ndarray):
        # - Replicate `tau` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        if np.size(value) != self._size:
            raise ValueError(
                self.start_print
                + "`tau` must have {:d} elements or be a scalar".format(self._size)
            )

        self._tau = np.reshape(value, self._size).astype("float")

    @property
    def bias(self) -> np.ndarray:
        return onp.array(self._bias)

    @bias.setter
    def bias(self, value: np.ndarray):
        # - Replicate `bias` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        if np.size(value) != self._size:
            raise ValueError(
                self.start_print
                + "`bias` must have {:d} elements or be a scalar".format(self._size)
            )

        self._bias = np.reshape(value, self._size).astype("float")

    @property
    def dt(self) -> float:
        return onp.array(self._dt).item(0)

    @dt.setter
    def dt(self, value: float):
        # - Ensure dt is numerically stable
        tau_min = np.min(self.tau) / 10.0
        if value is None:
            value = tau_min

        if value < tau_min:
            raise ValueError(
                self.start_print + "`tau` must be at least {:.2e}".format(tau_min)
            )

        self._dt = np.array(value).astype("float")


class RecRateEulerJax_IO(RecRateEulerJax):
    """
    ``JAX``-backed firing-rate recurrent layer, with input and output weights

    `.RecRateEulerJax_IO` implements a recurrent reservoir with input and output weighting, using a ``JAX``-implemented solver as a back end. The design permits gradient-based learning of weights, biases and time constants using `jax.grad`.

    `.RecRateEulerJax_IO` is compatible with the `.layers.training.jax_trainer` module.

    .. rubric:: Dynamics

    The layer implements the dynamics

    .. math:: \\tau \\cdot \\dot{x} + x = W_{in} \\dot I(t) + W_{rec} \\cdot H(x) + b + \\sigma \\cdot \\zeta(t)

    where :math:`\\tau`` is the neuron time constants; :math:`x` is the N-dimensional state of the neurons; :math:`I(t)` is the I-dimensional input vector; :math:`W_{in}` is the [IxN] matrix of input weights; :math:`W_{rec}` is the NxN-dimensional matrix of recurrent synaptic connections; :math:`b` is the N-dimensional vector of bias currents for the neurons; and :math:`\\sigma \\cdot \\zeta(t)` is a white noise process with std. dev :math:`\\sigma`.

    The outputs of the layer are given by :math:`W_{out} \\cdot H(x)`, where :math:`W_{out}` is the NxM output weight matrix, and :math:`H(x)` is a neuron transfer function. By default, :math:`H(x)` is the linear-threshold function

    .. math:: H_{ReLU}(x) = \\text{max}(x, 0)

    .. rubric:: Training

    `.RecRateEulerJax` implements the `.JaxTrainedLayer` training interface. See the documentation for :py:meth:`.JaxTrainedLayer.train_output_target` for information about training these layers.
    """

    def __init__(
        self,
        w_in: np.ndarray,
        w_recurrent: np.ndarray,
        w_out: np.ndarray,
        tau: FloatVector,
        bias: FloatVector,
        noise_std: float = 0.0,
        activation_func: Callable[[FloatVector], FloatVector] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        RecRateEulerJax_IO - ``JAX``-backed firing rate reservoir, with input and output weighting

        :param np.ndarray w_in:                     Input weights [IxN]
        :param np.ndarray w_recurrent:              Recurrent weights [NxN]
        :param np.ndarray w_out:                    Output weights [NxO]
        :param np.ndarray tau:                      Time constants [N]
        :param np.ndarray bias:                     Bias values [N]
        :param float noise_std:           White noise standard deviation applied to reservoir neurons. Default: ``0.0``
        :param Callable[[FloatVector], float] activation_func:   Neuron transfer function f(x: float) -> float. Must be vectorised. Default: H_ReLU
        :param Optional[float] dt:                  Reservoir time step. Default: ``np.min(tau) / 10.0``
        :param Optional[str] name:                  Name of the layer. Default: ``None``
        :param Optional[Jax RNG key] rng_key        Jax RNG key to use for noise. Default: Internally generated
        """

        # - Everything should be 2D
        w_in = np.atleast_2d(w_in)
        w_recurrent = np.atleast_2d(w_recurrent)
        w_out = np.atleast_2d(w_out)

        # transform to np.array if necessary
        tau = np.array(tau)
        bias = np.array(bias)

        # - Assign name such that Exceptions in setters work correctly
        if name is None:
            self.name = "unnamed"
        else:
            self.name = name

        # - Get information about network size
        self._size_in = w_in.shape[0]
        self._size = w_in.shape[1]
        self._size_out = w_out.shape[1]

        # -- Set properties
        self.w_in = w_in
        self.w_recurrent = w_recurrent
        self.w_out = w_out
        self.tau = tau
        self.bias = bias
        self._H = activation_func

        if dt is None:
            dt = np.min(tau) / 10.0

        # - Call super-class initialisation
        super().__init__(
            w_recurrent,
            tau,
            bias,
            noise_std,
            activation_func,
            dt,
            name,
            rng_key,
            *args,
            **kwargs,
        )

        # - Correct layer size
        self._size_in = w_in.shape[0]
        self._size_out = w_out.shape[1]

    def to_dict(self) -> dict:
        """
        Convert the parameters of this class to a dictionary

        :return dict:
        """
        # - Get base dictionary
        config = super().to_dict()
        config.pop("weights")

        # - Include class-specific aspects
        config.update(
            {
                "class_name": "RecRateEulerJax_IO",
                "w_in": onp.array(self.w_in).tolist(),
                "w_recurrent": onp.array(self.w_recurrent).tolist(),
                "w_out": onp.array(self.w_out).tolist(),
            }
        )

        # - Return configuration
        return config

    @property
    def w_in(self) -> np.ndarray:
        """ (np.ndarray) [IxN] input weights """
        return onp.array(self._w_in)

    @w_in.setter
    def w_in(self, value: np.ndarray):
        if np.ndim(value) != 2:
            raise ValueError(self.start_print, "`w_in` must be 2D")

        if value.shape != (
            self._size_in,
            self._size,
        ):
            raise ValueError(
                self.start_print
                + "`w_in` must be [{:d}, {:d}]".format(self._size_in, self._size)
            )

        self._w_in = np.array(value).astype("float")

    @property
    def w_out(self) -> np.ndarray:
        """ (np.ndarray) [NxO] output weights """
        return onp.array(self._w_out)

    @w_out.setter
    def w_out(self, value: np.ndarray):
        if np.ndim(value) != 2:
            raise ValueError(self.start_print + "`w_out` must be 2D")

        if value.shape != (
            self._size,
            self._size_out,
        ):
            raise ValueError(
                self.start_print
                + "`w_out` must be [{:d}, {:d}]".format(self._size, self._size_out)
            )

        self._w_out = np.array(value).astype("float")


class ForceRateEulerJax_IO(RecRateEulerJax_IO):
    """
    Implements a pseudo recurrent reservoir, for use in reservoir transfer

    In this layer, input and output weights are present, but no recurrent connectivity exists. Instead, "recurrent inputs" are injected into each layer neuron. The activations of the neurons are then compared to those of a target reservoir, and the recurrent weights can be solved for using linear regression.

    .. rubric:: Dynamics

    The layer implements the dynamics

    .. math:: \\tau \\cdot \\dot{x} + x = W_{in} \\dot I(t) + F(t) + b + \\sigma \\cdot \\zeta(t)

    where :math:`\\tau`` is the neuron time constants; :math:`x` is the N-dimensional state of the neurons; :math:`I(t)` is the I-dimensional input vector; :math:`W_{in}` is the [IxN] matrix of input weights; :math:`F(t)` is the N-dimensional vector of forcing currents --- these should be the recurrent input currents taking from another reservoir; :math:`b` is the N-dimensional vector of bias currents for the neurons; and :math:`\\sigma \\cdot \\zeta(t)` is a white noise process with std. dev :math:`\\sigma`.

    The outputs of the layer are given by :math:`W_{out} \\cdot H(x)`, where :math:`W_{out}` is the NxO output weight matrix, and :math:`H(x)` is a neuron transfer function. By default, :math:`H(x)` is the linear-threshold function

    .. math:: H_{ReLU}(x) = \\text{max}(x, 0)

    .. rubric:: Training

    `.RecRateEulerJax` implements the `.JaxTrainedLayer` training interface. See the documentation for :py:meth:`.JaxTrainedLayer.train_output_target` for information about training these layers.
    """

    def __init__(
        self,
        w_in: np.ndarray,
        w_out: np.ndarray,
        tau: FloatVector,
        bias: FloatVector,
        noise_std: float = 0.0,
        activation_func: Callable[[FloatVector], FloatVector] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        ``JAX``-backed firing rate reservoir, used for reservoir transfer

        :param np.ndarray w_in:                     Input weights [IxN]
        :param np.ndarray w_out:                    Output weights [NxO]
        :param np.ndarray tau:                      Time constants [N]
        :param np.ndarray bias:                     Bias values [N]
        :param Optional[float] noise_std:           White noise standard deviation applied to reservoir neurons. Default: ``0.0``
        :param Callable[[FloatVector], float] activation_func:  Neuron transfer function f(x: float) -> float. Must be vectorised. Default: ``H_ReLU``
        :param Optional[float] dt:                  Reservoir time step. Default: ``np.min(tau) / 10.0``
        :param Optional[str] name:                  Name of the layer. Default: ``None``
        :param Optional[Jax RNG key] rng_key        Jax RNG key to use for noise. Default: Internally generated
        """

        # - Everything should be 2D
        w_in = np.atleast_2d(w_in)
        w_out = np.atleast_2d(w_out)

        # - Call super-class initialisation
        super().__init__(
            w_in,
            np.zeros((w_in.shape[1], w_in.shape[1])),
            w_out,
            tau,
            bias,
            noise_std,
            activation_func,
            dt,
            name,
            rng_key,
            *args,
            **kwargs,
        )

        # - Get compiled evolution function for forced reservoir
        self._evolve_jit = _get_force_evolve_jit(activation_func)

    @property
    def _evolve_functional(self):
        """
        Return a functional form of the evolution function for this layer

        Returns a function ``evol_func`` with the signature::

            def evol_func(params, state, (inputs, forces)) -> (outputs, new_state):

        :return Callable[[Params, State, np.ndarray, np.ndarray], Tuple[np.ndarray, State]]:
        """

        def evol_func(
            params: Params,
            state: State,
            inputs_forces: Tuple[np.ndarray, np.ndarray],
        ) -> Tuple[np.ndarray, State, Dict[str, np.ndarray]]:
            # - Unpack inputs
            inputs, forces = inputs_forces

            # - Call the jitted evolution function for this layer
            new_state, res_inputs, res_acts, outputs, key1 = self._evolve_jit(
                state,
                params["w_in"],
                params["w_out"],
                params["bias"],
                params["tau"],
                inputs,
                forces,
                self._noise_std,
                self._rng_key,
                self._dt,
            )

            # - Maintain RNG key, if not under compilation
            if not isinstance(key1, jax.core.Tracer):
                self._rng_key = key1

            # - Return the outputs from this layer, and the final layer state
            states_t = {
                "res_inputs": res_inputs,
                "res_acts": res_acts,
                "outputs": outputs,
            }
            return outputs, new_state, states_t

        # - Return the evolution function
        return evol_func

    def get_output_from_state(self, state):
        activity = self._H(state)
        output = np.dot(activity, self._w_out)

        return output, activity

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        ts_force: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        evolve() - Evolve the reservoir state

        :param Optional[TSContinuous] ts_input: Input time series
        :param Optional[TSContinuous] ts_force: Forced time series
        :param Optional[float] duration:        Duration of evolution in seconds
        :param Optional[int] num_timesteps:     Number of time steps to evolve (based on self.dt)

        :return: ts_output:     TSContinuous Output time series
        """

        # - Prepare time base and inputs
        time_base_inp, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Extract forcing inputs
        if ts_force is None:
            forces = np.zeros((num_timesteps, self._size))
        else:
            forces = ts_force(time_base_inp)

        # - Call raw evolution function
        self._state, _, _, outputs, self._rng_key = self._evolve_jit(
            self._state,
            self._w_in,
            self._w_out,
            self._bias,
            self._tau,
            inps,
            forces,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        # - Wrap outputs as time series
        ts_output = TSContinuous.from_clocked(
            onp.array(outputs), t_start=self.t, dt=self.dt
        )

        # - Increment timesteps
        self._timestep += num_timesteps

        return ts_output

    def to_dict(self) -> dict:
        """
        Convert the layer to a dictionary for saving
        :return dict:
        """
        # - Get base dictionary
        config = super().to_dict()
        config.pop("w_recurrent")

        # - Include class-specific aspects
        config.update(
            {
                "class_name": "ForceRateEulerJax_IO",
                "w_in": onp.array(self.w_in).tolist(),
                "w_out": onp.array(self.w_out).tolist(),
            }
        )

        # - Return configuration
        return config


class FFRateEulerJax(RecRateEulerJax):
    """
    ``JAX``-backed firing-rate recurrent layer

    `.FFRateEulerJax` implements a feed-forward dynamical layer, using a ``JAX``-implemented solver as a back end. The design permits gradient-based learning of weights, biases and time constants using `jax.grad`.

    `.FFRateEulerJax` is compatible with the `.layers.training.jax_trainer` module.

    .. rubric:: Dynamics

    This layer implements the dynamical system

    .. math:: \\tau \\cdot \\dot{x} + x = W \\cdot i(t) + b + \\sigma\\cdot\\zeta(t)

    where :math:`\\tau`` is the neuron time constants; :math:`x` is the N-dimensional state vector of the layer neurons;
    :math:`i(t)` is the I-dimensional input signal at time :math:`t`; :math:`W` is an [IxN] matrix defining the weight matrix of this layer; :math:`b` is a vector of bias inputs for each neuron; and :math:`\\sigma\\cdot\\zeta(t)` is a white noise process with std. dev. :math:`\\sigma``.

    The output of the layer is :math:`H(x)`, where :math:`H(x)` is a neuron transfer function. By default, :math:`H(x)` is the linear-threshold function

    .. math:: H_{ReLU}(x) = \\text{max}(x, 0)

    .. rubric:: Training

    `.RecRateEulerJax` implements the `.JaxTrainedLayer` training interface. See the documentation for :py:meth:`.JaxTrainedLayer.train_output_target` for information about training these layers.
    """

    def __init__(
        self,
        w_in: np.ndarray,
        tau: np.ndarray,
        bias: np.ndarray,
        noise_std: float = 0.0,
        activation_func: Callable[[FloatVector], FloatVector] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Implement a ``JAX``-backed feed-forward dynamical neuron layer.

        :param np.ndarray w_in:                 Weights [IxN]
        :param np.ndarray tau:                  Time constants [N]
        :param np.ndarray bias:                 Bias values [N]
        :param float noise_std:                 White noise standard deviation applied to reservoir neurons. Default: ``0.0``
        :param Callable[[FloatVector], float] activation_func:   Neuron transfer function f(x: float) -> float. Must be vectorised. Default: H_ReLU
        :param Optional[float] dt:              Reservoir time step. Default: ``np.min(tau) / 10.0``
        :param Optional[str] name:              Name of the layer. Default: ``None``
        :param Optional[Jax RNG key] rng_key:   Jax RNG key to use for noise. Default: Internally generated
        """

        # - Everything should be 2D
        w_in = np.atleast_2d(w_in)

        # - Transform to np.array if necessary
        tau = np.array(tau)
        bias = np.array(bias)

        if dt is None:
            dt = np.min(tau) / 10.0

        # - Call super-class initialisation
        super().__init__(
            0.0,
            0.0,
            0.0,
            noise_std,
            activation_func,
            dt,
            name,
            rng_key,
            *args,
            **kwargs,
        )

        # - Correct layer size
        self._size_in = w_in.shape[0]
        self._size = w_in.shape[1]
        self._size_out = w_in.shape[1]

        # -- Set properties
        self.tau = tau
        self.bias = bias
        self.w_in = w_in
        self._weights = 0.0
        self._w_out = 1.0

        # - Reset layer state
        self.reset_all()

    @property
    def w_in(self) -> np.ndarray:
        """ (np.ndarray) [IxN] input weights """
        return onp.array(self._w_in)

    @w_in.setter
    def w_in(self, value: np.ndarray):
        if np.ndim(value) != 2:
            raise ValueError(self.start_print + "`w_in` must be 2D")

        if value.shape != (
            self._size_in,
            self._size,
        ):
            raise ValueError(
                self.start_print
                + "`w_in` must be [{:d}, {:d}]".format(self._size_in, self._size)
            )

        self._w_in = np.array(value).astype("float")

    def to_dict(self) -> dict:
        """
        Convert the parameters of this class to a dictionary

        :return dict:
        """
        # - Get base dictionary
        config = super().to_dict()
        config.pop("weights")

        # - Include class-specific aspects
        config.update(
            {
                "class_name": "FFRateEulerJax",
                "w_in": onp.array(self.w_in).tolist(),
            }
        )

        # - Return configuration
        return config
