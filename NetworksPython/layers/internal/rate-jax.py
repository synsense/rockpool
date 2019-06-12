##
# Firing rate reservoir with jax back-end
##

import jax.numpy as np
from jax import jit
from jax.lax import scan
import jax.random as rand
import numpy as onp
from typing import Optional, Tuple, Callable, Union

FloatVector = Union[float, np.ndarray]

from ..layer import Layer
from ...timeseries import TimeSeries, TSContinuous

__all__ = ["RecRateEulerJax", "ForceRateEulerJax", "H_ReLU", "H_tanh"]


def H_ReLU(x: FloatVector) -> FloatVector:
    return np.clip(x, 0, None)


def H_tanh(x: FloatVector) -> FloatVector:
    return np.tanh(x)


def _get_rec_evolve_jit(H: Callable):
    @jit
    def rec_evolve_jit(
        x0: np.ndarray,
        w_in: np.ndarray,
        w_recurrent: np.ndarray,
        w_out: np.ndarray,
        bias: np.ndarray,
        tau: np.ndarray,
        inputs: np.ndarray,
        noise_std: float,
        key,
        dt: float,
    ):
        """
        rec_evolve_jit() - Compiled recurrent evolution function

        :param x0:
        :param w_in:
        :param w_recurrent:
        :param w_out:
        :param bias:
        :param tau:
        :param inputs:
        :param noise_std:
        :param key:
        :param dt:
        :return:
        """
        # - Pre-compute dt/tau
        dt_tau = dt / tau

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inps):
            inp, rand = inps
            activation = H(x)
            rec_input = np.dot(activation, w_recurrent)
            dx = dt_tau * (-x + inp + bias + rand + rec_input)

            return x + dx, (rec_input, activation)

        # - Evaluate passthrough input layer
        res_inputs = np.dot(inputs, w_in)

        # - Compute random numbers for reservoir noise
        __all__, subkey = rand.split(key)
        noise = noise_std * rand.normal(subkey, shape=(inputs.shape[0], np.size(x0)))

        # - Use `scan` to evaluate reservoir
        x, (rec_inputs, res_acts) = scan(reservoir_step, x0, (res_inputs, noise))

        # - Evaluate passthrough output layer
        outputs = np.dot(res_acts, w_out)

        return x, res_inputs, rec_inputs, res_acts, outputs

    return rec_evolve_jit


def _get_force_evolve_jit(H: Callable):
    @jit
    def force_evolve_jit(
        x0: np.ndarray,
        w_in: np.ndarray,
        w_out: np.ndarray,
        bias: np.ndarray,
        tau: np.ndarray,
        inputs: np.ndarray,
        force: np.ndarray,
        noise_std: float,
        key,
        dt: float,
    ):
        """
        rforce_evolve_jit() - Compiled recurrent evolution function

        :param x0:
        :param w_in:
        :param w_recurrent:
        :param w_out:
        :param bias:
        :param tau:
        :param inputs:
        :param force:
        :param noise_std:
        :param key:
        :param dt:
        :return:
        """
        # - Pre-compute dt/tau
        dt_tau = dt / tau

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inps):
            inp, rand, force = inps
            activation = H(x)
            dx = dt_tau * (-x + inp + bias + rand + force)

            return x + dx, activation

        # - Evaluate passthrough input layer
        res_inputs = np.dot(inputs, w_in)

        # - Compute random numbers for reservoir noise
        __all__, subkey = rand.split(key)
        noise = noise_std * rand.normal(subkey, shape=(inputs.shape[0], np.size(x0)))

        # - Use `scan` to evaluate reservoir
        x, res_acts = scan(reservoir_step, x0, (res_inputs, noise, force))

        # - Evaluate passthrough output layer
        outputs = np.dot(res_acts, w_out)

        return x, res_inputs, res_acts, outputs

    return force_evolve_jit


class RecRateEulerJax(Layer):
    def __init__(
        self,
        weights: np.ndarray,
        w_recurrent: np.ndarray,
        w_out: np.ndarray,
        tau: np.ndarray,
        bias: np.ndarray,
        noise_std: Optional[float] = 0.0,
        activation_func: Optional[Callable[[FloatVector], FloatVector]] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key=None,
    ):
        """
        RecRateEulerJax - `jax`-backed firing rate reservoir

        :param w_in:
        :param w_recurrent:
        :param w_out:
        :param tau:
        :param bias:
        :param noise_std:
        :param activation_func:
        :param dt:
        :param name:
        """

        # - Everything should be 2D
        w_in = np.atleast_2d(weights)
        w_recurrent = np.atleast_2d(w_recurrent)
        w_out = np.atleast_2d(w_out)

        # - Get information about network size
        self._num_inputs = w_in.shape[0]
        self._size = w_in.shape[1]
        self._num_outputs = w_out.shape[1]

        # -- Set properties
        self.w_recurrent = w_recurrent
        self.w_out = w_out
        self.tau = tau
        self.bias = bias
        self._H = activation_func

        if dt is None:
            dt = np.min(tau) / 10.0

        # - Call super-class initialisation
        super().__init__(w_in, dt, noise_std, name)

        # - Get compiled evolution function
        self._evolve_jit = _get_rec_evolve_jit(activation_func)

        # - Reset layer state
        self.reset_all()

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        _, self._rng_key = rand.split(rng_key)

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        evolve() - Evolve the reservoir state

        :param ts_input:        TSContinuous Input time series
        :param duration:        float Duration of evolution in seconds
        :param num_timesteps:   int Number of time steps to evolve (based on self.dt)
        :return:
        """

        # - Prepare time base and inputs
        time_base, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call raw evolution function
        _, _, _, outputs = self._evolve_raw(inps)

        # - Wrap outputs as time series
        return TSContinuous(time_base, outputs)

    def _evolve_raw(
        self, inps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        _evolve_raw() - Raw evolution of an input array

        :param inps:    np.ndarray Input matrix [T, I]

        :return:  (res_inputs, rec_inputs, res_acts, outputs)
                res_inputs:     np.ndarray Weighted inputs to reservoir units [T, N]
                rec_inputs      np.ndarray Recurrent inputs to reservoir units [T, N]
                res_acts        np.ndarray Reservoir activity trace [T, N]
                outputs         np.ndarray Output of network [T, O]
        """
        # - Call compiled Euler solver to evolve reservoir
        self._state, res_inputs, rec_inputs, res_acts, outputs = self._evolve_jit(
            self._state,
            self._weights,
            self._w_recurrent,
            self._w_out,
            self._bias,
            self._tau,
            inps,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        # - Increment timesteps
        self._timestep += inps.shape[0] - 1

        return res_inputs, rec_inputs, res_acts, outputs

    @property
    def w_in(self) -> np.ndarray:
        return self._weights

    @w_in.setter
    def w_in(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_in` must be 2D"

        assert value.shape == (
            self._num_inputs,
            self._size,
        ), "`win` must be [{:d}, {:d}]".format(self._num_inputs, self._size)

        self._weights = value

    @property
    def w_recurrent(self) -> np.ndarray:
        return self._w_recurrent

    @w_recurrent.setter
    def w_recurrent(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_recurrent` must be 2D"

        assert value.shape == (
            self._size,
            self._size,
        ), "`w_recurrent` must be [{:d}, {:d}]".format(self._size, self._size)

        self._w_recurrent = value

    @property
    def w_out(self) -> np.ndarray:
        return self._w_out

    @w_out.setter
    def w_out(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_out` must be 2D"

        assert value.shape == (
            self._size,
            self._num_outputs,
        ), "`w_out` must be [{:d}, {:d}]".format(self._size, self._num_outputs)

        self._w_out = value

    @property
    def tau(self) -> np.ndarray:
        return self._tau

    @tau.setter
    def tau(self, value: np.ndarray):
        assert np.size(value) == self._size, "`tau` must have {:d} elements".format(
            self._size
        )
        self._tau = np.reshape(value, self._size)

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    @bias.setter
    def bias(self, value: np.ndarray):
        assert np.size(value) == self._size, "`bias` must have {:d} elements".format(
            self._size
        )
        self._bias = np.reshape(value, self._size)

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value: float):
        # - Ensure dt is numerically stable
        tau_min = np.min(self.tau) / 10.0
        if value is None:
            value = tau_min

        assert value >= tau_min, "`tau` must be at least {:.2e}".format(tau_min)

        self._dt = value


class ForceRateEulerJax(RecRateEulerJax):
    def __init__(
        self,
        weights: np.ndarray,
        w_out: np.ndarray,
        tau: np.ndarray,
        bias: np.ndarray,
        noise_std: float = 0.0,
        activation_func: Optional[Callable[[FloatVector], FloatVector]] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key=None,
    ):
        """
        ForceRateEulerJax - `jax`-backed firing rate reservoir, used for reservoir transfer

        :param w_in:
        :param w_out:
        :param tau:
        :param bias:
        :param noise_std:
        :param activation_func:
        :param dt:
        :param name:
        """

        # - Everything should be 2D
        w_in = np.atleast_2d(weights)
        w_out = np.atleast_2d(w_out)

        # - Get information about network size
        self._num_inputs = w_in.shape[0]
        self._size = w_in.shape[1]
        self._num_outputs = w_out.shape[1]

        # - Call super-class initialisation
        super().__init__(
            w_in,
            np.zeros((self._size, self._size)),
            w_out,
            tau,
            bias,
            noise_std,
            activation_func,
            dt,
            name,
            rng_key,
        )

        # - Get compiled evolution function for forced reservoir
        self._evolve_jit = _get_force_evolve_jit(activation_func)

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        ts_force: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        evolve() - Evolve the reservoir state

        :param ts_input:        TSContinuous Input time series
        :param ts_force:        TSContinuous Forced time series
        :param duration:        float Duration of evolution in seconds
        :param num_timesteps:   int Number of time steps to evolve (based on self.dt)
        :return:
        """

        # - Prepare time base and inputs
        time_base, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Extract forcing inputs
        if ts_force is None:
            forces = np.zeros((num_timesteps, self._size))
        else:
            forces = ts_force(time_base)

        # - Call raw evolution function
        _, _, outputs = self._evolve_raw(inps, forces)

        # - Wrap outputs as time series
        return TSContinuous(time_base, outputs)

    def _evolve_raw(
        self, inps: np.ndarray, forces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        _evolve_raw() - Raw evolution of an input array

        :param inps:    np.ndarray Input matrix [T, I]
        :param forces:  np.ndarray Forcing signals [T, N]

        :return:  (res_inputs, rec_inputs, res_acts, outputs)
                res_inputs:     np.ndarray Weighted inputs to forced reservoir units [T, N]
                res_acts        np.ndarray Reservoir activity trace [T, N]
                outputs         np.ndarray Output of network [T, O]
        """
        # - Call compiled Euler solver to evolve reservoir
        self._state, res_inputs, res_acts, outputs = self._evolve_jit(
            self._state,
            self._weights,
            self._w_out,
            self._bias,
            self._tau,
            inps,
            forces,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        # - Increment timesteps
        self._timestep += inps.shape[0] - 1

        return res_inputs, res_acts, outputs
