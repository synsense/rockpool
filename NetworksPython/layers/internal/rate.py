##
# rate.py - Non-spiking rate-coded dynamical layers, with ReLu / LT neurons. Euler solvers
##

import numpy as np
from typing import Callable
from numba import njit

from ...timeseries import TSContinuous
from ..layer import Layer
from typing import Optional, Union, Tuple, List

from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Relative tolerance for float comparisons
tolerance = 1e-5

### --- Configure exports

__all__ = ["FFRateEuler", "PassThrough", "RecRateEuler"]


### --- Helper functions


def is_multiple(a: float, b: float, tolerance: float = tolerance) -> bool:
    """
    is_multiple - Check whether a%b is 0 within some tolerance.
    :param a: float The number that may be multiple of b
    :param b: float The number a may be a multiple of
    :param tolerance: float Relative tolerance
    :return bool: True if a is a multiple of b within some tolerance
    """
    min_remainder = min(a % b, b - a % b)
    return min_remainder < tolerance * b


def print_progress(curr: int, total: int, passed: float):
    print(
        "Progress: [{:6.1%}]    in {:6.1f} s. Remaining:   {:6.1f}".format(
            curr / total, passed, passed * (total - curr) / max(0.1, curr)
        ),
        end="\r",
    )

@njit
def noisy(x: np.ndarray, std_dev: float) -> np.ndarray:
    """
    noisy - Add randomly distributed noise to each element of x
    :param x:  Array-like with values that noise is added to
    :param std_dev: Float, the standard deviation of the noise to be added
    :return:        Array-like, x with noise added
    """
    return std_dev * np.random.randn(*x.shape) + x


### --- Functions used in connection with FFRateEuler class


@njit
def re_lu(x: np.ndarray) -> np.ndarray:
    """
    Activation function for rectified linear units.
    :param x:             ndarray with current neuron potentials
    :return:                np.clip(x, 0, None)
    """
    cop = np.copy(x)
    cop[np.where(x < 0)] = 0
    return cop


def get_ff_evolution_function(activation_func: Callable[[np.ndarray], np.ndarray]):
    """
    get_ff_evolution_function: Construct a compiled Euler solver for a given activation function

    :param activation_func: Callable (x) -> f(x)
    :return: Compiled function evolve_Euler_complete(state, inp, weights, size, num_steps, gain, bias, alpha, noise_std)
    """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(
        state: np.ndarray,
        inp: np.ndarray,
        weights: np.ndarray,
        size: int,
        num_steps: int,
        gain: np.ndarray,
        bias: np.ndarray,
        alpha: np.ndarray,
        noise_std,
    ) -> np.ndarray:

        # - Initialise storage of layer output
        weighted_input = inp @ weights
        activities = np.zeros((num_steps + 1, size))

        # - Loop over time steps. The updated state already corresponds to
        # subsequent time step. Therefore skip state update in final step
        # and only update activation.
        for step in range(num_steps):
            # - Store layer activity
            activities[step, :] = activation_func(state + bias)

            # - Evolve layer state
            d_state = -state + noisy(gain * weighted_input[step, :], noise_std)
            state += d_state * alpha

        # - Compute final activity
        activities[-1, :] = activation_func(state + bias)

        return activities

    # - Return the compiled function
    return evolve_Euler_complete


def get_rec_evolution_function(activation_func: Callable[[np.ndarray], np.ndarray]):
    """
   get_rec_evolution_function: Construct a compiled Euler solver for a given activation function

   :param activation_func: Callable (x) -> f(x)
   :return: Compiled function evolve_Euler_complete(state, size, weights, input_steps, dt, num_steps, bias, tau)
   """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(
        state: np.ndarray,
        size: int,
        weights: np.ndarray,
        input_steps: np.ndarray,
        num_steps: int,
        dt: float,
        bias: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        # - Initialise storage of network output
        activity = np.zeros((num_steps + 1, size))

        # - Precompute dt / tau
        lambda_ = dt / tau

        # - Loop over time steps
        for step in range(num_steps):
            # - Evolve network state
            this_act = activation_func(state + bias)
            d_state = -state + input_steps[step, :] + this_act @ weights
            state += d_state * lambda_

            # - Store network state
            activity[step, :] = this_act

        # - Get final activation
        activity[-1, :] = activation_func(state + bias)

        return activity

    # - Return the compiled function
    return evolve_Euler_complete


### --- FFRateEuler class


class FFRateEuler(Layer):
    """ Feedforward layer consisting of rate-based neurons """

    def __init__(
        self,
        weights: np.ndarray,
        dt: float = None,
        name: str = None,
        noise_std: float = 0.0,
        activation_func: Callable[[np.ndarray], np.ndarray] = re_lu,
        tau: Union[float, np.ndarray] = 10.0,
        gain: Union[float, np.ndarray] = 1.0,
        bias: Union[float, np.ndarray] = 0.0,
    ):
        """
        FFRateEuler - Implement a feed-forward non-spiking neuron layer, with an Euler method solver

        :param weights:             np.ndarray [MxN] Weight matrix
        :param dt:             float Time step for Euler solver, in seconds
        :param name:         string Name of this layer
        :param noise_std:       float Noise std. dev. per second
        :param activation_func:    Callable a = f(x) Neuron activation function (Default: ReLU)
        :param tau:           np.ndarray [Nx1] Vector of neuron time constants
        :param gain:          np.ndarray [Nx1] Vector of gain factors
        :param bias:          np.ndarray [Nx1] Vector of bias currents
        """

        # - Make sure some required parameters are set
        assert weights is not None, "`weights` is required"

        assert tau is not None, "`tau` is required"

        assert bias is not None, "`bias` is required"

        assert gain is not None, "`gain` is required"

        # - Set a reasonable dt
        if dt is None:
            min_tau = np.min(tau)
            dt = min_tau / 10

        # - Call super-class initialiser
        super().__init__(
            weights=np.asarray(weights, float), dt=dt, noise_std=noise_std, name=name
        )

        # - Check all parameter shapes
        try:
            self.tau, self.gain, self.bias = map(
                self._correct_param_shape, (tau, gain, bias)
            )
        except AssertionError:
            raise AssertionError(
                "Numbers of elements in tau, gain and bias"
                + " must be 1 or match layer size."
            )

        # - Reset this layer state and set attributes
        self.reset_all()
        self.alpha = self._dt / self.tau
        self.activation_func = activation_func

    def _correct_param_shape(self, v) -> np.ndarray:
        """
        _correct_param_shape - Convert v to 1D-np.ndarray and verify
                              that dimensions match self.size
        :param v:   Float or array-like that is to be converted
        :return:    v as 1D-np.ndarray
        """
        v = np.array(v, dtype=float).flatten()
        assert v.shape in (
            (1,),
            (self.size,),
            (1, self.size),
            (self.size),
            1,
        ), "Numbers of elements in v must be 1 or match layer size"
        return v

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare time base
        time_base, inp, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        sample_act = self._evolveEuler(
            state=self._state,  # self._state is automatically updated
            inp=inp,
            weights=self._weights,
            size=self._size,
            num_steps=num_timesteps,
            gain=self._gain,
            bias=self._bias,
            alpha=self._alpha,
            # Without correction, standard deviation after some time will be
            # self._noise_std * sqrt(self._alpha/2)
            noise_std=self._noise_std * np.sqrt(2.0 / self._alpha),
        )

        # - Increment internal time representation
        self._timestep += num_timesteps

        return TSContinuous(time_base, sample_act)

    def stream(
        self, duration: float, dt: float, verbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        stream - Stream data through this layer
        :param duration:   float Total duration for which to handle streaming
        :param dt:         float Streaming time step
        :param verbose:    bool Display feedback

        :yield: (t, state)

        :return: Final (t, state)
        """

        # - Initialise simulation, determine how many dt to evolve for
        if verbose:
            print("Layer: I'm preparing")
        time_trace = np.arange(0, duration + dt, dt)
        num_steps = np.size(time_trace) - 1
        euler_steps_per_dt = int(np.round(dt / self._dt))

        if verbose:
            print("Layer: Prepared")

        # - Loop over dt steps
        for step in range(num_steps):
            if verbose:
                print("Layer: Yielding from internal state.")
            if verbose:
                print("Layer: step", step)
            if verbose:
                print("Layer: Waiting for input...")

            # - Yield current output, receive input for next time step
            inp = (
                yield self._t,
                np.reshape(self._activation(self._state + self._bias), (1, -1)),
            )

            # - Set zero input if no input provided
            if inp is None:
                inp = np.zeros(euler_steps_per_dt, self._size_in)
            else:
                inp = np.repeat(
                    np.atleast_2d(inp[1][0, :]), euler_steps_per_dt, axis=0
                )

            if verbose:
                print("Layer: Input was: ", inp)

            # - Evolve layer
            _ = self._evolveEuler(
                state=self._state,  # self._state is automatically updated
                inp=inp,
                weights=self._weights,
                size=self._size,
                num_steps=euler_steps_per_dt,
                gain=self._gain,
                bias=self._bias,
                alpha=self._alpha,
                # Without correction, standard deviation after some time will be
                # self._noise_std * sqrt(self._alpha/2)
                noise_std=self._noise_std * np.sqrt(2.0 / self._alpha),
            )

            # - Increment time
            self._timestep += euler_steps_per_dt

        # - Return final state
        return (
            self.t,
            np.reshape(self._activation(self._state + self._bias), (1, -1)),
        )

    def train_rr(
        self,
        ts_target: TSContinuous,
        ts_input: TSContinuous,
        regularize=0,
        is_first=True,
        is_last=False,
    ):
        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices from
                   previous batches.
        :param ts_target:    TSContinuous - target for current batch
        :param ts_input:     TSContinuous - input to self for current batch
        :regularize:       float - regularization for ridge regression
        :is_first:            bool - True if current batch is the first in training
        :is_last:            bool - True if current batch is the last in training
        """

        # - Discrete time steps for evaluating input and target time series
        num_timesteps = int(np.round(ts_input.duration / self.dt))
        time_base = self._gen_time_trace(ts_input.t_start, num_timesteps)

        if not is_last:
            # - Discard last sample to avoid counting time points twice
            time_base = time_base[:-1]

        # - Make sure time_base does not exceed ts_input
        time_base = time_base[time_base <= ts_input.t_stop]

        # - Prepare target data
        target = ts_target(time_base)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            target
        ).any(), "nan values have been found in target (where: {})".format(
            np.where(np.isnan(target))
        )

        # - Check target dimensions
        if target.ndim == 1 and self.size == 1:
            target = target.reshape(-1, 1)

        assert (
            target.shape[-1] == self.size
        ), "Target dimensions ({}) does not match layer size ({})".format(
            target.shape[-1], self.size
        )

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        inp = np.zeros((np.size(time_base), self.size_in + 1))
        inp[:, -1] = 1

        # Warn if input time range does not cover whole target time range
        if (
            not ts_target.contains(time_base)
            and not ts_input.periodic
            and not ts_target.periodic
        ):
            warn(
                "WARNING: ts_input (t = {} to {}) does not cover ".format(
                    ts_input.t_start, ts_input.t_stop
                )
                + "full time range of ts_target (t = {} to {})\n".format(
                    ts_target.t_start, ts_target.t_stop
                )
                + "Assuming input to be 0 outside of defined range.\n"
                + "If you are training by batches, check that the target signal is also split by batch.\n"
            )

        # - Sample input trace and check for correct dimensions
        inp[:, :-1] = self._check_input_dims(ts_input(time_base))

        # - Treat "NaN" as zero inputs
        inp[np.where(np.isnan(inp))] = 0

        # - For first batch, initialize summands
        if is_first:
            # Matrices to be updated for each batch
            self._xty = np.zeros(
                (self.size_in + 1, self.size)
            )  # inp.T (dot) target
            self._xtx = np.zeros(
                (self.size_in + 1, self.size_in + 1)
            )  # inp.T (dot) inp

            # Corresponding Kahan compensations
            self._kahan_comp_xty = np.zeros_like(self._xty)
            self._mfKahanCompXTX = np.zeros_like(self._xtx)

        # - New data to be added, including compensation from last batch
        #   (Matrix summation always runs over time)
        upd_xty = inp.T @ target - self._kahan_comp_xty
        upd_xtx = inp.T @ inp - self._mfKahanCompXTX

        if not is_last:
            # - Update matrices with new data
            new_xty = self._xty + upd_xty
            new_xtx = self._xtx + upd_xtx

            # - Calculate rounding error for compensation in next batch
            self._kahan_comp_xty = (new_xty - self._xty) - upd_xty
            self._mfKahanCompXTX = (new_xtx - self._xtx) - upd_xtx

            # - Store updated matrices
            self._xty = new_xty
            self._xtx = new_xtx

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self._xty += upd_xty
            self._xtx += upd_xtx

            # - Weight and bias update by ridge regression
            solution = np.linalg.solve(
                self._xtx + regularize * np.eye(self.size_in + 1), self._xty
            )

            self.weights = solution[:-1, :]
            self.bias = solution[-1, :]

            # - Remove data stored during this training
            self._xty = (
                self._xtx
            ) = self._kahan_comp_xty = self._mfKahanCompXTX = None

    # @njit
    # def potential(self, vInput: np.ndarray) -> np.ndarray:
    #     return (self._alpha * noisy(vInput@self.weights*self.gain + self.bias, self.noise_std)
    #             + (1-self._alpha)*self.state)

    def __repr__(self):
        return "FFRateEuler layer object `{}`.\nnSize: {}, size_in: {}   ".format(
            self.name, self.size, self.size_in
        )

    @property
    def vActivation(self):
        return self.activation_func(self.state)

    ### --- properties

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, new_tau):
        new_tau = self._correct_param_shape(new_tau)
        if not (new_tau >= self._dt).all():
            raise ValueError("All tau must be at least dt.")
        self._tau = new_tau
        self._alpha = self._dt / new_tau

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        new_alpha = self._correct_param_shape(new_alpha)
        if not (new_alpha <= 1).all():
            raise ValueError("All alpha must be at most 1.")
        self._alpha = new_alpha
        self._tau = self._dt / new_alpha

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = self._correct_param_shape(new_bias)

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, new_gain):
        self._gain = self._correct_param_shape(new_gain)

    @property
    def activation_func(self):
        return self._activation

    @activation_func.setter
    def activation_func(self, f):
        self._activation = f
        self._evolveEuler = get_ff_evolution_function(f)

    @Layer.dt.setter
    def dt(self, new_dt):
        if not (self.tau >= new_dt).all():
            raise ValueError("All tau must be at least dt.")
        self._dt = new_dt
        self._alpha = new_dt / self._tau


### --- PassThrough Class


class PassThrough(FFRateEuler):
    """ Neuron states directly correspond to input, but can be delayed. """

    def __init__(
        self,
        weights: np.ndarray,
        dt: float = 1.0,
        noise_std: float = 0.0,
        bias: Union[float, np.ndarray] = 0.0,
        delay: float = 0.0,
        name: str = None,
    ):
        """
        PassThrough - Implement a feed-forward layer that simply passes input (possibly delayed)

        :param weights:         np.ndarray [MxN] Weight matrix
        :param dt:         float Time step for Euler solver, in seconds
        :param noise_std:   float Noise std. dev. per second
        :param bias:      np.ndarray [Nx1] Vector of bias currents
        :param delay:      float Delay between input and output, in seconds
        :param name:     string Name of this layer
        """
        # - Set delay
        self._delay_steps = 0 if delay is None else int(np.round(delay / dt))

        # - Call super-class initialiser
        super().__init__(
            weights=np.asarray(weights, float),
            dt=dt,
            noise_std=noise_std,
            activation_func=lambda x: x,
            bias=bias,
            name=name,
        )

        self.reset_all()

    def reset_buffer(self):
        if self.delay != 0:
            vtBuffer = np.arange(self._delay_steps + 1) * self._dt
            self.ts_buffer = TSContinuous(
                vtBuffer, np.zeros((len(vtBuffer), self.size))
            )
        else:
            self.ts_buffer = None

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare time base
        time_base, inp, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Apply input weights and add noise
        in_processed = noisy(inp @ self.weights, self.noise_std)

        if self.ts_buffer is not None:
            # - Combined time trace for buffer and processed input
            num_time_steps_comb = num_timesteps + self._delay_steps
            time_comb = self._gen_time_trace(self.t, num_time_steps_comb)

            # - Array for buffered and new data
            samples_comb = np.zeros((time_comb.size, self.size_in))
            steps_in = time_base.size

            # - Buffered data: last point of buffer data corresponds to self.t,
            #   which is also part of current input
            samples_comb[:-steps_in] = self.ts_buffer.samples[:-1]

            # - Processed input data (weights and noise)
            samples_comb[-steps_in:] = in_processed

            # - Output data
            samples_out = samples_comb[:steps_in]

            # - Update buffer with new data
            self.ts_buffer.samples = samples_comb[steps_in - 1 :]

        else:
            # - Undelayed processed input
            samples_out = in_processed

        # - Update state and time
        self.state = samples_out[-1]
        self._timestep += num_timesteps

        # - Return time series with output data and bias
        return TSContinuous(time_base, samples_out + self.bias)

    def __repr__(self):
        return "PassThrough layer object `{}`.\nnSize: {}, size_in: {}, delay: {}".format(
            self.name, self.size, self.size_in, self.delay
        )

    def print_buffer(self, **kwargs):
        if self.ts_buffer is not None:
            self.ts_buffer.print(**kwargs)
        else:
            print("This layer does not use a delay.")

    @property
    def buffer(self):
        if self.ts_buffer is not None:
            return self.ts_buffer.samples
        else:
            print("This layer does not use a delay.")

    def reset_state(self):
        super().reset_state()
        self.reset_buffer()

    def reset_all(self):
        super().reset_all()
        self.reset_buffer()

    @property
    def delay(self):
        return self._delay_steps * self.dt

    @property
    def delay_steps(self):
        return self._delay_steps

    # @delay.setter
    # def delay(self, new_delay):
    # Some method to extend self.ts_buffer


class RecRateEuler(Layer):
    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray = 0.0,
        tau: np.ndarray = 1.0,
        activation_func: Callable[[np.ndarray], np.ndarray] = re_lu,
        dt: float = None,
        noise_std: float = 0.0,
        name: str = None,
    ):
        """
        RecRate: Implement a recurrent layer with firing rate neurons

        :param weights:             np.ndarray (NxN) matrix of recurrent weights
        :param bias:          np.ndarray (N) vector (or scalar) of bias currents
        :param tau:           np.ndarray (N) vector (or scalar) of neuron time constants
        :param activation_func:    Callable (x) -> f(x) Activation function
        :param dt:             float Time step for integration (Euler method)
        :param noise_std:       float Std. Dev. of state noise injected at each time step
        :param name:           str Name of this layer
        """

        # - Call super-class init
        super().__init__(weights=np.asarray(weights, float), name=name)

        # - Check size and shape of `weights`
        assert len(weights.shape) == 2, "`weights` must be a matrix with 2 dimensions"
        assert weights.shape[0] == weights.shape[1], "`weights` must be a square matrix"

        # - Check arguments
        assert tau is not None, "`tau` may not be None"

        assert noise_std is not None, "`noise_std` may not be None"

        # - Assign properties
        self.bias = bias
        self.tau = tau
        self.activation_func = activation_func
        self.noise_std = noise_std

        if dt is not None:
            self.dt = dt

        # - Reset the internal state
        self.reset_all()

    ### --- Properties

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    @bias.setter
    def bias(self, new_bias: np.ndarray):
        self._bias = self._expand_to_net_size(new_bias, "new_bias")

    @property
    def tau(self) -> np.ndarray:
        return self._tau

    @tau.setter
    def tau(self, new_tau: np.ndarray):
        self._tau = self._expand_to_net_size(new_tau, "new_tau")

        # - Ensure dt is reasonable for numerical accuracy
        self.dt = np.min(self.tau) / 10

    ### --- State evolution method

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:                TSContinuous  output spike series

        """

        # - Prepare time base
        time_base, input_steps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Generate a noise trace
        # Noise correction: Standard deviation after some time would be noise_std * sqrt(0.5*dt/tau)
        noise_step = (
            np.random.randn(np.size(time_base), self.size)
            * self.noise_std
            * np.sqrt(2.0 * self._tau / self._dt)
        )

        # - Call Euler method integrator
        #   Note: Bypass setter method for .state
        activity = self._evolveEuler(
            self._state,
            self._size,
            self._weights,
            input_steps + noise_step,
            num_timesteps,
            self._dt,
            self._bias,
            self._tau,
        )

        # - Increment internal time representation
        self._timestep += num_timesteps

        # - Construct a return TimeSeries
        return TSContinuous(time_base, activity)

    def stream(
        self, duration: float, dt: float, verbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        stream - Stream data through this layer
        :param duration:   float Total duration for which to handle streaming
        :param dt:         float Streaming time step
        :param verbose:    bool Display feedback

        :yield: (t, state)

        :return: Final (t, state)
        """

        # - Initialise simulation, determine how many dt to evolve for
        if verbose:
            print("Layer: I'm preparing")
        time_trace = np.arange(0, duration + dt, dt)
        num_steps = np.size(time_trace) - 1
        euler_steps_per_dt = int(dt / self._dt)

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(time_base), self.size)
            * self.noise_std
            * np.sqrt(2.0 * self._tau / self._dt)
        )

        if verbose:
            print("Layer: Prepared")

        # - Loop over dt steps
        for step in range(num_steps):
            if verbose:
                print("Layer: Yielding from internal state.")
            if verbose:
                print("Layer: step", step)
            if verbose:
                print("Layer: Waiting for input...")

            # - Yield current activity, receive input for next time step
            inp = (
                yield self._t,
                np.reshape(self._activation(self._state + self._bias), (1, -1)),
            )

            # - Set zero input if no input provided
            if inp is None:
                inp = np.zeros(euler_steps_per_dt, self._size_in)
            else:
                inp = np.repeat(
                    np.atleast_2d(inp[1][0, :]), euler_steps_per_dt, axis=0
                )

            if verbose:
                print("Layer: Input was: ", inp)

            # - Evolve layer
            _ = self._evolveEuler(
                state=self._state,  # self._state is automatically updated
                size=self._size,
                weights=self._weights,
                input_steps=inp + noise_step[step, :],
                num_steps=euler_steps_per_dt,
                dt=self._dt,
                bias=self._bias,
                tau=self._tau,
            )

            # - Increment time
            self._timestep += euler_steps_per_dt

        # - Return final activity
        return (
            self.t,
            np.reshape(self._activation(self._state + self._bias), (1, -1)),
        )

    ### --- Properties

    @Layer.dt.setter
    def dt(self, new_dt: float):
        # - Check that the time step is reasonable
        min_tau = np.min(self.tau)
        assert new_dt <= min_tau / 10, "`new_dt` must be <= {}".format(min_tau / 10)

        # - Call super-class setter
        super(RecRateEuler, RecRateEuler).dt.__set__(self, new_dt)

    @property
    def activation_func(self):
        return self._activation

    @activation_func.setter
    def activation_func(self, new_activation):
        self._activation = new_activation

        # - Build a state evolution function
        self._evolveEuler = get_rec_evolution_function(new_activation)
