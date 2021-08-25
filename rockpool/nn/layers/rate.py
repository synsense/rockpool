##
# rate.py - Non-spiking rate-coded dynamical layers, with ReLu / LT neurons. Euler solvers
##

from typing import Callable, Optional, Union, Tuple, List
from warnings import warn

from importlib import util

if util.find_spec("numba") is None:
    raise ModuleNotFoundError(
        "'numba' backend not found. Modules that rely on numba will not be available."
    )

import numpy as np
from numba import njit

from rockpool.timeseries import TSContinuous
from rockpool.nn.layers.layer import Layer

from rockpool.nn.modules.timed_module import astimedmodule

# from ..training.gpl.rr_trained_layer import RRTrainedLayer

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Relative tolerance for float comparions
tolerance = 1e-5

### --- Configure exports

__all__ = ["FFRateEuler", "PassThrough", "RecRateEuler"]


### --- Helper functions


def is_multiple(a: float, b: float, tolerance: float = tolerance) -> bool:
    """
    is_multiple - Check whether a%b is 0 within some tolerance.

    :param float a:             The number that may be multiple of b
    :param float b:             The number a may be a multiple of
    :param float tolerance:     Relative tolerance

    :return bool:               True if a is a multiple of b within some tolerance
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
def re_lu(x: np.ndarray) -> np.ndarray:
    cop = np.copy(x)
    cop[np.where(x < 0)] = 0
    return cop


@njit
def noisy(x: np.ndarray, std_dev: float) -> np.ndarray:
    """
    noisy - Add randomly distributed noise to each element of x

    :param np.ndarray x:      values that noise is added to
    :param float std_dev:   the standard deviation of the noise to be added

    :return: np.ndarray:      x with noise added
    """
    return std_dev * np.random.randn(*x.shape) + x


### --- Functions used in connection with FFRateEuler class


@njit
def re_lu(x: np.ndarray) -> np.ndarray:
    """
    Activation function for rectified linear units.

    :param np.ndarray x:    with current neuron potentials

    :return: np.ndarray               np.clip(x, 0, None)
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


### --- FFRateLayer base class


class FFRateLayerBase(Layer):
    """ Base class for feed-forward layers of rate based neurons """

    def __init__(
        self,
        weights: np.ndarray,
        dt: float = 1.0,
        noise_std: float = 0.0,
        bias: Union[float, np.ndarray] = 0.0,
        delay: float = 0.0,
        name: str = "unnamed",
    ):
        """
        Base class for feed-forward layers of rate based neurons

        :param ndarray weights:             [MxN] Weight matrix
        :param float dt:                    Evolution time step in seconds. Default: 1.0
        :param float noise_std:             Noise std. dev. per second. Default: 0.0, no noise
        :param ndarray bias:                [Nx1] Vector of bias currents. Default: 0.0, no bias
        :param str name:                    Name of this layer. Default: None
        """

        # - Make sure some required parameters are set
        if weights is None:
            raise TypeError(
                f"{self.class_name} `{name}`: `weights` must not be `None`."
            )
        if bias is None:
            raise TypeError(f"{self.class_name} `{name}`: `bias` must not be `None`.")

        # - Call super-class initialiser
        super().__init__(
            weights=np.asarray(weights, "float"), dt=dt, noise_std=noise_std, name=name
        )
        self.bias = self._correct_param_shape(bias)

    def _correct_param_shape(self, v) -> np.ndarray:
        """
        Convert an argument to a 1D-np.ndarray and verify that the dimensions match `self.size`

        :param float v: Scalar or array-like that is to be converted

        :return:        v as 1D-np.ndarray, possibly expanded to `self.size`
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

    def train(
        self,
        ts_target: TSContinuous,
        ts_input: TSContinuous,
        is_first: bool,
        is_last: bool,
        method: str = "rr",
        **kwargs,
    ):
        """
        Wrapper to standardize training syntax across layers. Use specified training method to train layer for current batch.

        :param ts_target:   Target time series for current batch.
        :param ts_input:    Input to the layer during the current batch.
        :param is_first:    Set ``True`` to indicate that this batch is the first in training procedure.
        :param is_last:     Set ``True`` to indicate that this batch is the last in training procedure.
        :param method:      String indicating which training method to choose. Currently only ridge regression (``'rr'``) is supported.
        :param kwargs:      will be passed on to corresponding training method.
        """
        # - Choose training method
        if method in {
            "rr",
            "ridge",
            "ridge regression",
            "regression",
            "linear regression",
            "linreg",
        }:
            training_method = self.train_rr
        else:
            raise ValueError(
                self.start_print
                + f"Training method `{method}` is currently not "
                + "supported. Use `rr` for ridge regression."
            )
        # - Call training method
        return training_method(
            ts_target, ts_input, is_first=is_first, is_last=is_last, **kwargs
        )

    def _prepare_training_data(
        self,
        ts_target: TSContinuous,
        ts_input: Optional[Union[TSContinuous, None]] = None,
        is_first: bool = True,
        is_last: bool = False,
    ):
        """
        Check and rasterize input and target data for this batch

        :param TSContinuous ts_target:                          Target time series for this batch
        :param Optional[Union[TSContinuous, None]] ts_input:    Input time series for this batch. Default: ``None``
        :param bool is_first:                                   Set to ``True`` if this is the first batch in training. Default: ``True``
        :param bool is_last:                                    Set to ``True`` if this is the last batch in training. Default: ``False``

        :return: (inp, target, time_base)
            inp np.ndarray:                 Rasterized input time series for this batch [T, M]
            target np.ndarray:              Rasterized target time series for this batch [T, O]
            time_base np.ndarray:           Time base for ``inp`` and ``target`` [T,]
        """

        # - Call superclass method
        __, target, time_base = super()._prepare_training_data(
            ts_target, ts_input, is_first, is_last
        )

        # - Prepare input data
        inp = np.zeros((np.size(time_base), self.size_in))

        if ts_input is None:
            # - Assume zero input
            print(self.start_print + "No `ts_input` defined, assuming input to be 0.")
        else:
            # - Sample input trace and check for correct dimensions
            exception_flag = ts_input.beyond_range_exception
            ts_input.beyond_range_exception = False
            inp = self._check_input_dims(ts_input(time_base))
            ts_input.beyond_range_exception = exception_flag

            # - Treat "NaN" as zero inputs
            inp[np.where(np.isnan(inp))] = 0

        return inp, target, time_base

    def to_dict(self) -> dict:
        """
        Convert parameters of `self` to a dict if they are relevant for reconstructing an identical layer

        :return dict:   A dictionary containing the parameters of this layer
        """
        config = super().to_dict()
        config["bias"] = self.bias.tolist()
        return config

    @property
    def bias(self):
        """
        (ArrayLike[float]) (N) Vector of bias parameters for this layer
        """
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = self._expand_to_net_size(new_bias)


### --- PassThrough class


@astimedmodule(
    parameters=["weights", "bias", "delay"],
    states=["ts_buffer"],
    simulation_parameters=["noise_std"],
)
class PassThrough(FFRateLayerBase):
    """ Feed-forward layer with neuron states directly corresponding to input with an optional delay """

    def __init__(
        self,
        weights: np.ndarray,
        dt: float = 1.0,
        noise_std: float = 0.0,
        bias: Union[float, np.ndarray] = 0.0,
        delay: float = 0.0,
        name: str = "unnamed",
    ):
        """
        Implement a feed-forward layer that simply passes input (possibly delayed)

        :param ndarray weights:             [MxN] Weight matrix
        :param float dt:                    Time step for Euler solver, in seconds. Default: 1.0
        :param float noise_std:             Noise std. dev. per second. Default: 0.0, no noise
        :param ndarray bias:                [Nx1] Vector of bias currents. Default: 0.0, no bias
        :param float delay:                 Delay between input and output, in seconds. Default: 0.0, no delay
        :param str name:                    Name of this layer. Default: None
        """
        # - Set delay
        self._delay_steps = 0 if delay is None else int(np.round(delay / dt))

        # - Call super-class initialiser
        super().__init__(
            weights=np.asarray(weights, float), dt=dt, noise_std=noise_std, name=name
        )

        self.reset_all()

    def reset_buffer(self):
        """
        Reset the internal buffer of this layer

        This method will wipe the internal buffer to zeros.
        """
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
        Evolve the state of this layer given an input

        :param Optional[TSContinuous] ts_input: Input time series
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then ``num_timesteps`` or the duration of ``ts_input`` will be used for the evolution duration
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then ``duration`` or the duration of ``ts_input`` will be used for the evolution duration
        :param bool verbose:                    Currently has no effect

        :return TSContinuous:                   Output time series
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

        # - Return time series with output data and bias
        ts_out = TSContinuous.from_clocked(
            samples=samples_out + self.bias, dt=self.dt, t_start=self.t, name="Outputs"
        )

        # - Update state and time
        self.state = samples_out[-1]
        self._timestep += num_timesteps

        return ts_out

    def __repr__(self):
        return (
            "PassThrough layer object `{}`.\nnSize: {}, size_in: {}, delay: {}".format(
                self.name, self.size, self.size_in, self.delay
            )
        )

    def print_buffer(self, **kwargs):
        """
        Display the internal buffer of this layer

        :param kwargs:  Optional arguments passed to the `.TSContinuous.print` method
        """
        if self.ts_buffer is not None:
            self.ts_buffer.print(**kwargs)
        else:
            print("This layer does not use a delay.")

    @property
    def buffer(self):
        """
        (ndarray) The internal buffer of this layer.
        """
        if self.ts_buffer is not None:
            return self.ts_buffer.samples
        else:
            print("This layer does not use a delay.")

    def reset_state(self):
        """
        Reset the internal state and buffer of this layer to zero
        """
        super().reset_state()
        self.reset_buffer()

    def reset_all(self):
        """
        Reset both the internal state and time stamp of this layer
        """
        super().reset_all()
        self.reset_buffer()

    def to_dict(self) -> dict:
        """
        Convert parameters of `self` to a dict if they are relevant for reconstructing an identical layer

        :return dict:   A dictionary containing the parameters of this layer
        """
        config = super().to_dict()
        config["delay"] = self.delay
        return config

    @property
    def delay(self):
        """
        (float) The delay imposed by this layer, in seconds
        """
        return self._delay_steps * self.dt

    @property
    def delay_steps(self):
        """
        (int) The delay imposed by this layer, in units of `.dt`
        """
        return self._delay_steps


### --- FFRateEuler class


@astimedmodule(
    parameters=["weights", "bias", "gain", "tau"],
    states=["_state"],
    simulation_parameters=["noise_std"],
)
class FFRateEuler(FFRateLayerBase):
    """
    Feedforward layer consisting of rate-based neurons

    `.FFRateEuler` is a simple feed-forward layer of dynamical neurons, backed with a forward-Euler solver with a fixed time step. The neurons in this layer implement the dynamics

    .. math::

        \\tau \\cdot \\dot{x} + x = g \\cdot W I(t) + \\sigma \\cdot \\zeta(t)

    where :math:`x` is the Nx1 vector of internal states of neurons in the layer; :math:`\\dot{x}` is the derivative of those staes with respect to time; :math:`\\tau` is the vector of time constants of the neurons in the layer; :math:`I(t)` is the instantaneous input injected into each neuron at time :math:`t`; :math:`W` is the MxN matrix of weights connecting the input to the neurons in the layer; and :math:`\\sigma \\cdot \\zeta(t)` is a white noise process with standard deviation :math:`\\sigma``.

    The output of the layer is given by

    .. math ::
        o = H(x + b)

    where :math:`H(x)` is the neuron transfer function, which by default is the linear-threshold (or "rectified linear" or ReLU) function :math:`H(x) = max(0, x)`; :math:`b` is the Nx1 vector of bias values for this layer; and :math:`g` is the Nx1 vector of gain parameters for the neurons in this layer.

    :Training:

    `.FFRateEuler` supports weight training with linear or ridge regression, using the `.train` method. To use this facility, use the `.train` method instead of the `.evolve` method, calling `.train` in turn over multiple batches::

        lyr = FFRateEuler(...)

        # - Loop over batches and train
        is_first = True
        is_last = False

        for (input_batch_ts, target_batch_ts) in batches[:-1]:
            lyr.train(target_batch_tsm, input_batch_ts, is_first, is_last)
            is_first = False

        # - Finalise training for last batch
        is_last = True
        (input_batch_ts, target_batch_ts) = batches[-1]
        lyr.train(target_batch_ts, input_batch_ts, is_first, is_last)

    """

    def __init__(
        self,
        weights: np.ndarray,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        noise_std: float = 0.0,
        activation_func: Callable[[np.ndarray], np.ndarray] = re_lu,
        tau: Union[float, np.ndarray] = 10.0,
        gain: Union[float, np.ndarray] = 1.0,
        bias: Union[float, np.ndarray] = 0.0,
    ):
        """
        Implement a feed-forward non-spiking neuron layer, with an Euler method solver

        :param ndarray weights:                             [MxN] Weight matrix
        :param Optional[float] dt:                          Time step for Euler solver, in seconds. Default: `None`, which will use `min(tau) / 10` as the time step, for numerical stability
        :param Optional[str] name:                          Name of this layer. Default: `None`
        :param float noise_std:                             Noise std. dev. per second. Default: 0.0, no noise
        :param Callable[[float], float] activation_func:    Callable a = f(x) Neuron activation function. Default: ReLU
        :param ArrayLike[float] tau:                        [Nx1] Vector of neuron time constants in seconds. Default: 10.0
        :param ArrayLike[float] gain:                       [Nx1] Vector of gain factors. Default: 1.0, unitary gain
        :param ArrayLike[float] bias:                       [Nx1] Vector of bias currents. Default: 0.0

        """

        # - Make sure some required parameters are set
        if tau is None:
            raise TypeError(f"{self.class_name} `{name}`: `tau` may not be `None`.")
        if gain is None:
            raise TypeError(f"{self.class_name} `{name}`: `gain` may not be `None`.")

        # - Set a reasonable dt
        if dt is None:
            min_tau = np.min(tau)
            dt = min_tau / 10

        # - Call super-class initialiser
        super().__init__(
            weights=weights, dt=dt, noise_std=noise_std, bias=bias, name=name
        )

        # - Check remaining parameter shapes
        try:
            self.tau, self.gain = map(self._correct_param_shape, (tau, gain))
        except AssertionError:
            raise AssertionError(
                self.start_print
                + "Numbers of elements in tau and gain must be 1 or match layer size."
            )

        # - Reset this layer state and set attributes
        self.reset_all()
        self.alpha = self._dt / self.tau
        self.activation_func = activation_func

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Evolve the state of this layer given an input

        :param Optional[TSContinuous] ts_input: Input time series. Default: `None`, no stimulus is provided
        :param Optional[float] duration:        Simulation/Evolution time, in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` is used to determine evolution time
        :param bool verbose:                    Currently no effect, just for conformity

        :return TSContinuous:                   Output time series
        """

        # - Prepare time base
        time_base_inp, inp, num_timesteps = self._prepare_input(
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

        time_base = np.r_[time_base_inp, self.t]

        return TSContinuous(time_base, sample_act, name="Outputs")

    def stream(
        self, duration: float, dt: float, verbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        Stream data through this layer

        :param float duration:          Total duration for which to handle streaming
        :param float dt:                Streaming time step
        :param bool verbose:            Display feedback. Default: `False`, don't display feedback

        :yield (float, ndarray):        (t, state)

        :return (float, ndarray):       Final output (t, state)
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
                inp = np.repeat(np.atleast_2d(inp[1][0, :]), euler_steps_per_dt, axis=0)

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
        return (self.t, np.reshape(self._activation(self._state + self._bias), (1, -1)))

    def to_dict(self) -> dict:
        """
        Convert parameters of `self` to a dict if they are relevant for reconstructing an identical layer

        :return dict: Dictionary of parameters to use when reconstructing this layer
        """
        config = super().to_dict()
        config["tau"] = self.tau.tolist()
        config["gain"] = self.gain.tolist()
        warn(
            f"FFRateEuler `{self.name}`: `activation_func` can not be stored with this "
            + "method. When creating a new instance from this dict, it will use the "
            + "default activation function."
        )
        return config

    @property
    def activation(self):
        """
        (ArrayLike[float]) The activation of this layer, after the activation function
        """
        return self.activation_func(self.state)

    ### --- properties

    @property
    def tau(self):
        """
        (ArayLike[float]) (N) Vector of time constants for the neurons in this layer
        """
        return self._tau

    @tau.setter
    def tau(self, new_tau):
        new_tau = self._expand_to_net_size(new_tau)
        if not (new_tau >= self._dt).all():
            raise ValueError("All tau must be at least dt.")
        self._tau = new_tau
        self._alpha = self._dt / new_tau

    @property
    def alpha(self):
        """
        (ndarray) (N) Vector `.tau` / `.dt` for the neurons in this layer
        """
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        new_alpha = self._expand_to_net_size(new_alpha)
        if not (new_alpha <= 1).all():
            raise ValueError("All alpha must be at most 1.")
        self._alpha = new_alpha
        self._tau = self._dt / new_alpha

    @property
    def gain(self):
        """
        (ArrayLike[float]) (N) Vector of gain parameters for this layer
        """
        return self._gain

    @gain.setter
    def gain(self, new_gain):
        self._gain = self._expand_to_net_size(new_gain)

    @property
    def activation_func(self):
        """
        (Callable[[ndarray], ndarray) Activation function for the neurons in this layer
        """
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


@astimedmodule(
    parameters=["weights", "bias", "tau"],
    states=["_state"],
    simulation_parameters=["noise_std"],
)
class RecRateEuler(Layer):
    """
    A standard recurrent non-spiking layer of dynamical neurons

    `.RecRateEuler` implements a very standard recurrent layer of dynamical neurons, which by default have linear-threshold (or "rectified-linear", or ReLU) transfer function. The layer is backed by a simple forward-Euler solver with a fixed time step.

    The neurons in the layer implement the dynamical system

    .. math::

        \\tau \\cdot \\dot{x} + x = W H(x + b) + I(t) + \\sigma \\cdot \\zeta(t)

    where :math:`x` is the Nx1 vector of internal states of each neuron; :math:`\\dot{x}` is the derivative of these states with respect to time; :math:`\\tau` is the vector of time constants for each neurons; :math:`W` is the [NxN] recurrent weight matrix for this layer; :math:`b` is the Nx1 vector of neuron biases for this layer; :math:`I(t)` is the input injected into each neuron in this layer at time :math:`t`; :math:`\\sigma \\cdot \\zeta(t)`` is a white noise process with standard deviation :math:`\\sigma` at each time step. :math:`H(x)` is the neuron transfer function, which by default is the linear threshold function

    .. math::

        H(x) = max(0, x)

    .. seealso:: The tutorial :ref:`/tutorials/building_reservoir.ipynb` demonstrates using this layer.
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray = 0.0,
        tau: np.ndarray = 1.0,
        activation_func: Callable[[np.ndarray], np.ndarray] = re_lu,
        dt: float = 0.1,
        noise_std: float = 0.0,
        name: str = "unnamed",
    ):
        """
        Implement a recurrent layer with non-spiking firing rate neurons, using a forward-Euler solver

        :param ndarray weights:                             (NxN) matrix of recurrent weights
        :param ArrrayLike[float] bias:                      (N) vector (or scalar) of bias currents. Default: 0.0
        :param ArrrayLike[float] tau:                       (N) vector (or scalar) of neuron time constants. Default: 1.0
        :param Callable[[float], float] activation_func:    Activation function for each neuron, with signature (x) -> f(x). Default: `re_lu`
        :param float dt:                                    Time step for integration (Euler method). Default: `None`, which results in taking a minimum time step of `min(tau) / 10.0` for numerical stability.
        :param float noise_std:                             Std. Dev. of state noise injected at each time step. Default: 0.0, no noise
        :param str name:                                    Name of this layer. Default: `None`
        """

        # - Call super-class init
        super().__init__(weights=np.asarray(weights, float), name=name, dt=dt)

        # - Check size and shape of `weights`
        if self.weights.ndim != 2:
            raise ValueError(
                f"{self.class_name} `{name}`: `weights` must be a matrix with 2 dimensions"
            )
        if self.weights.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"{self.class_name} `{name}`: `weights` must be a square matrix"
            )

        # - Check arguments
        if tau is None:
            raise TypeError(f"{self.class_name} `{name}`: `tau` may not be `None`.")
        if noise_std is None:
            raise TypeError(
                f"{self.class_name} `{name}`: `noise_std` may not be `None`."
            )

        # - Assign properties
        self.bias = bias
        self.tau = tau
        self.activation_func = activation_func
        self.noise_std = noise_std

        # - Reset the internal state
        self.reset_all()

    ### --- State evolution method

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Evolve the states of this layer given an input

        :param Optional[TSContinuous] ts_input: Input time series to use during evolution. Default: `None`, do not inject any input
        :param Optional[float] duration:        Desired evolution time in seconds. If not provided, then `num_timesteps` or the duration of `ts_input` will determine evolution duration
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of `.dt`. If not provided, then `duration` or the duration of `ts_input` will determine evolution duration
        :param bool verbose:                    Currently no effect, just for conformity

        :return TSContinuous:                   output time series
        """

        # - Prepare time base
        time_base_inp, input_steps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Generate a noise trace
        # Noise correction: Standard deviation after some time would be noise_std * sqrt(0.5*dt/tau)
        noise_step = (
            np.random.randn(num_timesteps, self.size)
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

        time_base = np.r_[time_base_inp, self.t]

        # - Construct a return TimeSeries
        return TSContinuous(time_base, activity, name="Outputs")

    def stream(
        self, duration: float, dt: float, verbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        Stream data through this layer

        :param float duration:  Total duration for which to handle streaming
        :param float dt:        Streaming time step
        :param bool verbose:    Display feedback

        :yield: (float, ndarray)    (t, state)

        :return (float, ndarray):   Final output (t, state)
        """

        # - Initialise simulation, determine how many dt to evolve for
        if verbose:
            print("Layer: I'm preparing")
        time_trace = np.arange(0, duration + dt, dt)
        num_steps = np.size(time_trace) - 1
        euler_steps_per_dt = int(dt / self._dt)

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(time_trace), self.size)
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
                inp = np.repeat(np.atleast_2d(inp[1][0, :]), euler_steps_per_dt, axis=0)

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
        return (self.t, np.reshape(self._activation(self._state + self._bias), (1, -1)))

    def to_dict(self) -> dict:
        """
        Convert parameters of `self` to a dict if they are relevant for reconstructing an identical layer

        :return dict: Dictionary of parameters to use when reconstructing this layer
        """
        config = super().to_dict()
        config["tau"] = self.tau.tolist()
        config["bias"] = self.bias.tolist()
        warn(
            f"FFRateEuler `{self.name}`: `activation_func` can not be stored with this "
            + "method. When creating a new instance from this dict, it will use the "
            + "default activation function."
        )
        return config

    ### --- Properties

    @Layer.dt.setter
    def dt(self, new_dt: float):
        # - Check that the time step is reasonable
        min_tau = np.min(self.tau)
        assert new_dt <= min_tau / 10, "`new_dt` must be <= {}".format(min_tau / 10)

        # - Call super-class setter
        self._dt = new_dt

    @property
    def activation_func(self):
        """
        (Callable) Activation function for this layer

        This function must have the signature Callable[[ndarray], ndarray]
        """
        return self._activation

    @activation_func.setter
    def activation_func(self, new_activation):
        self._activation = new_activation

        # - Build a state evolution function
        self._evolveEuler = get_rec_evolution_function(new_activation)

    @property
    def bias(self) -> np.ndarray:
        """
        (ndarray) (N) Vector of bias values for the neurons in this layer
        :return:
        """
        return self._bias

    @bias.setter
    def bias(self, new_bias: np.ndarray):
        self._bias = self._expand_to_net_size(new_bias, "new_bias")

    @property
    def tau(self) -> np.ndarray:
        """
        (ndarray) (N) Vector of time constants for the neurons in this layer
        :return:
        """
        return self._tau

    @tau.setter
    def tau(self, new_tau: np.ndarray):
        self._tau = self._expand_to_net_size(new_tau, "new_tau")

        # - Ensure dt is reasonable for numerical accuracy
        if np.min(self._tau) < 10 * self.dt:
            warn(
                self.start_print
                + "Some values in `tau` are quite small. For numerical stability"
                + "`dt` should be reduced to at most 0.1 * `smallest tau`."
            )
