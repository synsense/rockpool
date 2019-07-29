from warnings import warn
from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional
import json

import numpy as np

from ..timeseries import TimeSeries, TSContinuous, TSEvent
from ..utils import to_scalar

# - Configure exports
__all__ = ["Layer"]


# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9


### --- Implements the Layer abstract class


class Layer(ABC):
    def __init__(
        self,
        weights: np.ndarray,
        dt: Optional[float] = 1,
        noise_std: Optional[float] = 0,
        name: Optional[str] = "unnamed",
    ):
        """
        Layer class - Implement an abstract layer of neurons (no implementation)

        :param weights:         np.ndarray Weight matrix for this layer
        :param dt:         float Time-step used for evolving this layer. Default: 1
        :param noise_std:   float Std. Dev. of state noise when evolving this layer. Default: 0. Defined as the expected
                                    std. dev. after 1s of integration time
        :param name:       str Name of this layer. Default: 'unnamed'
        """

        # - Assign properties
        if name is None:
            self.name = "unnamed"
        else:
            self.name = name

        try:
            # Try this before enforcing with Numpy atleast to account for custom classes for weights
            self._size_in, self._size = weights.shape
            self._weights = weights
        except Exception:
            weights = np.atleast_2d(weights)
            self._size_in, self._size = weights.shape
            self._weights = weights

        # - Check and assign dt and noise_std
        assert (
            np.size(dt) == 1 and np.size(noise_std) == 1
        ), "Layer `{}`: `dt` and `noise_std` must be scalars.".format(self.name)

        # - Assign default noise
        if noise_std is None:
            noise_std = 0.0

        # - Check dt
        assert dt is not None, "`dt` must be a numerical value"

        self._dt = dt
        self.noise_std = noise_std
        self._timestep = 0

    ### --- Common methods

    def _determine_timesteps(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> int:
        """
        _determine_timesteps - Determine over how many time steps to evolve with the given input

        :param ts_input:       TimeSeries  TxM or Tx1 Input signals for this layer
        :param duration:     float  Duration of the desired evolution, in seconds
        :param num_timesteps: int  Number of evolution time steps

        :return num_timesteps: int  Number of evolution time steps
        """

        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer `{}`: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Layer `{}`: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + " `ts_input` finishes before the current evolution time."
                    )
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            assert (
                isinstance(num_timesteps, int) and num_timesteps >= 0
            ), "Layer `{}`: num_timesteps must be a non-negative integer.".format(
                self.name
            )

        return num_timesteps

    def _prepare_input(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray, float):
        """
        _prepare_input - Sample input, set up time base

        :param ts_input:       TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:     float Duration of the desired evolution, in seconds
        :param num_timesteps: int Number of evolution time steps

        :return: (time_base, input_steps, duration)
            time_base:     ndarray T1 Discretised time base for evolution
            input_steps:    ndarray (T1xN) Discretised input signal for layer
            num_timesteps:  int Actual number of evolution time steps
        """

        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        time_base = self._gen_time_trace(self.t, num_timesteps)

        if ts_input is not None:
            # - Make sure time_base matches ts_input
            if not isinstance(ts_input, TSEvent):
                if not ts_input.periodic:
                    # - If time base limits are very slightly beyond ts_input.t_start and ts_input.t_stop, match them
                    if (
                        ts_input.t_start - 1e-3 * self.dt
                        <= time_base[0]
                        <= ts_input.t_start
                    ):
                        time_base[0] = ts_input.t_start
                    if (
                        ts_input.t_stop
                        <= time_base[-1]
                        <= ts_input.t_stop + 1e-3 * self.dt
                    ):
                        time_base[-1] = ts_input.t_stop

                # - Warn if evolution period is not fully contained in ts_input
                if not (ts_input.contains(time_base) or ts_input.periodic):
                    warn(
                        "Layer `{}`: Evolution period (t = {} to {}) ".format(
                            self.name, time_base[0], time_base[-1]
                        )
                        + "not fully contained in input signal (t = {} to {})".format(
                            ts_input.t_start, ts_input.t_stop
                        )
                    )

            # - Sample input trace and check for correct dimensions
            input_steps = self._check_input_dims(ts_input(time_base))

            # - Treat "NaN" as zero inputs
            input_steps[np.where(np.isnan(input_steps))] = 0

        else:
            # - Assume zero inputs
            input_steps = np.zeros((np.size(time_base), self.size_in))

        return time_base, input_steps, num_timesteps

    def _prepare_input_events(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input_events - Sample input from TSEvent, set up time base

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            spike_raster:    ndarray Boolean or integer raster containing spike info
            num_timesteps:    ndarray Number of evlution time steps
        """
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Extract spike timings and channels
        if ts_input is not None:
            # Extract spike data from the input variable
            spike_raster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                num_timesteps=num_timesteps,
                channels=np.arange(self.size_in),
                add_events=(self.add_events if hasattr(self, "add_events") else False),
            )[2]
            # - Make sure size is correct
            spike_raster = spike_raster[:num_timesteps, :]

        else:
            spike_raster = np.zeros((num_timesteps, self.size_in))

        return spike_raster, num_timesteps

    def _check_input_dims(self, inp: np.ndarray) -> np.ndarray:
        """
        Verify if dimension of input matches layer instance. If input
        dimension == 1, scale it up to self._size_in by repeating signal.
            inp : np.ndarray with input data
            return : inp, possibly with dimensions repeated
        """
        # - Replicate `ts_input` if necessary
        if inp.ndim == 1 or (inp.ndim > 1 and inp.shape[1]) == 1:
            if self.size_in > 1:
                warn(
                    f"Layer `{self.name}`: Only one channel provided in input - will "
                    + f"be copied to all {self.size_in} input channels."
                )
            inp = np.repeat(inp.reshape((-1, 1)), self._size_in, axis=1)
        else:
            # - Check dimensionality of input
            assert (
                inp.shape[1] == self._size_in
            ), "Layer `{}`: Input dimensionality {} does not match layer input size {}.".format(
                self.name, inp.shape[1], self._size_in
            )

        # - Return possibly corrected input
        return inp

    def _gen_time_trace(self, t_start: float, num_timesteps: int) -> np.ndarray:
        """
        Generate a time trace starting at t_start, of length num_timesteps+1 with
        time step length self._dt. Make sure it does not go beyond
        t_start+duration.

        :return time_trace, duration
        """
        # - Generate a trace
        time_trace = np.arange(num_timesteps + 1) * self._dt + t_start

        return time_trace

    def _expand_to_shape(
        self, inp, shape: tuple, var_name: str = "input", allow_none: bool = True
    ) -> np.ndarray:
        """
        _expand_to_shape: Replicate out a scalar to an array of shape shape

        :param inp:          scalar or array-like (size)
        :param shape:        tuple of int Shape that input should be expanded to
        :param var_name:   str Name of the variable to include in error messages
        :param allow_none:      bool Allow None as argument for inp
        :return:                np.ndarray (N) vector
        """
        if not allow_none:
            assert inp is not None, "Layer `{}`: `{}` must not be None".format(
                self.name, var_name
            )

        total_size = reduce(lambda m, n: m * n, shape)

        if np.size(inp) == 1:
            # - Expand input to full size
            inp = np.repeat(inp, total_size)

        assert (
            np.size(inp) == total_size
        ), "Layer `{}`: `{}` must be a scalar or have {} elements".format(
            self.name, var_name, total_size
        )

        # - Return object of correct shape
        return np.reshape(inp, shape)

    def _expand_to_size(
        self, inp, size: int, var_name: str = "input", allow_none: bool = True
    ) -> np.ndarray:
        """
        _expand_to_size: Replicate out a scalar to size

        :param inp:          scalar or array-like (size)
        :param size:           integer Size that input should be expanded to
        :param var_name:   str Name of the variable to include in error messages
        :param allow_none:      bool Allow None as argument for inp
        :return:                np.ndarray (N) vector
        """
        return self._expand_to_shape(inp, (size,), var_name, allow_none)

    def _expand_to_net_size(
        self, inp, var_name: str = "input", allow_none: bool = True
    ) -> np.ndarray:
        """
        _expand_to_net_size: Replicate out a scalar to the size of the layer

        :param inp:          scalar or array-like (N)
        :param var_name:   str Name of the variable to include in error messages
        :param allow_none:      bool Allow None as argument for inp
        :return:                np.ndarray (N) vector
        """
        return self._expand_to_shape(inp, (self.size,), var_name, allow_none)

    def _expand_to_weight_size(
        self, inp, var_name: str = "input", allow_none: bool = True
    ) -> np.ndarray:
        """
        _expand_to_weight_size: Replicate out a scalar to the size of the layer's weights

        :param inp:          scalar or array-like (NxN)
        :param var_name:   str Name of the variable to include in error messages
        :param allow_none:      bool Allow None as argument for inp
        :return:                np.ndarray (NxN) vector
        """

        return self._expand_to_shape(inp, (self.size, self.size), var_name, allow_none)

    ### --- String representations

    def __str__(self):
        return '{} object: "{}" [{} {} in -> {} {} out]'.format(
            self.__class__.__name__,
            self.name,
            self.size_in,
            self.input_type.__name__,
            self.size,
            self.output_type.__name__,
        )

    def __repr__(self):
        return self.__str__()

    ### --- State evolution methods

    @abstractmethod
    def evolve(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        evolve - Abstract method to evolve the state of this layer

        :param ts_input:     TimeSeries (TxM) External input trace to use when evolving the layer
        :param duration:   float Duration in seconds to evolve the layer
        :param num_timesteps: int Number of time steps to evolve the layer
        :return:            TimeSeries (TxN) Output of this layer
        """
        pass

    # @abstractmethod
    # def stream(self,
    #            duration: float,
    #            dt: float,
    #            verbose: bool = False,
    #           ) -> TimeSeries:
    #     """
    #     stream - Abstract method to evolve the state of this layer, in a streaming format
    #
    #     :param duration: float Total duration to be streamed
    #     :param dt:       float Streaming time-step (multiple of layer.dt)
    #
    #     :yield TimeSeries raw tuple representation on each time step
    #     """
    #     pass

    def reset_time(self):
        """
        reset_time - Reset the internal clock
        :return:
        """
        self._timestep = 0

    def randomize_state(self):
        """
        randomize_state - Randomise the internal state of this layer

        :return: None
        """
        # create random initial state with a gaussian distribution with mean
        # the values that were given and std the 20% of the absolute value
        self.state = np.random.normal(
            self.state, np.abs(self.state) * 0.02, size=(self.size,)
        )

    def reset_all(self):
        self.reset_time()
        self.reset_state()

    @abstractmethod
    def to_dict(self) -> dict:
        """
        to_dict - Convert parameters of `self` to a dict if they are relevant for
                  reconstructing an identical layer.
        """
        config = {}
        config["weights"] = self.weights.tolist()
        config["dt"] = self.dt
        config["noise_std"] = self.noise_std
        config["name"] = self.name

        config["class_name"] = self.class_name

        return config

    def save(self, config: dict, filename: str):
        """save - Save parameters from `config` in a json file.
        :param config:    dict of attributes to be saved.
        :param filename:  Path of file where parameters are stored.
        """
        with open(filename, "w") as f:
            json.dump(config, f)

    def save_layer(self, filename: str):
        """save - Obtain layer paramters from `self.to_dict` and save in a json file.
        :param filename:  Path of file where parameters are stored.
        """
        config = self.to_dict()
        self.save(config, filename)

    @classmethod
    def load_from_file(cls, filename: str, **kwargs) -> "cls":
        """load_from_file - Generate instance of `cls` with parameters loaded from file.
        :param filename: Path to the file where parameters are stored.
        :param kwargs:   Any keyword argument of the class __init__ method where the
                         parameter stored in the file should be overwritten.
        :return:
            Instance of cls with paramters from file.
        """
        # - Load dict from file
        with open(filename, "r") as f:
            config = json.load(f)
        # - Instantiate new class member from dict
        return cls.load_from_dict(config, **kwargs)

    @classmethod
    def load_from_dict(cls, config: dict, **kwargs) -> "cls":
        """load_from_dict - Generate instance of `cls` with parameters loaded from dict.
        :param config: Dict with parameters.
        :param kwargs: Any keyword argument of the class __init__ method where the
                       parameter from `config` should be overwritten.
        :return:
            Instance of cls with paramters from dict.
        """
        # - Overwrite parameters with kwargs
        config = dict(config, **kwargs)
        # - Remove class name from dict
        config.pop("class_name")
        return cls(**config)

    def reset_state(self):
        """
        reset_state - Reset the internal state of this layer. Sets state to zero

        :return: None
        """
        self.state = np.zeros(self.size)

    #### --- Properties

    @property
    def class_name(self) -> str:
        """class_name - Return name of `self` as a string."""
        # - Determine class name by removing "<class '" and "'>" and the package information
        return str(self.__class__).split("'")[1].split(".")[-1]

    @property
    def start_print(self):
        return f"{self.class_name} '{self.name}': "

    @property
    def output_type(self):
        return TSContinuous

    @property
    def input_type(self):
        return TSContinuous

    @property
    def size(self) -> int:
        return self._size

    @property
    def size_in(self) -> int:
        return self._size_in

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, fNewDt: float):
        self._dt = to_scalar(fNewDt)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, new_w: np.ndarray):
        assert new_w is not None, "Layer `{}`: weights must not be None.".format(
            self.name
        )

        # - Ensure weights are at least 2D
        try:
            assert new_w.ndim >= 2
        except AssertionError:
            warn("Layer `{}`: `new_w must be at least of dimension 2".format(self.name))
            new_w = np.atleast_2d(new_w)

        # - Check dimensionality of new weights
        assert (
            new_w.size == self.size_in * self.size
        ), "Layer `{}`: `new_w` must be of shape {}".format(
            (self.name, self.size_in, self.size)
        )

        # - Save weights with appropriate size
        self._weights = np.reshape(new_w, (self.size_in, self.size))

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        assert (
            np.size(new_state) == self.size
        ), "Layer `{}`: `new_state` must have {} elements".format(self.name, self.size)

        self._state = new_state

    @property
    def noise_std(self):
        return self._noise_std

    @noise_std.setter
    def noise_std(self, new_noise_std):
        self._noise_std = to_scalar(new_noise_std)

    @property
    def t(self):
        return self._timestep * self.dt

    @t.setter
    def t(self, new_t):
        self._timestep = int(np.floor(new_t / self.dt))

    # - Temporary, for maintaining compatibility with layers that still use _t
    @property
    def _t(self):
        return self._timestep * self.dt

    @_t.setter
    def _t(self, new_t):
        self._timestep = int(np.floor(new_t / self.dt))
