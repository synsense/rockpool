"""
Layer v1 base class for Rockpool layers
"""

from warnings import warn
from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional, Any, Tuple, Dict
import json

import numpy as np

from rockpool.timeseries import TimeSeries, TSContinuous, TSEvent
from rockpool.utilities.type_handling import to_scalar

# - Configure exports
__all__ = ["Layer"]


# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9


### --- Implements the Layer abstract class


class Layer(ABC):
    """
    Base class for Layers in rockpool

    This abstract class acts as a base class from which to derive subclasses that represent layers of neurons. As an abstract class, :py:class:`Layer` cannot be instantiated.

    .. seealso:: See :ref:`layerssummary` for examples of instantiating and using :py:class:`Layer` subclasses. See "Writing a new Layer subclass" for how to design and implement a new :py:class:`Layer` subclass.
    """

    def __init__(
        self,
        weights: np.ndarray,
        dt: float = 1.0,
        noise_std: float = 0.0,
        name: str = "unnamed",
        *args,
        **kwargs,
    ):
        """
        Implement an abstract layer of neurons (no implementation, must be subclassed)

        :param ArrayLike[float] weights:    Weight matrix for this layer. Indexed as [pre, post]
        :param float dt:                    Time-step used for evolving this layer. Default: 1
        :param float noise_std:             Std. Dev. of state noise when evolving this layer. Default: 0. Defined as the expected std. dev. after 1s of integration time
        :param str name:                    Name of this layer. Default: 'unnamed'
        """
        # - Call super-class init
        super().__init__(*args, **kwargs)

        # - Assign properties
        if name is None:
            self.name = "unnamed"
        else:
            self.name = name

        try:
            # Try this before enforcing with Numpy atleast to account for custom classes for weights
            self._size_in, self._size = weights.shape
            self._size_out = self._size
            self._weights = weights
        except Exception:
            weights = np.atleast_2d(weights)
            self._size_in, self._size = weights.shape
            self._size_out = self._size
            self._weights = weights

        # - Make sure `dt` is a float
        try:
            self._dt = float(dt)
        except TypeError:
            raise TypeError(self.start_print + "`dt` must be a scalar.")

        # Handle format of `noise_std`
        try:
            self.noise_std = float(noise_std)
        except TypeError:
            if noise_std is None:
                self.noise_std = 0.0
            else:
                raise TypeError(
                    self.start_print + "`noise_std` must be a scalar or `None`"
                )

        self._timestep = 0

    ### --- Common methods

    def _determine_timesteps(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> int:
        """
        Determine how many time steps to evolve with the given input

        :param Optional[TimeSeries] ts_input:   TxM or Tx1 time series of input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, ``num_timesteps`` or the duration of ``ts_input`` will be used to determine evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps, in units of :py:attr:`.dt`. If not provided, ``duration`` or the duration of ``ts_input`` will be used to determine evolution time

        :return int:                            num_timesteps: Number of evolution time steps
        """

        if num_timesteps is None:
            # - Determine ``num_timesteps``
            if duration is None:
                # - Determine duration
                if ts_input is None:
                    raise TypeError(
                        self.start_print
                        + "One of `num_timesteps`, `ts_input` or `duration` must be supplied."
                    )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TimeSeries
                    duration = ts_input.t_stop - self.t
                    if duration <= 0:
                        raise ValueError(
                            self.start_print
                            + "Cannot determine an appropriate evolution duration."
                            + " `ts_input` finishes before the current evolution time.",
                        )
            num_timesteps = int(np.floor((duration + tol_abs) / self.dt))
        else:
            if not isinstance(num_timesteps, int):
                raise TypeError(
                    self.start_print + "`num_timesteps` must be a non-negative integer."
                )
            elif num_timesteps < 0:
                raise ValueError(
                    self.start_print + "`num_timesteps` must be a non-negative integer."
                )

        return num_timesteps

    def _prepare_input(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sample input, set up time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current layer

        :param Optional[TimeSeries] ts_input:   :py:class:`.TimeSeries` of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will define the evolution time
        :param Optional[int] num_timesteps:     Integer number of evolution time steps, in units of ``.dt``. If not provided, then ``duration`` or the duration of ``ts_input`` will define the evolution time

        :return (ndarray, ndarray, int): (time_base, input_steps, num_timesteps)
            time_base:      T1 Discretised time base for evolution
            input_raster    (T1xN) Discretised input signal for layer
            num_timesteps:  Actual number of evolution time steps, in units of ``.dt``
        """
        assert (ts_input is None) or isinstance(
            ts_input, self.input_type
        ), "The layer {} can only receive inputs of class {}".format(
            self.name, str(self.input_type)
        )

        if self.input_type is TSContinuous:
            return self._prepare_input_continuous(ts_input, duration, num_timesteps)

        elif self.input_type is TSEvent:
            return self._prepare_input_events(ts_input, duration, num_timesteps)

        else:
            TypeError(
                "Layer._prepare_input can only handle `TSContinuous` and `TSEvent` classes"
            )

    def _prepare_input_continuous(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sample input, set up time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current layer

        :param Optional[TSContinuous] ts_input: :py:class:`.TSContinuous` of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will define the evolution time
        :param Optional[int] num_timesteps:     Integer number of evolution time steps, in units of ``.dt``. If not provided, then ``duration`` or the duration of ``ts_input`` will define the evolution time

        :return (ndarray, ndarray, int): (time_base, input_steps, num_timesteps)
            time_base:      T1 Discretised time base for evolution
            input_steps:    (T1xN) Discretised input signal for layer
            num_timesteps:  Actual number of evolution time steps, in units of ``.dt``
        """

        # - Work out how many time steps to take
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        time_base = self._gen_time_trace(self.t, num_timesteps)

        if ts_input is not None:

            # - Make sure time series is of correct type
            if not isinstance(ts_input, TSContinuous):
                raise TypeError(
                    self.start_print
                    + "`ts_input` must be of type `TSContinuous` or `None`."
                )

            # - Make sure time_base matches ts_input
            t_start_expected = time_base[0]
            t_stop_expected = time_base[-1]
            if not ts_input.periodic:
                # - If time base limits are very slightly beyond ts_input.t_start and ts_input.t_stop, match them
                if (
                    ts_input.t_start - 1e-3 * self.dt
                    <= t_start_expected
                    <= ts_input.t_start
                ):
                    t_start_expected = ts_input.t_start
                if (
                    ts_input.t_stop
                    <= t_stop_expected
                    <= ts_input.t_stop + 1e-3 * self.dt
                ):
                    t_stop_expected = ts_input.t_stop

            # - Warn if evolution period is not fully contained in ts_input
            if not (ts_input.contains(time_base) or ts_input.periodic):
                warn(
                    "Layer `{}`: Evolution period (t = {} to {}) ".format(
                        self.name, t_start_expected, t_stop_expected
                    )
                    + "is not fully contained in input signal (t = {} to {}).".format(
                        ts_input.t_start, ts_input.t_stop
                    )
                    + " You may need to use a `periodic` time series."
                )

            # - Sample input trace and check for correct dimensions
            input_steps = self._check_input_dims(ts_input(time_base))

            # - Treat "NaN" as zero inputs
            input_steps[np.where(np.isnan(input_steps))] = 0

        else:
            # - Assume zero inputs
            input_steps = np.zeros((num_timesteps, self.size_in))

        return time_base, input_steps, num_timesteps

    def _prepare_input_events(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sample input from a :py:class:`TSEvent` time series, set up evolution time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current layer

        :param Optional[TSEvent] ts_input:  TimeSeries of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:    Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will determine evolution itme
        :param Optional[int] num_timesteps: Number of evolution time steps, in units of ``.dt``. If not provided, then either ``duration`` or the duration of ``ts_input`` will determine evolution time

        :return (ndarray, ndarray, int):
            time_base:      T1X1 vector of time points -- time base for the rasterisation
            spike_raster:   Boolean or integer raster containing spike information. T1xM array
            num_timesteps:  Actual number of evolution time steps, in units of ``.dt``
        """

        # - Work out how many time steps to take
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        time_base = self._gen_time_trace(self.t, num_timesteps)

        # - Extract spike timings and channels
        if ts_input is not None:

            # - Make sure time series is of correct type
            if not isinstance(ts_input, TSEvent):
                raise TypeError(
                    self.start_print + "`ts_input` must be of type `TSEvent` or `None`."
                )

            # Extract spike data from the input variable
            spike_raster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                num_timesteps=np.size(time_base),
                channels=np.arange(self.size_in),
                add_events=(self.add_events if hasattr(self, "add_events") else False),
            )

        else:
            spike_raster = np.zeros((np.size(time_base), self.size_in))

        # - Check for correct input dimensions
        spike_raster = self._check_input_dims(spike_raster)

        return time_base, spike_raster, num_timesteps

    def _check_input_dims(self, inp: np.ndarray) -> np.ndarray:
        """
        Verify if dimensions of an input matches this layer instance

        If input dimension == 1, scale it up to self._size_in by repeating signal.

        :param ndarray inp: ArrayLike containing input data

        :return ndarray: ``inp``, possibly with dimensions repeated
        """
        # - Replicate input data if necessary
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
        Generate a time trace starting at ``t_start``, of length ``num_timesteps + 1`` with time step length :py:attr:`._dt`

        :param float t_start:       Start time, in seconds
        :param int num_timesteps:   Number of time steps to generate, in units of ``.dt``

        :return (ndarray): Generated time trace
        """
        # - Generate a trace
        time_trace = np.arange(num_timesteps) * self.dt + t_start

        return time_trace

    def _expand_to_shape(
        self, inp, shape: tuple, var_name: str = "input", allow_none: bool = True
    ) -> np.ndarray:
        """
        Replicate out a scalar to an array of shape ``shape``

        :param Any inp:                     scalar or array-like of input data
        :param Tuple[int] shape:            tuple defining array shape that input should be expanded to
        :param Optional[str] var_name:      Name of the variable to include in error messages. Default: "input"
        :param Optional[bool] allow_none:   If ``True``, then ``None`` is permitted as argument for ``inp``. Otherwise an error will be raised. Default: ``True``, allow ``None``

        :return ndarray:                    ``inp``, replicated to the correct shape

        :raises AssertionError:             If ``inp`` is shaped incompatibly to be replicated to the desired shape
        :raises AssertionError:             If ``inp`` is ``None`` and ``allow_none`` is ``False``
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
        Replicate out a scalar to a desired size

        :param Any inp:                     scalar or array-like
        :param int size:                    Size that input should be expanded to
        :param Optional[str] var_name:      Name of the variable to include in error messages. Default: "input"
        :param Optional[bool] allow_none:   If ``True``, allow None as a value for ``inp``. Otherwise and error will be raised. Default: ``True``, allow ``None``

        :return ndarray:                    Array of ``inp``, possibly expanded to the desired size

        :raises AssertionError:             If ``inp`` is incompatibly shaped to expand to the desired size
        :raises AssertionError:             If ``inp`` is ``None`` and ``allow_none`` is ``False``
        """
        return self._expand_to_shape(inp, (size,), var_name, allow_none)

    def _expand_to_net_size(
        self, inp, var_name: str = "input", allow_none: bool = True
    ) -> np.ndarray:
        """
        Replicate out a scalar to the size of the layer

        :param Any inp:                     scalar or array-like
        :param Optional[str] var_name:      Name of the variable to include in error messages. Default: "input"
        :param Optionbal[bool] allow_none:  If ``True``, allow ``None`` as a value for ``inp``. Otherwise an error will be raised. Default: ``True``, allow ``None``

        :return ndarray:                    Values of ``inp``, replicated out to the size of the current layer

        :raises AssertionError:             If ``inp`` is incompatibly sized to replicate out to the layer size
        :raises AssertionError:             If ``inp`` is ``None``, and ``allow_none`` is ``False``
        """
        return self._expand_to_shape(inp, (self.size,), var_name, allow_none)

    def _expand_to_weight_size(
        self, inp, var_name: str = "input", allow_none: bool = True
    ) -> np.ndarray:
        """
        Replicate out a scalar to the size of the layer's weights

        :param Any inp:                     scalar or array-like
        :param Optional[str] var_name:      Name of the variable to include in error messages. Default: "input"
        :param Optionbal[bool] allow_none:  If ``True``, allow ``None`` as a value for ``inp``. Otherwise an error will be raised. Default: ``True``, allow ``None``

        :return ndarray:                    Values of ``inp``, replicated out to the size of the current layer

        :raises AssertionError:             If ``inp`` is incompatibly sized to replicate out to the layer size
        :raises AssertionError:             If ``inp`` is ``None``, and ``allow_none`` is ``False``
        """
        return self._expand_to_shape(inp, (self.size, self.size), var_name, allow_none)

    ### --- String representations

    def __str__(self):
        return '{} object: "{}" [{} {} in -> {} internal -> {} {} out]'.format(
            self.__class__.__name__,
            self.name,
            self.size_in,
            self.input_type.__name__,
            self.size,
            self.size_out,
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
        Abstract method to evolve the state of this layer

        This method must be overridden to produce a concrete :py:class:`Layer` subclass. The :py:class:`evolve` method is the main interface for simulating a layer. It must accept an input time series which determines the signals injected into the layer as input, and return an output time series representing the output of the layer.

        :param Optional[TimeSeries] ts_input:   (TxM) External input trace to use when evolving the layer
        :param Optional[float] duration:        Duration in seconds to evolve the layer. If not provided, then ``num_timesteps`` or the duration of ``ts_input`` is used to determine evolution time
        :param Optional[int] num_timesteps:     Number of time steps to evolve the layer, in units of ``.dt``. If not provided, then ``duration`` or the duration of ``ts_input`` is used to determine evolution time

        :return TimeSeries:                     (TxN) Output of this layer
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
        Reset the internal clock of this layer to 0
        """
        self._timestep = 0

    def randomize_state(self):
        """
        Randomize the internal state of this layer

        Unless overridden, this method randomizes the layer state based on the current state, using a Normal distribution with std. dev. of 20% of the current state values
        """
        # create random initial state with a gaussian distribution with mean the values that were given and std the 20% of the absolute value
        self.state = np.random.normal(
            self.state, np.abs(self.state) * 0.02, size=(self.size,)
        )

    def reset_all(self):
        """
        Reset both the internal clock and the internal state of the layer
        """
        self.reset_time()
        self.reset_state()

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        Convert parameters of this layer to a dict if they are relevant for reconstructing an identical layer

        The base class :py:class:`.Layer` configures the dictionary, by storing attributes :py:attr:`~.Layer.weights`; :py:attr:`~.Layer.dt`; :py:attr:`~.Layer.noise_std`; :py:attr:`~.Layer.name`; and :py:attr:`~.Layer.class_name`. To enable correct saving / loading of your derived :py:class:`.Layer` subclass, you should first call :py:meth:`self.super().to_dict` and then store all additional arguments to :py:meth:`__init__` required by your class to instantiate an identical object.

        :return Dict:   A dictionary that can be used to reconstruct the layer
        """
        config = {}
        if isinstance(self.weights, np.ndarray):
            config["weights"] = self.weights.tolist()
        else:
            config["weights"] = self.weights

        config["dt"] = self.dt
        config["noise_std"] = self.noise_std
        config["name"] = self.name

        config["class_name"] = self.class_name

        return config

    def save(self, config: Dict, filename: str):
        """
        Save a set of parameters to a ``json`` file

        :param Dict config:     Dictionary of attributes to be saved
        :param str filename:    Path of file where parameters are stored
        """
        with open(filename, "w") as f:
            json.dump(config, f)

    def save_layer(self, filename: str):
        """
        Obtain layer paramters from `.to_dict` and save in a ``json`` file

        :param str filename:    Path of file where parameters are to be stored
        """
        config = self.to_dict()
        assert isinstance(config, dict), (
            self.start_print
            + "This should not have happened. If you encounter this statement, please "
            + f"the developers of this package. ({self.class_name})"
        )
        self.save(config, filename)

    @classmethod
    def load_from_file(cls: Any, filename: str, **kwargs) -> "cls":
        """
        Generate an instance of a :py:class:`.Layer` subclass, with parameters loaded from a file

        :param Any cls:         A :py:class:`.Layer` subclass. This class will be used to reconstruct a layer based on the parameters stored in `filename`
        :param str filename:    Path to the file where parameters are stored
        :param kwargs:          Any keyword arguments of the class `.__init__` method where the parameter stored in the file should be overridden

        :return `.Layer`: Instance of `.Layer` subclass with parameters loaded from ``filename``
        """
        # - Load dict from file
        with open(filename, "r") as f:
            config = json.load(f)

        # - Instantiate new class member from dict
        return cls.load_from_dict(config, **kwargs)

    @classmethod
    def load_from_dict(cls: Any, config: Dict, **kwargs) -> "cls":
        """
        Generate instance of a :py:class:`.Layer` subclass with parameters loaded from a dictionary

        :param Any cls:         A :py:class:`.Layer` subclass. This class will be used to reconstruct a layer based on the parameters stored in ``filename``
        :param Dict config:     Dictionary containing parameters of a :py:class:`.Layer` subclass
        :param kwargs:          Any keyword arguments of the class :py:meth:`.__init__` method where the parameters from ``config`` should be overridden

        :return `.Layer`:       Instance of `.Layer` subclass with parameters from ``config``
        """
        # - Overwrite parameters with kwargs
        config = dict(config, **kwargs)

        # - Remove class name from dict
        config.pop("class_name")
        return cls(**config)

    def reset_state(self):
        """
        Reset the internal state of this layer

        Sets `.state` attribute to all zeros
        """
        self.state = np.zeros(self.size)

    #### --- Properties

    @property
    def class_name(self) -> str:
        """
        (str) Class name of ``self``
        """
        # - Determine class name by removing "<class '" and "'>" and the package information
        return str(self.__class__).split("'")[1].split(".")[-1]

    @property
    def start_print(self):
        """
        (str) Return a string containing the layer subclass name and the layer `.name` attribute
        """
        return f"{self.class_name} '{self.name}': "

    @property
    def output_type(self):
        """
        (Type[TimeSeries]) Output :py:class:`.TimeSeries` subclass emitted by this layer.
        """
        return TSContinuous

    @property
    def input_type(self):
        """
        (Type[TimeSeries]) Input :py:class:`.TimeSeries` subclass accepted by this layer.
        """
        return TSContinuous

    @property
    def size(self) -> int:
        """
        (int) Number of units in this layer (N)
        """
        return self._size

    @property
    def size_in(self) -> int:
        """
        (int) Number of input channels accepted by this layer (M)
        """
        return self._size_in

    @property
    def size_out(self) -> int:
        """
        (int) Number of output channels produced by this layer (O)
        """
        return self._size_out

    @property
    def dt(self) -> float:
        """
        (float) Simulation time step of this layer
        """
        return self._dt

    @dt.setter
    def dt(self, fNewDt: float):
        self._dt = to_scalar(fNewDt)

    @property
    def weights(self) -> np.ndarray:
        """
        (ndarray) Weights encapsulated by this layer (MxN)
        """
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
        if new_w.size != self.size_in * self.size:
            raise ValueError(
                self.start_print
                + f"new_w` must be of shape {(self.size_in, self.size)}"
            )

        # - Save weights with appropriate size
        self._weights = np.reshape(new_w, (self.size_in, self.size))

    @property
    def state(self):
        """
        (ndarray) Internal state of this layer (N)
        """
        return self._state

    @state.setter
    def state(self, new_state):
        assert (
            np.size(new_state) == self.size
        ), "Layer `{}`: `new_state` must have {} elements".format(self.name, self.size)

        self._state = new_state

    @property
    def noise_std(self):
        """
        (float) Noise injected into the state of this layer during evolution

        This value represents the standard deviation of a white noise process. When subclassing :py:class:`Layer`, this value should be corrected by the :py:attr:`.dt` attribute
        """
        return self._noise_std

    @noise_std.setter
    def noise_std(self, new_noise_std):
        self._noise_std = to_scalar(new_noise_std)

    @property
    def t(self):
        """
        (float) The current evolution time of this layer
        """
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
