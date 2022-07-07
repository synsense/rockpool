"""
Classes to manage time series data
"""

## -- Import statements

# - Built-ins
import copy
import collections.abc
from pathlib import Path
from tempfile import TemporaryFile
from typing import (
    Union,
    List,
    Tuple,
    Optional,
    Iterable,
    TypeVar,
    Type,
    Dict,
    Hashable,
    Any,
)
from warnings import warn

# - Third party libraries
import numpy as np
import scipy.interpolate as spint

# - Plotting backends
_global_plotting_backend = None
try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    _MPL_AVAILABLE = True
    _global_plotting_backend = "matplotlib"
except ModuleNotFoundError:
    _MPL_AVAILABLE = False

try:
    import holoviews as hv

    _HV_AVAILABLE = True
    if not _MPL_AVAILABLE:
        _global_plotting_backend = "holoviews"
except ModuleNotFoundError:
    _HV_AVAILABLE = False
    if not _MPL_AVAILABLE:
        _global_plotting_backend = None


# - Define exports
__all__ = [
    "TimeSeries",
    "TSEvent",
    "TSContinuous",
    "TSDictOnDisk",
    "set_global_ts_plotting_backend",
    "get_global_ts_plotting_backend",
    "load_ts_from_file",
]

TS = TypeVar("TS", bound="TimeSeries")
# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

### -- Code for setting plotting backend

# - Absolute tolerance, e.g. for comparing float values
_TOLERANCE_ABSOLUTE = 1e-9
_TOLERANCE_RELATIVE = 1e-6

# - Global plotting backend
def set_global_ts_plotting_backend(backend: Union[str, None], verbose=True):
    """
    Set the plotting backend for use by :py:class:`TimeSeries` classes

    :param str backend:     One of {"holoviews", "matplotlib"}
    :param bool verbose:    If ``True``, print feedback about the backend. Default: ``True``
    """
    global _global_plotting_backend

    if backend in ("holoviews", "holo", "Holoviews", "HoloViews", "hv"):
        if _HV_AVAILABLE:
            _global_plotting_backend = "holoviews"
            if verbose:
                print("Global plotting backend has been set to holoviews.")
        else:
            raise RuntimeError("Holoviews is not available.")
    elif backend in ("matplotlib", "mpl", "mp", "pyplot", "plt"):
        if _MPL_AVAILABLE:
            _global_plotting_backend = "matplotlib"
            if verbose:
                print("Global plotting backend has been set to matplotlib.")
        else:
            raise RuntimeError("Matplotlib is not available.")

    elif backend is None:
        _global_plotting_backend = None
        if verbose:
            print("No global plotting backend is set.")

    else:
        raise ValueError("Plotting backend not recognized.")


def get_global_ts_plotting_backend() -> str:
    """
    Return a string representing the current plotting backend

    :return str:    Current plotting backend. One of  {"holoviews", "matplotlib"}
    """
    global _global_plotting_backend
    return _global_plotting_backend


def load_ts_from_file(
    path: Union[str, Path], expected_type: Optional[str] = None
) -> "TimeSeries":
    """
    Load a timeseries object from an ``npz`` file

    :param Union[str, Path] path:       Filepath to load file
    :param Optional[str] expected_type: Specify expected type of timeseires (:py:class:`TSContinuous` or py:class:`TSEvent`). Default: ``None``, use whichever type is loaded.

    :return TimeSeries: Loaded time series object
    :raises TypeError:  Unsupported or unexpected type
    """
    # - Load npz file from specified path
    try:
        # Loading from temporary files may require "rewinding".
        path.seek(0)
    except AttributeError:
        pass
    loaded_data = np.load(path)

    # - Check for expected type
    try:
        loaded_type = loaded_data["str_type"].item()
    except KeyError:
        try:
            loaded_type = loaded_data["strType"].item()
        except KeyError:
            type_key = [k for k in loaded_data if k.startswith("type")]
            if type_key:
                loaded_type = type_key[0][5:]
            else:
                if expected_type is not None:
                    loaded_type = expected_type
                    warn(
                        f"Cannot determine type of Timeseries at {path}. "
                        + f"Will assume expected type ('{expected_type}')."
                    )
                else:
                    raise KeyError(f"Cannot determine type of Timeseries at {path}.")

    if expected_type is not None:
        if not loaded_type == expected_type:
            raise TypeError(
                "Timeseries at `{}` is of type `{}`, which does not match expected type `{}`.".format(
                    path, loaded_type, expected_type
                )
            )

    if "name" in loaded_data:
        name = loaded_data["name"].item()
    else:
        name_keys = [k for k in loaded_data if k.startswith("name")]
        if name_keys:
            name = name_keys[0][5:]
        else:
            name = "unnamed"

    if loaded_type == "TSContinuous":
        if "interp_kind" in loaded_data:
            interp_kind = loaded_data["interp_kind"].item()
        else:
            interp_kind_keys = [k for k in loaded_data if k.startswith("interp_kind")]
            if interp_kind_keys:
                interp_kind = interp_kind_keys[0][12:]
            else:
                interp_kind = "linear"

        ts = TSContinuous(
            times=loaded_data["times"],
            samples=loaded_data["samples"],
            t_start=loaded_data["t_start"].item(),
            t_stop=loaded_data["t_stop"].item(),
            interp_kind=interp_kind,
            periodic=loaded_data["periodic"].item(),
            name=name,
        )

        if "trial_start_times" in loaded_data:
            ts.trial_start_times = loaded_data["trial_start_times"]

    elif loaded_type == "TSEvent":
        ts = TSEvent(
            times=loaded_data["times"],
            channels=loaded_data["channels"],
            t_start=loaded_data["t_start"].item(),
            t_stop=loaded_data["t_stop"].item(),
            periodic=loaded_data["periodic"].item(),
            num_channels=loaded_data["num_channels"].item(),
            name=name,
        )

        if "trial_start_times" in loaded_data:
            ts.trial_start_times = loaded_data["trial_start_times"]

    else:
        raise TypeError("Type `{}` not supported.".format(loaded_type))

    return ts


def _extend_periodic_times(
    t_start: float, t_stop: float, series: "TimeSeries"
) -> np.ndarray:
    """
    Replicate out a periodic time base for later trimming, to ensure that the original time base is repeated correctly

    :param float t_start:       Desired start time of the new series
    :param float t_stop:        Desired end time of the new series
    :param TimeSeries series:   The periodic :py:class:`TimeSeries` to replicate

    :return np.array:           A vector of times corresponding to the replicated time base
    """
    # - Repeat events sufficiently often
    # Number of additional required repetitions to append before and after
    num_reps_after = (
        int(np.ceil((t_stop - series.t_start) / series.duration))
        if t_stop > series.t_stop
        else 1
    )
    num_reps_before = (
        int(np.ceil((series.t_start - t_start) / series.duration))
        if t_start < series.t_start
        else 0
    )
    num_reps_total = num_reps_before + num_reps_after

    # - Correct times so they extend over the prolongued period and do not repeat
    # Enumerate periods so that originally defined period is 0
    periods = np.arange(num_reps_total) - num_reps_before
    correct_periods = series.duration * np.repeat(periods, series.times.size)

    return np.tile(series.times, num_reps_total) + correct_periods


## - Convenience method to return a nan array
def full_nan(shape: Union[tuple, int]) -> np.array:
    """
    Build an all-NaN array

    :param ArrayLike[int] shape:    The desired shape of the NaN matrix

    :return np.array:               The all-NaN matrix
    """
    a = np.empty(shape)
    a.fill(np.nan)
    return a


### --- TimeSeries base class


class TimeSeries:
    """
    Base class to represent a continuous or event-based time series. You should use the subclasses `.TSContinuous` and
    `.TSEvent` to represent continuous-time and event-based time series, respectively. See :ref:`/basics/time_series.ipynb` for futher explanation and examples.
    """

    def __init__(
        self,
        times: ArrayLike = [],
        periodic: bool = False,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        plotting_backend: Optional[str] = None,
        name: str = "unnamed",
    ):
        """
        Represent a continuous or event-based time series

        :param ArrayLike times:                 [Tx1] vector of time samples
        :param bool periodic:                   Treat the time series as periodic around the end points. Default: ``False``
        :param Optional[float] t_start:         If not ``None``, the series start time is ``t_start``, otherwise ``times[0]``
        :param Optional[float] t_stop:          If not ``None``, the series stop time is ``t_stop``, otherwise ``times[-1]``
        :param Optional[str] plotting_backend:  Determines plotting backend. If ``None``, backend will be chosen automatically based on what is available.
        :param str name:                        Name of the `.TimeSeries` object. Default: "unnamed"
        """

        # - Convert time trace to numpy arrays
        times = np.atleast_1d(times).flatten().astype(float)

        if (np.diff(times) < 0).any():
            raise ValueError(
                f"TimeSeries `{name}`: The time trace must be not decreasing"
            )

        # - Assign attributes
        self._times = times
        self.periodic = periodic
        self.name = name
        self._t_start = (
            (0 if np.size(times) == 0 else times[0])
            if t_start is None
            else float(t_start)
        )
        self.t_stop = (
            (self.t_start if np.size(times) == 0 else times[-1])
            if t_stop is None
            else float(t_stop)
        )
        self.set_plotting_backend(
            plotting_backend if plotting_backend is not None else None, verbose=False
        )

    def delay(self: TS, offset: Union[int, float], inplace: bool = False) -> TS:
        """
        Return a copy of ``self`` that is delayed by an offset

        For delaying self, use the ``inplace`` argument, or ``.times += ...`` instead.

        :param float Offset:    Time by which to offset this time series
        :param bool inplace:    If ``True``, conduct operation in-place (Default: ``False``; create a copy)
        :return TimeSeries:     New `.TimeSeries`, delayed
        """
        if not inplace:
            series = self.copy()
        else:
            series = self

        # - Store previous t_start and t_stop
        t_start_old = series.t_start
        t_stop_old = series.t_stop

        # - Shift time trace
        if not self.isempty():
            series.times += offset
        # - Shift t_start and t_stop
        series._t_start = t_start_old + offset
        series._t_stop = t_stop_old + offset

        return series

    def start_at_zero(self: TS, inplace: bool = False) -> TS:
        """
        Convenience function that calls the `~.TimeSeries.delay` method such that ``self.t_start`` falls at ``0``.

        :return TimeSeries:     New TimeSeries, with t_start at 0
        """
        return self.delay(offset=-self.t_start, inplace=inplace)

    def start_at(self: TS, t_start: float, inplace: bool = False) -> TS:
        """
        Convenience function that calls the `~.TimeSeries.delay` method such that ``self.t_start`` falls at ``t_start``.

        :param float t_start:   Time to which ``self.t_start`` should be shifted;
        :param bool inplace:    If ``True``, conduct operation in-place (Default: ``False``; create a copy)
        :return TimeSeries:     New `.TimeSeries`, delayed
        """
        return self.delay(offset=t_start - self.t_start, inplace=inplace)

    def isempty(self) -> bool:
        """
        Test if this `.TimeSeries` object is empty

        :return bool: ``True`` iff the `.TimeSeries` object contains no samples
        """
        return np.size(self.times) == 0

    def print(self):
        """Print an overview of the time series"""
        print(self.__repr__())

    def set_plotting_backend(self, backend: Union[str, None], verbose: bool = True):
        """
        Set which plotting backend to use with the `~.TimeSeries.plot` method

        :param str backend:     Specify a backend to use. Supported: {"holoviews", "matplotlib"}
        :param bool verbose:    If True, print feedback about which backend has been set
        """
        if backend in ("holoviews", "holo", "Holoviews", "HoloViews", "hv"):
            if _HV_AVAILABLE:
                self._plotting_backend = "holoviews"
                if verbose:
                    print(
                        "{} `{}`: Plotting backend has been set to holoviews.".format(
                            type(self).__name__, self.name
                        )
                    )
            else:
                raise RuntimeError("Holoviews is not available.")

        elif backend in ("matplotlib", "mpl", "mp", "pyplot", "plt"):
            if _MPL_AVAILABLE:
                self._plotting_backend = "matplotlib"
                if verbose:
                    print(
                        "{} `{}`: Plotting backend has been set to matplotlib.".format(
                            type(self).__name__, self.name
                        )
                    )
            else:
                raise RuntimeError("Matplotlib is not available.")

        elif backend is None:
            self._plotting_backend = None
            if verbose:
                print(
                    "{} `{}`: Using global plotting backend.".format(
                        type(self).__name__, self.name
                    )
                )

        else:
            raise ValueError("Plotting backend not recognized.")

    def copy(self: TS) -> TS:
        """
        Return a deep copy of this time series

        :return TimeSeries: copy of `self`
        """
        return copy.deepcopy(self)

    def contains(self, times: Union[int, float, ArrayLike]) -> bool:
        """
        Does the time series contain the time range specified in the given time trace?
        Always true for periodic series

        :param ArrayLike times: Array-like containing time points

        :return bool:           True iff all specified time points are contained within this time series
        """
        return self.periodic or (
            self.t_start <= np.min(times) and self.t_stop >= np.max(times)
        )

    def _modulo_period(
        self, times: Union[ArrayLike, float, int]
    ) -> Union[ArrayLike, float, int]:
        """_modulo_period - Calculate provided times modulo `self.duration`"""
        return self.t_start + np.mod(times - self.t_start, self.duration)

    def __len__(self):
        return self._times.size

    @classmethod
    def concatenate_t(
        cls: Type[TS],
        series: Iterable[TS],
        offset: Union[None, float, Iterable[Union[float, None]]] = None,
    ) -> TS:
        """
        Append multiple TimeSeries objects in time to a new series

        :param Iterable series:                     Time series to be tacked at the end of each other. These series must have the same number of channels.
        :param Union[None, float, Iterable] offset: Offset to be introduced between time traces. First value corresponds to delay of first time series.
        :return TimeSeries:                         Time series with data from series in ``series``
        """
        # - Convert `series` to list, to be able to extract information about objects
        if isinstance(series, cls):
            series = [series]
        elif isinstance(series, collections.abc.Iterable):
            series = list(series)
        else:
            cls_name = str(cls).split("'")[1].split(".")[-1]
            raise TypeError(f"{cls_name}: `series` must be of type {cls_name}.")

        # - Determine t_start of first series, to avoid wrong delays.
        #   Determine class to enable calling the method through `TimeSeries` parent class.
        try:
            t_start = series[0].t_start
            subclass = series[0].__class__
        except IndexError:  # `series` is empty
            return cls()

        new_series = subclass(t_start=t_start)
        return new_series.append_t(series, offset=offset, inplace=False)

    @classmethod
    def load(
        cls: Type[TS], path: Union[str, Path], expected_type: Optional[str] = None
    ) -> TS:
        """
        Load TimeSeries object from file. If called from a subclass of :py:class'TimeSeries`,
        the type of the stored object must match that of the method class.

        :param Union[str, Path] path:           Path to load from.
        :param Optional[str] expected_type:     Specify expected type of timeseires (:py:class:`TSContinuous` or py:class:`TSEvent`). Can only be set if method is called from py:class:`TimeSeries` class. Default: ``None``, use whichever type is loaded.

        :return TimeSeries: Loaded time series object
        :raises TypeError:  Unsupported or unexpected type
        :raises TypeError: Argument `expected_type` is defined if class is not `TimeSeries`.
        """
        if cls != TimeSeries:
            if expected_type is not None:
                raise TypeError(
                    "Argument `expected_type` can only be provided when calling this "
                    + "method from `TimeSeries` class."
                )
            # - Extract class name as string
            expected_type = str(cls).strip("'<>").split(".")[-1]

        return load_ts_from_file(path, expected_type)

    @property
    def times(self):
        """(ArrayLike[float]) Array of sample times"""
        return self._times

    @times.setter
    def times(self, new_times: ArrayLike):
        # - Check time trace for correct size
        if np.size(new_times) != np.size(self._times):
            raise ValueError(
                f"TSContinuous `{self.name}`: "
                + "New time trace must have the same number of elements as the original trace."
            )

        # - Make sure time trace is sorted
        if (np.diff(new_times) < 0).any():
            raise ValueError(
                f"TSContinuous `{self.name}`: "
                + "The time trace must be sorted and not decreasing"
            )

        # - Store new time trace
        self._times = np.atleast_1d(new_times).flatten().astype(float)

        if np.size(self._times) > 0:
            # - Fix t_start and t_stop
            self._t_start = min(self._t_start, new_times[0])
            self._t_stop = max(self._t_stop, new_times[-1])

    @property
    def t_start(self) -> float:
        """(float) Start time of time series"""
        return self._t_start

    @t_start.setter
    def t_start(self, new_start):
        try:
            # - Largest allowed value for new_start
            max_start = self._times[0] if self._times.size > 0 else self._t_stop
            if new_start < max_start:
                self._t_start = new_start
            elif new_start - max_start < _TOLERANCE_ABSOLUTE:
                self._t_start = max_start
            else:
                raise ValueError(
                    "TimeSeries `{}`: t_start must be less or equal to {}. It was {}.".format(
                        self.name, max_start, new_start
                    )
                )
        except AttributeError:
            # - If self._t_stop is not defined yet (instantiation)
            self._t_start = float(new_start)

    @property
    def t_stop(self) -> float:
        """(float) Stop time of time series (final sample)"""
        return self._t_stop

    @t_stop.setter
    def t_stop(self, new_stop):
        # - Smallest allowed value for new_stop
        min_stop = self._times[-1] if self._times.size > 0 else self._t_start
        if new_stop >= min_stop:
            self._t_stop = new_stop
        elif min_stop - new_stop < _TOLERANCE_ABSOLUTE:
            self._t_stop = min_stop
        else:
            raise ValueError(
                "TimeSeries `{}`: t_stop must be greater or equal to {}. It was {}.".format(
                    self.name, min_stop, new_stop
                )
            )

    @property
    def duration(self) -> float:
        """(float) Duration of TimeSeries"""
        return self._t_stop - self._t_start

    @property
    def plotting_backend(self):
        """(str) Current plotting backend"""
        return (
            self._plotting_backend
            if self._plotting_backend is not None
            else _global_plotting_backend
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: Optional[str] = None):
        # - Default name: 'unnamed'
        self._name = "unnamed" if new_name is None else new_name


### --- Continuous-valued time series


class TSContinuous(TimeSeries):
    """
    Represents a continuously-sampled time series. Mutliple time series can be represented by a single `.TSContinuous` object, and have identical time bases. Temporal periodicity is supported. See :ref:`/basics/time_series.ipynb` for further explanation and examples.

    :Examples:

    Build a linearly-increasing time series that extends from 0 to 1 second

    >>> time_base = numpy.linspace(0, 1, 100)
    >>> samples = time_base
    >>> ts = TSContinuous(time_base, samples)

    Build a periodic time series as a sinusoid

    >>> time_base = numpy.linspace(0, 2 * numpy.pi, 100)
    >>> samples = numpy.sin(time_base)
    >>> ts = TSContinuous(time_base, samples, periodic = True)

    Build an object containing five random time series

    >>> time_base = numpy.linspace(0, 1, 100)
    >>> samples = numpy.random.rand((100, 5))
    >>> ts = TSContinuous(time_base, samples)

    Manipulate time series using standard operators

    >>> ts + 5
    >>> ts - 3
    >>> ts * 2
    >>> ts / 7
    >>> ts // 3
    >>> ts ** 2
    >>> ts1 + ts2
    ...

    Manipulate time series data in time

    >>> ts.delay(4)
    >>> ts.clip(start, stop, [channel1, channel2, channel3])

    Combine time series data

    >>> ts1.append_t(ts2)    # Appends the second time series, along the time axis
    >>> ts1.append_c(ts2)    # Appends the second time series as an extra channel

    .. note:: All :py:class:`TSContinuous` manipulation methods return a copy by default. Most methods accept an optional `inplace` flag, which if ``True`` causes the operation to be performed in place.

    Resample a time series using functional notation, list notation, or using the :py:func:`.resample` method.

    >>> ts(0.5)
    >>> ts([0, .1, .2, .3])
    >>> ts(numpy.array([0, .1, .2, .3]))
    >>> ts[0.5]
    >>> ts[0, .1, .2, .3]
    >>> ts.resample(0.5)
    >>> ts.resample([0, .1, .2, .3])

    Resample using slice notation

    >>> ts[0:.1:1]

    Resample and select channels simultaneously

    >>> ts[0:.1:1, :3]

    """

    _samples = []

    def __init__(
        self,
        times: Optional[ArrayLike] = None,
        samples: Optional[ArrayLike] = None,
        num_channels: Optional[int] = None,
        periodic: bool = False,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        name: str = "unnamed",
        units: Optional[str] = None,
        interp_kind: str = "previous",
        fill_value: str = "extrapolate",
    ):
        """
        TSContinuous - Represents a continuously-sample time series, supporting interpolation and periodicity.

        :param ArrayLike times:             [Tx1] vector of time samples
        :param ArrayLike samples:           [TxM] matrix of values corresponding to each time sample
        :param Optional[in] num_channels:   If ``samples`` is None, determines the number of channels of ``self``. Otherwise it has no effect at all.
        :param bool periodic:               Treat the time series as periodic around the end points. Default: ``False``
        :param Optional[float] t_start:     If not ``None``, the series start time is ``t_start``, otherwise ``times[0]``
        :param Optional[float] t_stop:      If not ``None``, the series stop time is ``t_stop``, otherwise ``times[-1]``
        :param str name:                    Name of the `.TSContinuous` object. Default: ``"unnamed"``
        :param Optional[str] units:         Units of the `.TSContinuous` object. Default: ``None``
        :param str interp_kind:             Specify the interpolation type. Default: ``"previous"``
        :param str fill_value:              Specify the method to fill values outside sample times. Default: ``"extrapolate"``. **Sampling beyond `.t_stop` is still not permitted**

        If the time series is not periodic (the default), then NaNs will be returned for any values outside `t_start` and `t_stop`.
        """

        if times is None:
            times = np.array([])

        if samples is None:
            num_channels = 0 if num_channels is None else num_channels
            samples = np.zeros((0, num_channels))

        # - Convert everything to numpy arrays
        times = np.atleast_1d(times).flatten().astype(float)
        samples = np.atleast_1d(samples)

        # - Check arguments
        if np.any(np.diff(times) < 0):
            raise ValueError(
                f"TSContinuous `{name}`: The time trace must be sorted and not decreasing."
            )

        # - Initialize superclass
        super().__init__(
            times=times, periodic=periodic, t_start=t_start, t_stop=t_stop, name=name
        )

        # - Assign attributes
        self.interp = None
        self.fill_value = fill_value
        self._interp_kind = interp_kind
        self.samples = samples.astype("float")  # Also creates an interpolator
        self.units = units

        # - Default: Throw exceptions when sampled output contains `NaN`s
        self.beyond_range_exception = True
        # - Default: Change sample times that are slightly out of range
        self.approx_limit_times = True

    ## -- Alternative constructor for clocked time series
    @staticmethod
    def from_clocked(
        samples: np.ndarray,
        dt: float,
        t_start: float = 0.0,
        periodic: bool = False,
        name: str = None,
        interp_kind: str = "previous",
    ) -> "TSContinuous":
        """
        Convenience method to create a new continuous time series from a clocked sample.

        ``samples`` is an array of clocked samples, sampled at a regular interval ``dt``. Each sample is assumed to occur at the **start** of a time bin, such that the first sample occurs at ``t = 0`` (or ``t = t_start``). A continuous time series will be returned, constructed using ``samples``, and filling the time ``t = 0`` to ``t = N*dt``, with ``t_start`` and ``t_stop`` set appropriately.

        :param np.ndarray samples:  A clocked set of contiguous-time samples, with a sample interval of ``dt``. ``samples`` must be of shape ``[T, C]``, where ``T`` is the number of time bins, and ``C`` is the number of channels.
        :param float dt:            The sample interval for ``samples``
        :param float t_start:       The time of the first sample.
        :param bool periodic:       Flag specifying whether or not the time series will be generated as a periodic series. Default:``False``, do not generate a periodic time series.
        :param Optional[str] name:  Optional string to set as the name for this time series. Default: ``None``
        :param str interp_kind:     String specifying the interpolation method to be used for the returned time series. Any string accepted by `scipy.interp1d` is accepted. Default: `"previous"`, sample-and-hold interpolation.

        :return `.TSContinuous` :   A continuous time series containing ``samples``.
        """
        if samples is None or np.size(samples) == 0:
            raise TypeError(
                "TSContinuous.from_clocked: `samples` must not be empty or `None`."
            )

        # - Ensure that `samples` is an ndarray
        samples = np.atleast_1d(samples)

        # - Build a time base
        time_base = np.arange(0, np.shape(samples)[0]) * dt + t_start

        # - Return a continuous time series
        return TSContinuous(
            time_base,
            samples,
            t_stop=time_base[-1] + dt,
            periodic=periodic,
            name=name,
            interp_kind=interp_kind,
        )

    ## -- Methods for plotting and printing

    def plot(
        self,
        times: Optional[Union[int, float, ArrayLike]] = None,
        target: Optional[Union["mpl.axes.Axes", "hv.Curve", "hv.Overlay"]] = None,
        channels: Optional[Union[ArrayLike, int]] = None,
        stagger: Optional[Union[float, int]] = None,
        skip: Optional[int] = None,
        dt: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """
        Visualise a time series on a line plot

        :param Optional[ArrayLike] times: Time base on which to plot. Default: time base of time series
        :param Optional target:  Axes (or other) object to which plot will be added.
        :param Optional[ArrayLike] channels:  Channels of the time series to be plotted.
        :param Optional[float] stagger: Stagger to use to separate each series when plotting multiple series. (Default: `None`, no stagger)
        :param Optional[int] skip: Skip several series when plotting multiple series
        :param Optiona[float] dt: Resample time series to this timestep before plotting

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """
        if dt is not None and times is None:
            times = np.arange(self.t_start, self.t_stop, dt)
            samples = self(times)
        elif times is not None:
            samples = self(times)
        else:
            times = self.times
            samples = self.samples

        if channels is not None:
            samples = samples[:, channels]

        if skip is not None and skip != 0:
            samples = samples[:, ::skip]

        if stagger is not None and stagger != 0:
            samples = samples + np.arange(0, samples.shape[1] * stagger, stagger)

        if target is None:
            # - Determine plotting backend
            if self._plotting_backend is None:
                backend = _global_plotting_backend
            else:
                backend = self._plotting_backend

            # - Handle holoviews plotting
            if backend == "holoviews":
                if kwargs == {}:
                    vhCurves = [
                        hv.Curve((times, data)).redim(x="Time") for data in samples.T
                    ]
                else:
                    vhCurves = [
                        hv.Curve((times, data)).redim(x="Time").options(*args, **kwargs)
                        for data in samples.T
                    ]

                if len(vhCurves) > 1:
                    return hv.Overlay(vhCurves).relabel(group=self.name)
                else:
                    return vhCurves[0].relabel(self.name)

            elif backend == "matplotlib":
                # - Add `self.name` as label only if a label is not already present
                kwargs["label"] = kwargs.get("label", self.name)

                # - Get current axes
                ax = plt.gca()

                # - Set the ylabel, if it isn't already set
                if ax.get_ylabel() == "" and self.units is not None:
                    ax.set_ylabel(self.units)

                # - Set the xlabel, if it isn't already set
                if ax.get_xlabel() == "":
                    ax.set_xlabel("Time (s)")

                # - Set the title, if it isn't already set
                if ax.get_title() == "" and self.name != "unnamed":
                    ax.set_title(self.name)

                # - Set the extent of the time axis
                ax.set_xlim(times[0], times[-1])

                # - Plot the curves
                return ax.plot(times, samples, **kwargs)

            else:
                raise RuntimeError(
                    f"TSContinuous: `{self.name}`: No plotting back-end set."
                )

        else:
            # - Infer current plotting backend from type of `target`
            if _HV_AVAILABLE and isinstance(target, (hv.Curve, hv.Overlay)):
                if kwargs == {}:
                    for data in samples.T:
                        target *= hv.Curve((times, data)).redim(x="Time")
                else:
                    for data in samples.T:
                        target *= (
                            hv.Curve((times, data))
                            .redim(x="Time")
                            .options(*args, **kwargs)
                        )
                return target.relabel(group=self.name)

            elif _MPL_AVAILABLE and isinstance(target, mpl.axes.Axes):
                # - Add `self.name` as label only if a label is not already present
                kwargs["label"] = kwargs.get("label", self.name)
                target.plot(times, samples, **kwargs)
                return target
            else:
                raise TypeError(
                    f"TSContinuous: `{self.name}`: Unrecognized type for `target`. "
                    + "It must be matplotlib Axes or holoviews Curve or Overlay and "
                    + "the corresponding backend must be installed in your environment."
                )

    def print(
        self,
        full: bool = False,
        num_first: int = 4,
        num_last: int = 4,
        limit_shorten: int = 10,
    ):
        """
        Print an overview of the time series and its values

        :param bool full:          Print all samples of ``self``, no matter how long it is
        :param int num_first:      Shortened version of printout contains samples at first `num_first` points in `.times`
        :param int num_last:       Shortened version of printout contains samples at last `num_last` points in `.times`
        :param int limit_shorten:  Print shortened version of self if it comprises more than `limit_shorten` time points and if `full` is False
        """

        s = "\n"
        if len(self.times) <= 10 or full:
            summary = s.join(
                [
                    "{}: \t {}".format(t, samples)
                    for t, samples in zip(self.times, self.samples)
                ]
            )
        else:
            summary0 = s.join(
                [
                    "{}: \t {}".format(t, samples)
                    for t, samples in zip(
                        self.times[:num_first], self.samples[:num_first]
                    )
                ]
            )
            summary1 = s.join(
                [
                    "{}: \t {}".format(t, samples)
                    for t, samples in zip(
                        self.times[-num_last:],
                        self.samples[-num_last + int(self.periodic) :],
                    )
                ]
            )
            if self.periodic:
                summary1 += f"\n{self.times[-1]}: \t {self(self.times[-1])}"
                summary1 += "\n\t (...periodic...)"
            summary = summary0 + "\n\t...\n" + summary1
        print(self.__repr__() + "\n" + summary)

    def to_dict(
        self,
        dtype_times: Union[None, str, type, np.dtype] = None,
        dtype_samples: Union[None, str, type, np.dtype] = None,
    ) -> Dict:
        """
        Store data and attributes of this :py:class:`.TSContinuous` in a :py:class:`dict`.

        :param Union[None, str, type, np.dtype] dtype_times:    Data type in which `times` are to be returned, for example to save space.
        :param Union[None, str, type, np.dtype] dtype_samples:  Data type in which `samples` are to be returned, for example to save space.
        :return:    Dict with data and attributes of this :py:class:`.TSContinuous`.
        """

        if dtype_times is not None:
            # Make sure that broadcast values in `times` are not beyond `t_stop` and `t_start`
            times = self.times.astype(dtype_times)
            # Cannot simply clip `times` because of rounding issues.
            t_start = np.clip(self.t_start, None, np.min(times))
            t_stop = np.clip(self.t_stop, np.max(times), None)
        else:
            times = self.times
            t_start = self.t_start
            t_stop = self.t_stop
        if dtype_samples is not None:
            samples = self.samples.astype(dtype_samples)
        else:
            samples = self.samples

        # - Collect attributes in dict
        attributes = {
            "times": times,
            "samples": samples,
            "t_start": np.array(t_start),
            "t_stop": np.array(t_stop),
            f"interp_kind_{self._interp_kind}": np.array([]),
            "periodic": np.array(self.periodic),
            f"name_{self.name}": np.array([]),
            f"type_TSContinuous": np.array(
                []
            ),  # Indicate that this object is TSContinuous
        }

        # - Some modules add a `trial_start_times` attribute to the object.
        if hasattr(self, "trial_start_times"):
            attributes["trial_start_times"] = np.asarray(self.trial_start_times)

        return attributes

    def save(
        self,
        path: Union[str, Path],
        verbose: bool = False,
        dtype_times: Union[None, str, type, np.dtype] = None,
        dtype_samples: Union[None, str, type, np.dtype] = None,
    ):
        """
        Save this time series as an ``npz`` file using np.savez

        :param str path:        Path to save file
        :param Union[None, str, type, np.dtype] dtype_times:    Data type in which `times` are to be stored, for example to save space.
        :param Union[None, str, type, np.dtype] dtype_samples:  Data type in which `samples` are to be stored, for example to save space.
        """

        # - Collect attributes in dict
        attributes = self.to_dict(dtype_times=dtype_times, dtype_samples=dtype_samples)

        # - Write the file
        np.savez(path, **attributes)

        if verbose:
            missing_ending = path.split(".")[-1] != "npz"  # np.savez will add ending
            print(
                "TSContinuous `{}` has been stored in `{}`.".format(
                    self.name, path + missing_ending * ".npz"
                )
            )

    ## -- Methods for manipulating timeseries

    def clip(
        self,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        channels: Union[int, ArrayLike, None] = None,
        include_stop: bool = True,
        sample_limits: bool = True,
        inplace: bool = False,
    ) -> "TSContinuous":
        """
        Return a TSContinuous which is restricted to given time limits and only contains events of selected channels

        :param float t_start:       Time from which on events are returned
        :param float t_stop:        Time until which events are returned
        :param ArrayLike channels:  Channels of which events are returned
        :param bool include_stop:   True -> If there are events with time ``t_stop`` include them. False -> Exclude these samples. Default: True.
        :param bool sample_limits:  If True, make sure that a sample exists at ``t_start`` and, if ``include_stop`` is True, at ``t_stop``, as long as not both are None.
        :param bool inplace:        Conduct operation in-place (Default: False; create a copy)

        :return TSContinuous:       clipped_series:     New TSContinuous clipped to bounds
        """
        # - Create a new time series, or modify this time series
        if not inplace:
            clipped_series = self.copy()
        else:
            clipped_series = self

        # Handle `None` time limits
        t_start: float = self.t_start if t_start is None else t_start
        include_stop = True if t_stop is None else include_stop
        t_stop: float = self.t_stop if t_stop is None else t_stop

        # - Ensure time bounds are sorted
        t_start, t_stop = sorted((t_start, t_stop))

        # - Handle periodic time series
        times_to_choose: np.ndarray = (
            _extend_periodic_times(t_start, t_stop, clipped_series)
            if clipped_series.periodic
            else clipped_series.times
        )

        # - Mark which times lie within bounds
        times_in_limits: np.ndarray = np.logical_and(
            times_to_choose >= t_start, times_to_choose < t_stop
        )
        if include_stop:
            # - Include samples at time `t_stop`
            times_in_limits[times_to_choose == t_stop] = True
        # - Pick matching times
        times: np.ndarray = times_to_choose[times_in_limits]
        if sample_limits:
            add_start: bool = times.size == 0 or times[0] > t_start
            if not clipped_series.contains(t_start):
                warn(
                    f"TSContinuous '{self.name}`: `t_start` beyond interpolation-range. "
                )
            add_stop: bool = times.size == 0 or (times[-1] < t_stop and include_stop)
            if not clipped_series.contains(t_stop):
                warn(
                    f"TSContinuous '{self.name}`: `t_stop` beyond interpolation-range. "
                )
            # - Only generate new array once
            if add_start and add_stop:
                times = np.r_[t_start, times, t_stop]
            elif add_start:
                times = np.r_[t_start, times]
            elif add_stop:
                times = np.r_[times, t_stop]

        # - Sample at the chosen time points and return
        clipped_series.resample(times, channels, inplace=True)
        # - Update t_start and t_stop
        clipped_series._t_start = t_start
        clipped_series._t_stop = t_stop

        return clipped_series

    def resample(
        self,
        times: Union[int, float, ArrayLike],
        channels: Union[int, float, ArrayLike, None] = None,
        inplace: bool = False,
    ) -> "TSContinuous":
        """
        Return a new time series sampled to the supplied time base

        :param ArrayLike times:                 T desired time points to resample
        :param Optional[ArrayLike] channels:    Channels to be used. Default: None (use all channels)
        :param bool inplace:                    True -> Conduct operation in-place (Default: False; create a copy)
        :return TSContinuous:                   Time series resampled to new time base and with desired channels.
        """
        if not inplace:
            resampled_series = self.copy()
        else:
            resampled_series = self

        # - Make sure `times` is array
        times = np.atleast_1d(times)

        # - Resample time series
        if channels is None:
            resampled_series._samples = self(times)
        else:
            # - Convert to 1d array so that integer as index also results in 2D samples
            channels = np.atleast_1d(channels)
            if self.num_channels == 0 and channels.size > 0:
                # - Handle empty series
                raise IndexError(
                    f"TSContinuous `{self.name}` does not have any channels."
                )
            try:
                resampled_series._samples = self(times)[:, channels]
            except IndexError:
                raise IndexError(
                    f"TSContinuous `{self.name}`: "
                    + f"Channels must be between 0 and {self.num_channels - 1}."
                )
        resampled_series._times = times
        if times.size > 0:
            resampled_series._t_start = times[0]
            resampled_series._t_stop = times[-1]
        resampled_series._create_interpolator()
        return resampled_series

    def to_clocked(
        self,
        dt: float,
    ) -> np.ndarray:
        """
        Resample this time series to a synchronous clock and return the samples

        This method will generate a time base that begins at :py:attr:`.t_start` and extends to at least :py:attr:`.t_stop`, sampled on a clock defined by ``dt``. The time series will be resampled to that time base, using the defined interpolation method, and the clocked samples will be returned as a raster.

        Args:
            dt (float): The desired clock time step, in seconds

        Returns:
            np.ndarray: The samples from the clocked time series
        """
        # - Build a time base that spans the extend of the time series
        num_samples = np.ceil((self.t_stop - self.t_start) / dt)
        time_base = np.arange(num_samples) * dt + self.t_start

        # - Resample the time series to the supplied time base and return
        return self.resample(time_base).samples

    ## -- Methods for combining time series

    def merge(
        self,
        other_series: Union["TSContinuous", Iterable["TSContinuous"]],
        remove_duplicates: bool = True,
        inplace: bool = False,
    ) -> "TSContinuous":
        """
        Merge other time series to this one, by interleaving in time. Maintain each time series' time values and channel IDs.

        :param Union["TSContinuous", Iterable["TSContinuous"]] other_series:    time series that is merged to self or iterable thereof to merge multiple series
        :param bool remove_duplicates:                                If ``True``, time points in other series that are also in ``self.times`` are discarded. Otherwise they are included in the new time trace and come after the corresponding points of self.times.
        :param bool inplace:                                          Conduct operation in-place (Default: ``False``; create a copy)

        :return TSContinuous:                       The merged time series
        """

        # - Create a new time series, or modify this time series
        if not inplace:
            merged_series = self.copy()
        else:
            merged_series = self

        # - Ensure there is a list of timeseries to work on
        if isinstance(other_series, TSContinuous):
            series_list = [merged_series, other_series]
        else:
            try:
                series_list = [merged_series] + list(other_series)
            except TypeError:
                raise TypeError(
                    f"TSContinuous `{self.name}`: `other_series` must be `TSContinuous`"
                    " or iterable thereof."
                )
            # - Check series class
            if not all(isinstance(series, TSContinuous) for series in series_list):
                raise TypeError(
                    f"TSContinuous `{self.name}`: Can only merge with `TSContinuous` objects."
                )

        # - Handle empty `self`
        if merged_series.num_channels == 0 and len(merged_series) == 0:
            num_channels = max(series.num_channels for series in series_list)
            merged_series._samples = np.zeros((0, num_channels))

        # - Handle empty other series (without changing original object) and channel numbers
        for i_series, series in enumerate(series_list[1:]):
            if series.num_channels == 0 and len(series) == 0:
                series = series.copy()
                series._samples = np.zeros((0, merged_series.num_channels))
                series_list[i_series + 1] = series.copy()
            elif series.num_channels != merged_series.num_channels:
                raise ValueError(
                    f"TSContinuous `{self.name}`: `other_series` must include "
                    f"the same number of traces ({merged_series.num_channels}). "
                    f"Other series number {i_series} has {series.num_channels}."
                )

        if remove_duplicates:
            # - For each time point in each series a boolean array indicating whether points are used or removed
            use_points_list = [
                np.ones(series.times.size, bool) for series in series_list
            ]
            # - Iterate over series pairwise and remove duplicate time points in second series of pair
            for i_s0, series0 in enumerate(series_list[:-1]):
                for series1, use_points1 in zip(
                    series_list[i_s0 + 1 :], use_points_list[i_s0 + 1 :]
                ):
                    self._mask_duplicate_time_points(series0, series1, use_points1)
            times_series = [
                series._times[use] for series, use in zip(series_list, use_points_list)
            ]
            samples_series = [
                series._samples[use]
                for series, use in zip(series_list, use_points_list)
            ]
        else:
            times_series = [series._times for series in series_list]
            samples_series = [series._samples for series in series_list]

        # - Merge time traces and samples
        times_new: np.ndarray = np.concatenate(times_series)
        samples_new: np.ndarray = np.concatenate(samples_series, axis=0)

        #  - Indices for sorting new time trace and samples. Use mergesort as stable sorting algorithm.
        idcs_sorted: np.ndarray = np.argsort(times_new, kind="mergesort")

        # - Update data of new time series
        merged_series._times = times_new[idcs_sorted]
        merged_series._samples = samples_new[idcs_sorted]
        merged_series._t_start: float = min(series.t_start for series in series_list)
        merged_series._t_stop: float = max(series.t_stop for series in series_list)

        # - Create new interpolator
        merged_series._create_interpolator()

        # - Return merged TS
        return merged_series

    @staticmethod
    def _mask_duplicate_time_points(series0, series1, use_points1):
        if not (series0.t_start > series1.t_stop or series0.t_stop < series1.t_start):
            # Determine region of overlap
            overlap: np.ndarray = np.where(
                (series0.times >= series1.t_start) & (series0.times <= series1.t_stop)
            )[0]
            # Array of bools indicating which sampled time points of series1 do not occur in series0
            not_unique = [(t == series0.times[overlap]).any() for t in series1.times]
            # Update which points of series1 are to be used
            use_points1[not_unique] = False

    def append_c(
        self, other_series: "TSContinuous", inplace: bool = False
    ) -> "TSContinuous":
        """
        Append another time series to this one, along the samples axis (i.e. add new channels)

        :param TSContinuous other_series:   Another time series. Will be resampled to the time base of ``self``
        :param bool inplace:                Conduct operation in-place (Default: ``False``; create a copy)

        :return `TSContinuous`:             Current time series, with new channels appended
        """
        # - Check other_series
        if not isinstance(other_series, TSContinuous):
            raise TypeError(
                f"TSContinuous `{self.name}`: "
                + "`other_series` must be a TSContinuous object."
            )

        # - Resample other_series to own time base
        other_samples = other_series(self.times)

        # - Create a new time series, or modify this time series
        if not inplace:
            appended_series = self.copy()
        else:
            appended_series = self

        # - Combine samples
        appended_series.samples = np.concatenate(
            (np.atleast_2d(appended_series.samples), other_samples), 1
        )

        # - Create new interpolator
        appended_series._create_interpolator()

        # - Return appended TS
        return appended_series

    def append_t(
        self,
        other_series: Union["TSContinuous", Iterable["TSContinuous"]],
        offset: Union[float, Iterable[Union[float, None]], None] = None,
        inplace: bool = False,
    ) -> "TSContinuous":
        """
        Append another time series to this one, along the time axis

        :param Union["TSContinuous", Iterable[TSContinuous]] other_series:    Time series to be tacked on to the end of the called series object. These series must have the same number of channels as ``self`` or be empty.
        :param Union[float, Iterable[float], Iterable[None], None] offset:    If not None, defines distance between last sample of one series and first sample of the next. Otherwise the offset will be the median of all timestep sizes of the first of the two series, or 0 if that series has len < 2.
        :param bool inplace:                                                  Conduct operation in-place (Default: ``False``; create a copy)

        :return TSContinuous:                                                 Time series containing data from ``self``, with the other series appended in time
        """

        # - Ensure there is a list of timeseries to work on
        if isinstance(other_series, TSContinuous):
            other_series = [other_series]
        else:
            try:
                other_series = list(other_series)
            except TypeError:
                raise TypeError(
                    f"TSContinuous `{self.name}`: `other_series` must be `TSContinuous`"
                    " or iterable thereof."
                )
            # - Check series class
            if not all(isinstance(series, TSContinuous) for series in other_series):
                raise TypeError(
                    f"TSContinuous `{self.name}`: Can only merge with `TSContinuous` objects."
                )

        # - Same for offsets
        if not isinstance(offset, collections.abc.Iterable):
            offset_list = [offset] * len(other_series)
        else:
            offset_list = list(offset)
            if len(offset_list) != len(other_series):
                warn(
                    f"TSContinuous `{self.name}`: Numbers of provided offsets and "
                    + "TSContinuous objects do not match. Will ignore excess elements."
                )
        for i, (prev_series, series, offset) in enumerate(
            zip([self] + other_series[:-1], other_series, offset_list)
        ):
            if offset is None:
                if len(series) > 0 and len(prev_series) > 1:
                    # - If ``self`` is empty then append new elements directly. Otherwise leave an offset
                    #   corresponding to the median distance between time points in `self._times`.
                    offset_list[i] = np.median(np.diff(prev_series._times))
                else:
                    # - No offset with empty series
                    offset_list[i] = 0

        # - Translate offsets so that they correspond to indiviual delays for each series
        # Delay for first appended series:
        delay1 = offset_list[0] + self.t_stop - other_series[0].t_start
        delay_list = [delay1]
        # Add delays for other lists
        for prev_series, curr_series, offset in zip(
            other_series[:-1], other_series[1:], offset_list[1:]
        ):
            # Time at which previous series stops
            stop_previous = delay_list[-1] + prev_series.t_stop
            # Delay for current series
            delay_list.append(stop_previous + offset - curr_series.t_start)
        other_series = [
            series.delay(delay) for series, delay in zip(other_series, delay_list)
        ]

        # - Let ``self.merge()`` do the rest
        try:
            return self.merge(
                other_series=other_series, remove_duplicates=False, inplace=inplace
            )
        except TypeError:
            # - Provide matching exception
            raise TypeError(
                f"TSContinuous `{self.name}`: Can only append `TSContinuous` objects."
            )

    ## -- Internal methods

    def _create_interpolator(self):
        """
        Build an interpolator for the samples in this TimeSeries.

        Replaces the current interpolator.
        """
        if np.size(self.times) == 0:
            self.interp = lambda t: None

        elif np.size(self.times) == 1:

            # - Handle sample for single time step (`interp1d` would cause error)
            def single_sample(t):
                times = np.array(t).flatten()
                samples = np.empty((times.size, self.num_channels))
                samples.fill(np.nan)
                if self._fill_value == "extrapolate":
                    assign_val = np.logical_and(
                        self.t_start <= times, times <= self.t_stop
                    )
                else:
                    assign_val = times == self.times[0]
                samples[assign_val] = self.samples[0]
                return samples

            self.interp = single_sample

        else:
            # - Construct interpolator
            self.interp = spint.interp1d(
                self._times,
                self._samples,
                kind=self._interp_kind,
                axis=0,
                assume_sorted=True,
                bounds_error=False,
                copy=False,
                fill_value=self._fill_value,
            )

    def _interpolate(self, times: Union[int, float, ArrayLike]) -> np.ndarray:
        """
        Interpolate the time series to the provided time points

        :param ArrayLike times: Array of ``T`` desired interpolated time points

        :return np.ndarray:     Array of interpolated values. Will have the shape ``TxN``, where ``N`` is the number of channels in ``self``
        """

        # Make sure `times` is an array
        times = np.array(times)

        # - Handle empty series
        if self.isempty():
            return np.zeros((np.size(times), 0))

        # - Enforce periodicity
        if self.periodic and self.duration > 0:
            times = (times - self._t_start) % self.duration + self._t_start

        # Time points that define the range of the interpolator
        t_first = self.times[0]
        t_last = self.t_stop if self._interp_kind == "previous" else self.times[-1]

        # Time points outside of this range
        is_early = np.array(times < t_first)
        is_late = np.array(times > t_last)

        # - Correct time points that are slightly out of range
        if self.approx_limit_times:

            tol = min(_TOLERANCE_ABSOLUTE, _TOLERANCE_RELATIVE * self.duration)
            # Find values in `times` that are slightly before first or slightly after
            # last sample
            t_first_approx = t_first - tol
            t_last_approx = t_last + tol
            set_t_first = np.logical_and(is_early, times >= t_first_approx)
            set_t_last = np.logical_and(is_late, times <= t_last_approx)
            times = np.where(set_t_first, t_first, times)
            times = np.where(set_t_last, t_last, times)
            if np.logical_or(set_t_first, set_t_last).any():
                warn(
                    f"TSContinuous `{self.name}`: Some of the requested time points "
                    + "were slightly outside the time range of this series (by at "
                    + f"most {tol} s) and were approximated by "
                    + f"the first or last time point of this series. To prevent this "
                    + f"behavior, set the `approx_limit_times` attribute to `False`."
                )
                is_early[set_t_first] = False
                is_late[set_t_last] = False

        # - Warn or throw exception if output contains `NaN`s
        if is_early.any() or is_late.any():
            error_msg = (
                f"TSContinuous `{self.name}`: Some of the requested time points are "
                + "beyond the first and last time points of this series and cannot "
                + "be sampled.\n"
                + "If you think that this is due to rounding errors, try setting "
                + "the `approx_limit_times` attribute to `True`.\n"
            )
            if self.beyond_range_exception:
                raise ValueError(
                    error_msg
                    + "If you want to sample at these time points anyway, you can "
                    + "set the `beyond_range_exception` attribute of this time series "
                    + "to `False` and will receive `NaN` as values."
                )
            else:
                warn(
                    error_msg
                    + "Will return `NaN` for these time points."
                    + "To raise a ValueError in situations like this, set the "
                    + "`beyond_range_exception` attribute of this time series to `True`."
                )

        samples = np.reshape(self.interp(times), (-1, self.num_channels))

        # - Catch invalid times, replace with NaN
        invalid_times = np.logical_or(
            np.array(times) < self.t_start, np.array(times) > self.t_stop
        )
        samples[invalid_times, :] = np.nan

        # - Return the sampled data
        return samples

    def _compatible_shape(self, other_samples) -> np.ndarray:
        """
        Attempt to make ``other_samples`` a compatible shape to ``self.samples``.

        :param ArrayLike other_samples: Samples to convert

        :return np.ndarray:             Array the same shape as ``self.samples``
        :raises:                        ValueError if broadcast fails
        """
        try:
            return np.broadcast_to(other_samples, self.samples.shape).copy()
        except ValueError:
            raise ValueError(
                f"TSContinuous `{self.name}`: Input data (shape {other_samples.shape})"
                f" could not be broadcast to samples shape ({self.samples.shape})."
            )

    ## -- Magic methods

    def __call__(self, times: Union[int, float, ArrayLike]):
        """
        ts(tTime1, tTime2, ...) - Interpolate the time series to the provided time points

        :param ArrayLike t_time:    Scalar, list or ``np.array`` of ``T`` desired interpolated time points
        :return np.array:           Array of interpolated values. Will have the shape ``TxN``
        """
        return self._interpolate(times)

    def __getitem__(
        self,
        # indices_time: Union[ArrayLike, float, slice, None] = None,
        # indices_channel: Union[ArrayLike, int, slice, None] = None,
        indices: Union[
            Tuple[
                Union[ArrayLike, float, slice, None], Union[ArrayLike, int, slice, None]
            ],
            ArrayLike,
            float,
            slice,
            None,
        ] = None,
    ) -> "TSContinuous":
        """
        ts[indices_time, indices_channel] - Interpolate the time series to the provided time points or, if a slice is provided between given limits with given step size. Use channels provided in indices_channel, in matching order.

        :param ArrayLike[float] indices_time:       float, array-like or slice of T desired interpolated time points
        :param ArrayLike[int] indices_channel:      int, array-like or slice of desired channels in desired order

        :return TSContinuous:                       TSContinuous with chosen time points and channels
        """
        # - Handle format of funciton argument
        if isinstance(indices, tuple):
            if len(indices) == 0:
                indices_time = indices_channel = None
            elif len(indices) == 1:
                # Assume indices refer to time
                indices_time = indices[0]
                indices_channel = None
            elif len(indices) == 2:
                # Both time and channel indices are given
                indices_time, indices_channel = indices
            else:
                raise IndexError(
                    f"TSContinuous: `{self.name}`: Supports at most 2 indices"
                    + " (times and channels)."
                )
        else:
            # Assume indices refer to time
            indices_time = indices
            indices_channel = None

        # - Handle channel indices
        if isinstance(indices_channel, slice):
            ch_start = 0 if indices_channel.start is None else indices_channel.start
            ch_stop = (
                self.num_channels
                if indices_channel.stop is None
                else indices_channel.stop
            )
            ch_step = 1 if indices_channel.step is None else indices_channel.step
            channels = np.arange(ch_start, ch_stop, abs(ch_step))
            if ch_step < 0:
                # - Invert order of channels
                channels = channels[::-1]
        else:
            channels = indices_channel

        # - Handle time indices
        if indices_time is None:
            indices_time = slice(None)

        if isinstance(indices_time, slice):
            if indices_time.step is None:
                # - Use `self.clip`
                return self.clip(
                    t_start=indices_time.start,
                    t_stop=indices_time.stop,
                    channels=channels,
                    inplace=False,
                    include_stop=indices_time.stop is None,
                    sample_limits=False,
                )
            else:
                # - Prepare time points for `self.resample`
                t_start: float = (
                    self.t_start if indices_time.start is None else indices_time.start
                )
                t_stop: float = (
                    self.t_stop if indices_time.stop is None else indices_time.stop
                )
                # - Determine time points at which series is sampled
                time_points: np.ndarray = np.arange(
                    t_start, t_stop, abs(indices_time.step)
                )
                # - Make sure time points are within limits
                time_points = time_points[
                    np.logical_and(
                        time_points >= self.t_start, time_points < self.t_stop
                    )
                ]
                # - Invert order if step is negative
                if indices_time.step < 0:
                    time_points = time_points[::-1]
        else:
            time_points = indices_time

        return self.resample(time_points, channels, inplace=False)

    def __repr__(self) -> str:
        """
        Return a string representation of this object

        :return str: String description
        """
        if self.isempty():
            beginning: str = f"Empty TSContinuous object `{self.name}` "
        else:
            beginning: str = "{}periodic TSContinuous object `{}` ".format(
                int(not self.periodic) * "non-", self.name
            )
        return beginning + "from t={} to {}. Samples: {}. Channels: {}".format(
            self.t_start, self.t_stop, self.samples.shape[0], self.num_channels
        )

    # - Iteration
    def __iter__(self):
        """
        Yield tuples of sample times and values

        :yield Tuple[float, float]:   Yields a tuple [times, samples]

        """
        for t, val in zip(self.times, self.samples):
            yield (t, val)

    ## -- Operator overloading

    # - Addition

    def __add__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        return self.copy().__iadd__(other_samples)

    def __radd__(self, other_samples: TimeSeries) -> TimeSeries:
        return self + other_samples

    def __iadd__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        if isinstance(other_samples, TSContinuous):
            other_samples = self._compatible_shape(other_samples(self.times))
        else:
            other_samples = self._compatible_shape(other_samples)

        # - Treat NaNs as zero
        is_nan_self = np.isnan(self.samples)
        is_nan_other = np.isnan(other_samples)
        self.samples[is_nan_self] = 0
        other_samples[is_nan_other] = 0

        # - Perform addition
        new_samples = self.samples + other_samples
        self.samples = new_samples

        # - Fill in nans
        self.samples[np.logical_and(is_nan_self, is_nan_other)] = np.nan

        return self

    # - Subtraction

    def __sub__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        return self.copy().__isub__(other_samples)

    def __rsub__(self, other_samples: TimeSeries) -> TimeSeries:
        return -(self - other_samples)

    def __isub__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        if isinstance(other_samples, TSContinuous):
            other_samples = self._compatible_shape(other_samples(self.times))
        else:
            other_samples = self._compatible_shape(other_samples)

        # - Treat NaNs as zero
        is_nan_self = np.isnan(self.samples)
        is_nan_other = np.isnan(other_samples)
        self.samples[is_nan_self] = 0
        other_samples[is_nan_other] = 0

        # - Perform subtraction
        self.samples -= other_samples

        # - Fill in nans
        self.samples[np.logical_and(is_nan_self, is_nan_other)] = np.nan

        return self

    # - Multiplication

    def __mul__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        return self.copy().__imul__(other_samples)

    def __rmul__(self, other_samples: TimeSeries) -> TimeSeries:
        return self * other_samples

    def __imul__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        if isinstance(other_samples, TSContinuous):
            other_samples = self._compatible_shape(other_samples(self.times))
        else:
            other_samples = self._compatible_shape(other_samples)

        # - Propagate NaNs
        is_nan_self = np.isnan(self.samples)
        is_nan_other = np.isnan(other_samples)

        # - Perform multiplication
        self.samples *= other_samples

        # - Fill in nans
        self.samples[np.logical_or(is_nan_self, is_nan_other)] = np.nan

        return self

    # - Division

    def __truediv__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        return self.copy().__itruediv__(other_samples)

    def __rtruediv__(self, other_samples: TimeSeries) -> TimeSeries:
        self_copy = self.copy()
        self_copy.samples = 1 / self_copy.samples
        return self_copy * other_samples

    def __itruediv__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        if isinstance(other_samples, TSContinuous):
            other_samples = self._compatible_shape(
                np.reshape(other_samples(self.times), (np.size(self.times), -1))
            )
        else:
            other_samples = self._compatible_shape(other_samples)

        # - Propagate NaNs
        is_nan_self = np.isnan(self.samples)
        is_nan_other = np.isnan(other_samples)

        # - Perform division
        self.samples /= other_samples

        # - Fill in nans
        self.samples[np.logical_or(is_nan_self, is_nan_other)] = np.nan

        # - Re-create interpolator
        self._create_interpolator()
        return self

    # - Floor division

    def __floordiv__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        return self.copy().__ifloordiv__(other_samples)

    def __rfloordiv__(self, other_samples: TimeSeries) -> TimeSeries:
        self_copy = self.copy()
        self_copy.samples = 1 / self_copy.samples
        return self_copy // (1 / other_samples)

    def __ifloordiv__(self, other_samples: Union[TimeSeries, Any]) -> TimeSeries:
        if isinstance(other_samples, TSContinuous):
            other_samples = self._compatible_shape(other_samples(self.times))
        else:
            other_samples = self._compatible_shape(other_samples)

        # - Propagate NaNs
        is_nan_self = np.isnan(self.samples)
        is_nan_other = np.isnan(other_samples)

        # - Perform division
        self.samples //= other_samples

        # - Fill in nans
        self.samples[np.logical_or(is_nan_self, is_nan_other)] = np.nan

        return self

    # - Matrix multiplication

    def __matmul__(self, matrix) -> TimeSeries:
        return self.copy().__imatmul__(matrix)

    def __rmatmul__(self, matrix) -> TimeSeries:
        raise NotImplementedError

    def __imatmul__(self, matrix) -> TimeSeries:
        self.samples = self.samples @ matrix
        return self

    # - Exponentiation

    def __pow__(self, exponent: Union[TimeSeries, Any]) -> TimeSeries:
        return self.copy().__ipow__(exponent)

    def __rpow__(self, base: Union[np.ndarray, Any]) -> TimeSeries:
        new_series = self.copy()

        base = new_series._compatible_shape(base)

        # - Propagate NaNs
        is_nan_self = np.isnan(new_series.samples)
        is_nan_other = np.isnan(base)

        # - Perform exponentiation
        new_series.samples = base**new_series.samples

        # - Fill in nans
        new_series.samples[np.logical_or(is_nan_self, is_nan_other)] = np.nan

        return new_series

    def __ipow__(self, exponent: Union[TimeSeries, Any]) -> TimeSeries:
        if isinstance(exponent, TSContinuous):
            exponent = self._compatible_shape(exponent(self.times))
        else:
            exponent = self._compatible_shape(exponent)

        # - Propagate NaNs
        is_nan_self = np.isnan(self.samples)
        is_nan_other = np.isnan(exponent)

        # - Perform exponentiation
        self.samples **= exponent

        # - Fill in nans
        self.samples[np.logical_or(is_nan_self, is_nan_other)] = np.nan

        return self

    # - Absolute

    def __abs__(self) -> TimeSeries:
        self_copy = self.copy()
        self_copy.samples = np.abs(self_copy.samples)
        return self_copy

    # - Negative

    def __neg__(self) -> TimeSeries:
        self_copy = self.copy()
        self_copy.samples = -self_copy.samples
        return self_copy

    ## -- Properties

    @property
    def samples(self):
        """(ArrayLike[float]) Value of time series at sampled times"""
        return self._samples

    @samples.setter
    def samples(self, new_samples: ArrayLike):
        # - Make sure that if assigned empty samples array, number of traces is implicityly
        #   with as second dimension of `new_samples`
        if np.size(new_samples) == 0 and np.ndim(new_samples) < 2:
            new_samples = new_samples.reshape(0, 0)

        # - Promote to 2d
        new_samples = np.atleast_2d(new_samples)

        # - Permit a one-dimensional sample input, promote to 2d
        if (new_samples.shape[0] == 1) and (np.size(self.times) > 1):
            new_samples = np.reshape(new_samples, (np.size(self.times), -1))

        # - Check samples for correct size
        if new_samples.shape[0] != np.size(self.times):
            raise ValueError(
                f"TSContinuous `{self.name}`: "
                + "New samples matrix must have the same number of samples as `.times`."
            )

        # - Store new time trace
        self._samples = new_samples

        # - Create a new interpolator
        self._create_interpolator()

    # - Extend setter of times to update interpolator
    @property
    def times(self):
        """(ArrayLike[float]) Array of sample times"""
        return self._times

    @times.setter
    def times(self, new_times: ArrayLike):
        super(TSContinuous, self.__class__).times.fset(self, new_times)

        # - Create a new interpolator
        self._create_interpolator()

    @property
    def num_traces(self):
        """(int) Synonymous to ``num_channels``"""
        return self.samples.shape[1]

    @property
    def num_channels(self):
        """(int) Number of channels (dimension of sample vectors) in this TimeSeries object"""
        return self.samples.shape[1]

    @property
    def max(self):
        """(float) Maximum value of time series"""
        return np.nanmax(self.samples)

    @property
    def min(self):
        """(float) Minimum value of time series"""
        return np.nanmin(self.samples)

    @property
    def beyond_range_exception(self):
        return self._nan_exception

    @beyond_range_exception.setter
    def beyond_range_exception(self, raise_exception: bool):
        try:
            self._nan_exception = bool(raise_exception)
        except (TypeError, ValueError):
            raise TypeError(
                f"TSContinuous `{self.name}`: `beyond_range_exception` must be of boolean type."
            )

    @property
    def approx_limit_times(self):
        return self._approx_limit_times

    @approx_limit_times.setter
    def approx_limit_times(self, approx: bool):
        try:
            self._approx_limit_times = bool(approx)
        except (TypeError, ValueError):
            raise TypeError(
                f"TSContinuous `{self.name}`: `approx_limit_times` must be of boolean type."
            )

    @property
    def fill_value(self):
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        if isinstance(value, str):
            if value != "extrapolate":
                raise ValueError(
                    f"TSContinuous `{self.name}`: "
                    + "Fill_value` must be either `extrapolate` or a fill value to pass to `scipy.interpolate`."
                )

        self._fill_value = value
        if self.interp is not None:
            self._create_interpolator()


### --- Event time series


class TSEvent(TimeSeries):
    """
    Represents a discrete time series, composed of binary events (present or absent). This class is primarily used to represent spike trains or event trains to communicate with spiking neuron layers, or to communicate with event-based computing systems. See :ref:`/basics/time_series.ipynb` for further explanation and examples.

    `.TSEvent` supports multiple channels of event time series encapsulated by a single object, as well as periodic time series.

    :Examples:

    Build a series of several random event times

    >>> times = numpy.cumsum(numpy.random.rand(10))
    >>> ts = TSEvent(times)

    """

    def __init__(
        self,
        times: Optional[ArrayLike] = None,
        channels: Optional[Union[int, ArrayLike]] = None,
        periodic: bool = False,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        name: Optional[str] = None,
        num_channels: Optional[int] = None,
    ):
        """
        Represent discrete events in time

        :param Optional[ArrayLike[float]] times:     ``Tx1`` vector of event times
        :param Optional[ArrayLike[int]] channels:     ``Tx1`` vector of event channels (Default: all events are in channel 0)

        :param bool periodic:               Is this a periodic TimeSeries (Default: False; non-periodic)

        :param float t_start:               Explicitly specify the start time of this series. If ``None``, then ``times[0]`` is taken to be the start time
        :param float t_stop:                Explicitly specify the stop time of this series. If ``None``, then ``times[-1]`` is taken to be the stop time

        :param Optional[str] name:                    Name of the time series (Default: None)

        :param Optional[int] num_channels:            Total number of channels in the data source. If ``None``, the total channel number is taken to be ``max(channels)``
        """

        # - Default time trace: empty
        if times is None:
            times = np.array([])
        else:
            times = np.atleast_1d(times).flatten().astype(float)

        # - Make sure `t_stop` is larger than any provided time point.
        if times.size > 0:
            if t_stop is None:
                raise TypeError(
                    "If `times` is not `None`, `t_stop` must be a float strictly "
                    + "greater than the largest entry in `times`."
                )
            if np.max(times) >= t_stop:
                raise ValueError(
                    f"`t_stop` (here {t_stop}) must be strictly greater than the "
                    + f"largest entry in `times` (here {np.max(times)})."
                )

        # - Default name: 'unnamed'
        name = "unnamed" if name is None else name

        # - Default channel: zero
        if channels is None or np.size(channels) == 0:
            channels = np.zeros(np.size(times))
            min_num_ch = min(np.size(times), 1)
        # - Handle scalar channel
        elif isinstance(channels, int):
            min_num_ch = channels + 1
            channels = np.array([channels for _ in times])
        # - Array-like of channels
        else:
            if np.size(channels) != np.size(times):
                # - Make sure item sizes match
                raise ValueError(
                    f"TSEvent `{name}`: `channels` must have the same number of "
                    + "elements as `times`, be an integer or None."
                )
            else:
                min_num_ch = np.amax(channels) + 1

        if num_channels is None:
            # - Infer number of channels from maximum channel id in channels
            num_channels = min_num_ch
        else:
            if num_channels < min_num_ch:
                raise ValueError(
                    f"TSEvent `{name}`: num_channels must be None or greater than the highest channel ID."
                )

        # - Initialize superclass
        super().__init__(
            times=times, periodic=periodic, t_start=t_start, t_stop=t_stop, name=name
        )

        # - Store total number of channels
        self._num_channels = int(num_channels)

        # - Store channels
        self.channels = np.array(channels, "int").flatten()

    def print(
        self,
        full: bool = False,
        num_first: int = 4,
        num_last: int = 4,
        limit_shorten: int = 10,
    ):
        """
        Print an overview of the time series and its values

        :param bool full:           Print all samples of ``self``, no matter how long it is. Default: ``False``
        :param int limit_shorten:   Print shortened version of ``self`` if it comprises more than ``limit_shorten`` time points and ``full`` is ``False``. Default: 4
        :param int num_first:       Shortened version of printout contains samples at first ``num_first`` points in ``self.times``. Default: 4
        :param int num_last:        Shortened version of printout contains samples at last ``num_last`` points in ``self.times``. Default: 4
        """

        s = "\n"
        if len(self.times) <= 10 or full:
            summary = s.join(
                [f"{t}: \t {ch}" for t, ch in zip(self.times, self.channels)]
            )
        else:
            summary0 = s.join(
                [
                    f"{t}: \t {ch}"
                    for t, ch in zip(self.times[:num_first], self.channels[:num_first])
                ]
            )
            summary1 = s.join(
                [
                    f"{t}: \t {ch}"
                    for t, ch in zip(self.times[-num_last:], self.channels[-num_last:])
                ]
            )
            summary = summary0 + "\n\t...\n" + summary1
        print(self.__repr__() + "\nTime \t Ch.-ID" + "\n" + summary)

    ## -- Methods for plotting and printing

    def plot(
        self,
        time_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        target: Union["mpl.axes.Axes", "hv.Scatter", "hv.Overlay", None] = None,
        channels: Union[ArrayLike, int, None] = None,
        *args,
        **kwargs,
    ):
        """
        Visualise this time series on a scatter plot

        :param Optional[float, float] time_limits:  Tuple with times between which to plot. Default: plot all times
        :param Optional[axis] target:               Object to which plot will be added. Default: new plot
        :param ArrayLike[int] channels:             Channels that are to be plotted. Default: plot all channels
        :param args, kwargs:                        Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """
        # - Filter spikes by time
        if time_limits is None:
            t_start = self.t_start
            t_stop = self.t_stop
        else:
            exception_limits = (
                f"TSEvent `{self.name}`: `time_limits` must be None or tuple "
                + "of length 2."
            )
            try:
                # - Make sure `time_limits` has correct length
                if len(time_limits) != 2:
                    raise ValueError(exception_limits)
                else:
                    t_start = self.t_start if time_limits[0] is None else time_limits[0]
                    t_stop = self.t_stop if time_limits[1] is None else time_limits[1]
            except TypeError:
                raise TypeError(exception_limits)

        # - Choose matching events
        times, channels = self(t_start, t_stop, channels)

        if target is None:
            if self._plotting_backend is None:
                backend = _global_plotting_backend
            else:
                backend = self._plotting_backend
            if backend == "holoviews":
                return (
                    hv.Scatter((times, channels), *args, **kwargs)
                    .redim(x="Time", y="Channel")
                    .relabel(self.name)
                )

            elif backend == "matplotlib":
                # - Add `self.name` as label only if a label is not already present
                kwargs["label"] = kwargs.get("label", self.name)

                # - Get current axes
                ax = plt.gca()

                # - Set the ylabel, if it isn't already set
                if ax.get_ylabel() == "":
                    ax.set_ylabel("Channels")

                # - Set the xlabel, if it isn't already set
                if ax.get_xlabel() == "":
                    ax.set_xlabel("Time (s)")

                # - Set the title, if it isn't already set
                if ax.get_title() == "" and self.name != "unnamed":
                    ax.set_title(self.name)

                # - Set the extent of the time axis
                ax.set_xlim(self.t_start, self.t_stop)

                # - Set the extent of the channels axis
                ax.set_ylim(-1, self.num_channels)

                # - Plot the curves
                return ax.scatter(times, channels, *args, **kwargs)

            else:
                raise RuntimeError(f"TSEvent: `{self.name}`: No plotting back-end set.")

        else:
            # - Infer current plotting backend from type of `target`
            if _HV_AVAILABLE and isinstance(target, (hv.Curve, hv.Overlay)):
                target *= (
                    hv.Scatter((times, channels), *args, **kwargs)
                    .redim(x="Time", y="Channel")
                    .relabel(self.name)
                )
                return target.relabel(group=self.name)
            elif _MPL_AVAILABLE and isinstance(target, mpl.axes.Axes):
                # - Add `self.name` as label only if a label is not already present
                kwargs["label"] = kwargs.get("label", self.name)
                target.scatter(times, channels, *args, **kwargs)
                return target
            else:
                raise TypeError(
                    f"TSEvent: `{self.name}`: Unrecognized type for `target`. "
                    + "It must be matplotlib Axes or holoviews Curve or Overlay and "
                    + "the corresponding backend must be installed in your environment."
                )

    ## -- Methods for manipulating timeseries

    def clip(
        self,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        channels: Union[int, ArrayLike, None] = None,
        remap_channels: bool = False,
        inplace: bool = False,
    ) -> "TSEvent":
        """
        Return a `TSEvent` which is restricted to given time limits and only contains events of selected channels

        If time limits are provided, `.t_start` and `.t_stop` attributes of the new time series will correspond to those. If `remap_channels` is ``True``, channels IDs will be mapped to a continuous sequence of integers starting from 0 (e.g. [1, 3, 6]->[0, 1, 2]). In this case `.num_channels` will be set to the number of different channels in ``channels``. Otherwise `.num_channels` will keep its original values, which is also the case for all other attributes. If `inplace` is True, modify ``self`` accordingly.

        :param Optional[float] t_start:             Time from which on events are returned. Default: `.t_start`
        :param Optional[float] t_stop:              Time until which events are returned. Default: `.t_stop`
        :param Optional[ArrayLike[int]] channels:   Channels of which events are returned. Default: All channels
        :param bool remap_channels:                 Map channel IDs to continuous sequence starting from 0. Set `num_channels` to largest new ID + 1. Default: ``False``, do not remap channels
        :param bool inplace:                        Iff ``True``, the operation is performed in place (Default: False)

        :return `.TSEvent`:                         `.TSEvent` containing events from the requested channels
        """

        if not inplace:
            new_series = self.copy()
        else:
            new_series = self

        # - Extract matching events
        time_data, channel_data = new_series(t_start, t_stop, channels)

        # - Update new timeseries
        new_series._times = time_data
        if t_start is not None:
            new_series._t_start = t_start
        if t_stop is not None:
            new_series._t_stop = t_stop
        if remap_channels:
            if channel_data.size > 0:
                # - Set channel IDs to sequence starting from 0
                unique_channels, channel_indices = np.unique(
                    channel_data, return_inverse=True
                )
                num_channels = unique_channels.size
                new_series._channels = np.arange(num_channels)[channel_indices]
            else:
                new_series._channels = channel_data
            new_series._num_channels = (
                np.unique(channels).size
                if channels is not None
                else np.unique(channel_data).size
            )
        else:
            new_series._channels = channel_data

        return new_series

    def remap_channels(
        self, channel_map: ArrayLike, inplace: bool = False
    ) -> "TSEvent":
        """
        Renumber channels in the :py:class:`.TSEvent`

        Maps channels 0..``self.num_channels-1`` to the channels in ``channel_map``.

        :param ArrayLike[int] channel_map:  List of channels that existing channels should be mapped to, in order. Must be of size ``self.num_channels``.
        :param bool inplace:                Specify whether operation should be performed in place (Default: ``False``, a copy is returned)
        """

        if not inplace:
            new_series = self.copy()
        else:
            new_series = self

        channel_map = np.asarray(channel_map)
        if not channel_map.size == new_series.num_channels:
            raise ValueError(
                f"TSEvent `{new_series.name}`: "
                + f"`channel_map` must be of size {new_series.num_channels}."
            )
        new_series.channels = channel_map[new_series.channels]

        return new_series

    ## -- Methods for finding and extracting data

    def raster(
        self,
        dt: float,
        t_start: float = None,
        t_stop: float = None,
        num_timesteps: int = None,
        channels: np.ndarray = None,
        add_events: bool = False,
    ) -> np.ndarray:
        """
        Return a rasterized version of the time series data, where each data point represents a time step

        Events are represented in a boolean matrix, where the first axis corresponds to time, the second axis to the channel. Events that happen between time steps are projected to the preceding step. If two events happen during one time step within a single channel, they are counted as one, unless ``add_events`` is ``True``.

        Time bins for the raster extend ``[t, t+dt)``, that is **explicitly excluding events that occur at** ``t+dt``. Such events would be included in the following time bin.

        To generate a time trace that corresponds to the raster, you can use :py:func:`numpy.arange` as follows::

            num_timesteps = np.ceil((t_stop - t_start) / dt)
            bin_starts = np.arange(num_timesteps) * dt + t_start
            bin_stops = bin_starts + dt
            bin_mid = bin_starts + dt/2

        Note that the modulo computation is numerically unstable as expressed above. Internally we use a numerically more stable version, with::

            def mod(num, div):
                return (num - div * np.floor(num/div))

            num_timesteps = int(np.ceil((t_stop - t_start) / dt)

        :param float dt:                            Duration of single time step in raster
        :param Optional[float] t_start:             Time where to start raster. Default: None (use ``self.t_start``)
        :param Optional[float] t_stop:              Time where to stop raster. This time point is not included in the raster. Default: ``None`` (use ``self.t_stop``. If ``num_timesteps`` is provided, ``t_stop`` is ignored.
        :param Optional[int] num_timesteps:         Specify number of time steps directly, instead of providing ``t_stop``. Default: ``None`` (use ``t_start``, ``t_stop`` and ``dt`` to determine raster size)
        :param Optional[ArrayLike[int]] channels:   Channels from which data is to be used. Default: ``None`` (use all channels)
        :param bool add_events:                     If ``True``, return an integer raster containing number of events for each time step and channel. Default: ``False``, merge simultaneous events in a single channel, and return a boolean raster
        :param bool endpoint:                       If ``True``, an extra time bin is added to the raster after ``t_stop``, to ensure that any events occurring at ``t_stop`` are included in the raster. Default: ``False``, do not include events occurring at ``t_stop``.

        :return ArrayLike:  event_raster            Boolean matrix with ``True`` indicating presence of events for each time step and channel. If ``add_events == True``, the raster consists of integers indicating the number of events per time step and channel. First axis corresponds to time, second axis to channel.
        """

        # - Numerically stable modulo function
        def mod(num, div):
            return num - div * np.floor(num / div)

        # - Filter time and channels
        t_start = self.t_start if t_start is None else t_start
        if channels is None:
            channels = channels_clip = np.arange(self.num_channels)

        elif np.amax(channels) >= self.num_channels:
            # - Only use channels that are within range of channels of this timeseries
            channels_clip = np.intersect1d(channels, np.arange(self.num_channels))

            # - Channels for which series is not defined
            channels_undefined = np.setxor1d(channels_clip, channels)
            warn(
                f"TSEvent `{self.name}` is not defined for some of the channels provided "
                + f"in `channels` argument ({', '.join(channels_undefined.astype(str))}). "
                + f"Will assume that there are no events for these channels."
            )
        else:
            channels_clip = channels

        # - Determine t_stop and num_timesteps
        if num_timesteps is not None and t_stop is not None:
            # - Check that only one of `t_stop` and `num_timesteps` is provided
            raise ValueError(
                "Only one of `t_stop` and `num_timesteps` may be provided."
            )

        elif num_timesteps is not None:
            # - Use `num_timesteps` to determine `t_stop`
            t_stop = t_start + num_timesteps * dt

        elif t_stop is None:
            # - Use own `t_stop`
            t_stop = self.t_stop

        # - Compute number of raster timesteps
        num_timesteps = int(np.ceil((t_stop - t_start) / dt))

        # - Clip the time series to include only the events of interest
        series = self.clip(
            t_start=t_start,
            t_stop=t_stop,
            channels=channels_clip,
            remap_channels=False,
        )

        # - Create raster for storing event data
        raster_type = int if add_events else bool
        event_raster = np.zeros((num_timesteps, channels.size), raster_type)

        # - Handle empty time series
        if len(series) == 0:
            return event_raster

        # - Select data according to time base
        event_times = series.times
        event_channels = series.channels

        ## -- Convert input events and samples to boolean or integer raster
        # - Only consider rasters that have non-zero length
        if num_timesteps > 0:
            # - Compute indices for event times and filter to valid time bins
            time_indices = np.floor((event_times - t_start) / dt).astype(int)
            time_indices = time_indices[time_indices < num_timesteps]

            if add_events:
                # - Accumulate events per time step and channel
                for idx_t, idx_ch in zip(time_indices, event_channels):
                    event_raster[idx_t, idx_ch] += 1
            else:
                # - Print a warning if there are multiple spikes in one time step and channel
                if (
                    (
                        np.diff(np.c_[time_indices, event_channels], axis=0)
                        == np.zeros(2)
                    )
                    .all(axis=1)
                    .any(axis=0)
                ):
                    print(
                        f"TSEvent `{self.name}`: There are channels with multiple events"
                        + " per time step. Consider using a smaller `dt` or setting `add_events = True`."
                    )
                # - Mark spiking indices with True
                event_raster[time_indices, event_channels] = True

        # - Return the raster
        return event_raster

    def xraster(
        self,
        dt: float,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        channels: Optional[np.ndarray] = None,
        add_events: Optional[bool] = None,
        endpoint: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Generator which ``yield`` s a rasterized time series data, where each data point represents a time step

        Events are represented in a boolean matrix, where the first axis corresponds to time, the second axis to the channel. Events that happen between time steps are projected to the preceding one. If two events happen during one time step within a single channel, they are counted as one.

        :param float dt:                            Duration of single time step in raster
        :param Optional[float] t_start:             Time where to start raster. Default: ``None`` (use ``self.t_start``)
        :param Optional[float] t_stop:              Time where to stop raster. This time point is not included in the raster. Default: ``None`` (use ``self.t_stop``. If ``num_timesteps`` is provided, ``t_stop`` is ignored.
        :param Optional[int] num_timesteps:         Specify number of time steps directly, instead of providing ``t_stop``. Default: ``None`` (use ``t_start``, ``t_stop`` and ``dt`` to determine raster size.
        :param Optional[ArrayLike[int]] channels:   Channels from which data is to be used. Default: ``None`` (use all channels)
        :param Optional[bool] add_events:           If ``True``, return an integer raster containing number of events for each time step and channel. Default: ``False``, merge simultaneous events in a single channel, and return a boolean raster
        :param Optional[bool] endpoint:            If ``True``, an extra time bin is added to the raster after ``t_stop``, to ensure that any events occurring at ``t_stop`` are included in the raster. Default: ``False``, do not include events occurring at ``t_stop``.

        :yields ArrayLike: event_raster - Boolean matrix with ``True`` indicating presence of events for each time step and channel. If ``add_events == True``, the raster consists of integers indicating the number of events per time step and channel. First axis corresponds to time, second axis to channel.
        """
        event_raster = self.raster(
            dt=dt,
            t_start=t_start,
            t_stop=t_stop,
            num_timesteps=num_timesteps,
            channels=channels,
            add_events=add_events,
            endpoint=endpoint,
        )
        yield from event_raster  # Yield one row at a time

    @staticmethod
    def from_raster(
        raster: np.ndarray,
        dt: float = 1.0,
        t_start: float = 0.0,
        t_stop: Optional[float] = None,
        name: Optional[str] = None,
        periodic: bool = False,
        num_channels: Optional[int] = None,
        spikes_at_bin_start: bool = False,
    ) -> "TSEvent":
        """
        Create a `.TSEvent` object from a raster array

        Given a rasterised event time series, with dimensions [TxC], `~.TSEvent.from_raster` will generate a event
        time series as a `.TSEvent` object.

        .. rubric:: Example

        The following code will generate a Poisson event train with 200 time steps of 1ms each, and 20 channels, with a spiking probability of 10% per time bin::

            T = 200
            C = 20
            dt = 1e-3
            spike_prob = 0.1

            raster = np.random.rand((T, C)) > spike_prob
            spikes_ts = TSEvent.from_raster(raster, dt)

        :param np.ndarray raster:           An array of events ``(T, C)``. Each row corresponds to a clocked time step of  `dt` duration. Each bin contains the number of spikes present in that bin
        :param float dt:                    Duration of each time bin in seconds
        :param float t_start:               The start time of the first bin in ``raster``. Default: ``0.``
        :param float t_stop:                The stop time of the time series. Default: the total duration of the provided raster
        :param Optional[str] name:          The name of the returned time series. Default: ``None``
        :param bool periodic:               The ``periodic`` flag passed to the new time series
        :param Optional[int] num_channels:  The ``num_channels`` argument passed to the new time series. Default: ``None``, use the number of channels ``C`` in ``raster``
        :param bool spikes_at_bin_start:    Iff ``True``, then spikes in ``raster`` are considered to occur at the start of the time bin. If ``False``, then spikes occur half-way through each time bin. Default: ``False``, spikes occur half-way through each time bin.

        :return TSEvent: A new `.TSEvent` containing the events in ``raster``
        """

        # - Make sure ``raster`` is a numpy array of integer type
        raster = np.asarray(raster, int)

        # - Reshape if the array is 1d
        if len(raster.shape) == 1:
            raster = np.atleast_2d(raster).T

        # - Compute `t_stop` if not provided
        if t_stop is None:
            t_stop = raster.shape[0] * dt + t_start

        # - Find spike events
        spike_present = raster > 0
        spikes_per_bin = raster[spike_present]
        spikes = np.repeat(np.argwhere(raster), spikes_per_bin, axis=0)

        # - Determine the number of channels
        num_channels = raster.shape[1] if num_channels is None else num_channels

        # - Convert to a new TSEvent object and return
        return TSEvent(
            spikes[:, 0] * dt + t_start + dt / 2 * int(not spikes_at_bin_start),
            spikes[:, 1],
            name=name,
            periodic=periodic,
            num_channels=num_channels,
            t_start=t_start,
            t_stop=t_stop,
        )

    def to_dict(
        self,
        dtype_times: Union[None, str, type, np.dtype] = None,
        dtype_channels: Union[None, str, type, np.dtype] = None,
    ) -> Dict:
        """
        Store data and attributes of this :py:class:`.TSEvent` in a ``Dict``.

        :param Union[None, str, type, np.dtype] dtype_times:    Data type in which ``times`` are to be returned, for example to save space.
        :param Union[None, str, type, np.dtype] dtype_channels:  Data type in which ``channels`` are to be returned, for example to save space.
        :return:    Dict with data and attributes of this :py:class:`.TSEvent`.
        """

        if dtype_times is not None:
            # Make sure that broadcast values in `times` are not beyond `t_stop` and `t_start`
            times = self.times.astype(dtype_times)
            # Cannot simply clip `times` because of rounding issues.
            t_start = np.clip(self.t_start, None, np.min(times))
            res = np.finfo(dtype_times).resolution
            t_stop = np.clip(self.t_stop, np.max(times) + res, None)
        else:
            times = self.times
            t_start = self.t_start
            t_stop = self.t_stop
        if dtype_channels is not None:
            if np.iinfo(dtype_channels).max < self.num_channels:
                raise ValueError(
                    f"TSEvent `{self.name}`: type `{dtype_channels}` not sufficient "
                    + f"for number of channels in this series ({self.num_channels})."
                )
            channels = self.channels.astype(dtype_channels)
        else:
            channels = self.channels

        # - Collect attributes in dict
        attributes = {
            "times": times,
            "channels": channels,
            "t_start": np.array(t_start),
            "t_stop": np.array(t_stop),
            "periodic": np.array(self.periodic),
            "num_channels": np.array(self.num_channels),
            f"name_{self.name}": np.array([]),
            f"type_TSEvent": np.array([]),  # Indicate that the object is TSEvent
        }

        # - Some modules add a `trial_start_times` attribute to the object.
        if hasattr(self, "trial_start_times"):
            attributes["trial_start_times"] = self.trial_start_times

        return attributes

    def save(
        self,
        path: Union[str, Path],
        verbose: bool = False,
        dtype_times: Union[None, str, type, np.dtype] = None,
        dtype_channels: Union[None, str, type, np.dtype] = None,
    ):
        """
        Save this :py:class:`.TSEvent` as an ``npz`` file using :py:meth:`np.savez`

        :param str path:        Path to save file
        :param bool verbose:    Print path information after successfully saving.
        :param Union[None, str, type, np.dtype] dtype_times:    Data type in which `times` are to be stored, for example to save space.
        :param Union[None, str, type, np.dtype] dtype_channels:  Data type in which `channels` are to be stored, for example to save space.
        """

        # - Collect attributes in dict
        attributes = self.to_dict(
            dtype_times=dtype_times, dtype_channels=dtype_channels
        )

        # - Write the file
        np.savez(path, **attributes)

        if verbose:
            missing_ending = path.split(".")[-1] != "npz"  # np.savez will add ending
            print(
                "TSEvent `{}` has been stored in `{}`.".format(
                    self.name, path + missing_ending * ".npz"
                )
            )

    ## -- Methods for combining time series

    def append_c(self, other_series: "TSEvent", inplace: bool = False) -> "TSEvent":
        """
        Append another time series to ``self`` along the channels axis

        The channel IDs in ``other_series`` are shifted by ``self.num_channels``. Event times remain the same.

        :param TSEvent other_series:    :py:class:`TSEvent` or list of :py:class:`TSEvent` that will be appended to ``self``.
        :param bool inplace:  Conduct operation in-place (Default: ``False``; create a copy)

        :return TSEvent:                :py:class:`TSEvent` containing data in ``self``, with other TS appended along the channels axis
        """

        # - Create a new time series, or modify this time series
        if not inplace:
            appended_series = self.copy()
        else:
            appended_series = self

        # - Ensure we have a list of timeseries to work on
        if isinstance(other_series, TSEvent):
            series_list = [appended_series, other_series]
        else:
            try:
                series_list = [appended_series] + list(other_series)
            except TypeError:
                raise TypeError(
                    f"TSEvent `{self.name}`: `other_series` must be `TSEvent` or list thereof."
                )

        # - Check series class
        if not all(isinstance(series, TSEvent) for series in series_list):
            raise TypeError(
                f"TSEvent `{self.name}`: Can only append `TSEvent` objects."
            )

        # - Determine t_start and t_stop
        t_start_new = min(series.t_start for series in series_list)
        t_stop_new = max(series.t_stop for series in series_list)

        # - Determine number of channels for each series
        nums_channels = [series.num_channels for series in series_list]
        # - Shift for each TSEvent's channels
        channel_shifts = np.cumsum([0] + nums_channels[:-1])

        # - Stop if no non-empty series is left
        if not series_list:
            return appended_series

        # - Merge all samples
        times_new = np.concatenate([series.times for series in series_list])
        channels_new = np.concatenate(
            [
                series.channels + shift
                for series, shift in zip(series_list, channel_shifts)
            ]
        )

        # - Sort on time and merge
        sort_indices = np.argsort(times_new)
        appended_series._times = times_new[sort_indices]
        appended_series._channels = channels_new[sort_indices].astype(int)
        appended_series._t_start = t_start_new
        appended_series._t_stop = t_stop_new
        appended_series._num_channels = int(np.sum(nums_channels))

        return appended_series

    def append_t(
        self,
        other_series: Union[TimeSeries, Iterable[TimeSeries]],
        offset: Union[float, Iterable[Union[float, None]], None] = None,
        remove_duplicates: bool = False,
        inplace: bool = False,
    ) -> "TSEvent":
        """
        Append another time series to this one along the time axis

        ``t_start`` from ``other_series`` is shifted to ``self.t_stop + offset``.

        :param TSEvent other_series:                :py:class:`TSEvent` or list of :py:class:`TSEvent` that will be appended to ``self`` along the time axis
        :param Optional[float] offset:              Scalar or iterable with at least the same number of elements as ``other_series``. If scalar, use same value for all timeseries. Event times from ``other_series`` will be shifted by ``self.t_stop + offset``. Default: 0
        :param bool remove_duplicates:              If ``True``, duplicate events will be removed from the resulting timeseries. Duplicates can occur if ``offset`` is negative. Default: ``False``, do not remove duplicate events.
        :param bool inplace:                        If ``True``, conduct operation in-place (Default: ``False``; return a copy)

        :return TSEvent: :py:class:`TSEvent` containing events from ``self``, with other TS appended in time
        """

        # - Ensure we have a list of timeseries to work on
        if isinstance(other_series, TSEvent):
            other_series = [other_series]
        else:
            try:
                other_series = list(other_series)
            except TypeError:
                raise TypeError(
                    f"TSEvent `{self.name}`: `other_series` must be `TSEvent` or list thereof."
                )
        # - Same for offsets
        if not isinstance(offset, collections.abc.Iterable):
            # - Handle `None` offsets
            if offset is None:
                offset = 0
            offset_list = [offset] * len(other_series)
        else:
            offset_list = list(offset)
            if len(offset_list) != len(other_series):
                warn(
                    f"TSEvent `{self.name}`: Numbers of provided offsets and TSEvent "
                    + "objects do not match. Will ignore excess elements."
                )
            # - Handle `None` offsets
            for i, os in enumerate(offset_list):
                if os is None:
                    offset_list[i] = 0

        # - Translate offsets so that they correspond to indiviual delays for each series
        # Delay for first appended series:
        delay1 = offset_list[0] + self.t_stop - other_series[0].t_start
        delay_list = [delay1]
        # Add delays for other lists
        for prev_series, curr_series, offset in zip(
            other_series[:-1], other_series[1:], offset_list[1:]
        ):
            # Time at which previous series stops
            stop_previous = delay_list[-1] + prev_series.t_stop
            # Delay for current series
            delay_list.append(stop_previous + offset - curr_series.t_start)
        # - Let self.merge do the rest
        try:
            return self.merge(
                other_series=other_series,
                delay=delay_list,
                remove_duplicates=remove_duplicates,
                inplace=inplace,
            )
        except TypeError:
            # - Provide matching exception
            raise TypeError(
                f"TSEvent `{self.name}`: Can only append `TSEvent` objects."
            )

    def merge(
        self,
        other_series: Union[TimeSeries, Iterable[TimeSeries]],
        delay: Union[float, Iterable[float]] = 0,
        remove_duplicates: bool = False,
        inplace: bool = False,
    ) -> "TSEvent":
        """
        Merge another :py:class:`TSEvent` into this one so that they may overlap in time

        :param TSEvent other_series:    :py:class:`TSEvent` or list of :py:class:`TSEvent` to merge into ``self``
        :param Union[float, Iterable[float]] delay:   Scalar or iterable with at least the number of elements as other_series. If scalar, use same value for all timeseries. Delay ``other_series`` series by this value before merging.
        :param bool remove_duplicates:  If ``True``, remove duplicate events in resulting timeseries. Default: ``False``, do not remove duplicates.
        :param bool inplace:  If ``True``, operation will be performed in place (Default: ``False``, return a copy)

        :return TSEvent:                ``self`` with new samples included
        """

        # - Create a new time series, or modify this time series
        if not inplace:
            merged_series = self.copy()
        else:
            merged_series = self

        # - Ensure we have a list of timeseries to work on
        if isinstance(other_series, TSEvent):
            series_list = [merged_series, other_series]
        else:
            try:
                series_list = [merged_series] + list(other_series)
            except TypeError:
                raise TypeError(
                    f"TSEvent `{self.name}`: `other_series` must be `TSEvent` or list thereof."
                )
        # - Same for offsets
        if not isinstance(delay, collections.abc.Iterable):
            delay_list = [0] + [delay] * (len(series_list) - 1)
        else:
            delay_list = [0] + list(delay)
            if len(delay_list) != len(series_list):
                warn(
                    f"TSEvent `{self.name}`: Numbers of provided offsets and TSEvent "
                    + "objects do not match. Will ignore excess elements."
                )

        # - Check series class
        if not all(isinstance(series, TSEvent) for series in series_list):
            raise TypeError(
                f"TSEvent `{self.name}`: Can only merge with `TSEvent` objects."
            )

        # - Apply delay
        series_list = [
            series.delay(delay) for series, delay in zip(series_list, delay_list)
        ]
        # - Determine number of channels
        merged_series._num_channels = max(series.num_channels for series in series_list)
        # - Determine t_start and t_stop
        t_start_new = min(series.t_start for series in series_list)
        t_stop_new = max(series.t_stop for series in series_list)
        # - Merge all samples
        times_new = np.concatenate([series.times for series in series_list])
        channels_new = np.concatenate([series.channels for series in series_list])

        # - Remove events with same times and channels
        if remove_duplicates:
            times_new, channels_new = np.unique((times_new, channels_new), axis=1)

        # - Sort on time and merge
        sort_indices = np.argsort(times_new)
        merged_series._times = times_new[sort_indices]
        merged_series._channels = channels_new[sort_indices].astype(int)
        merged_series._t_start = t_start_new
        merged_series._t_stop = t_stop_new

        return merged_series

    ## -- Internal methods

    def _matching_channels(
        self,
        channels: Union[int, ArrayLike, None] = None,
        event_channels: Union[int, ArrayLike, None] = None,
    ) -> np.ndarray:
        """
        Return boolean array of which events match a given channel selection

        :param ArrayLike[int] channels:         Channels of which events are to be indicated ``True``. Default: ``None``, use all channels
        :params ArrayLike[int] event_channels:  Channel IDs for each event. If not provided (Default: ``None``), then use self._channels

        :return ArrayLike[bool]:        A matrix ``TxC`` indicating which events match the requested channels
        """

        if channels is None:
            channels = np.arange(self.num_channels)
        else:
            # - Check `channels` for validity
            if np.asarray(channels).size > 0 and not (
                np.min(channels) >= 0 and np.max(channels) < self.num_channels
            ):
                raise IndexError(
                    f"TSEvent `{self.name}`: `channels` must be between 0 and {self.num_channels}."
                )
        # - Make sure elements in `channels` are unique for better performance
        channels = np.unique(channels)

        # - Use a defined list of event channels, if provided
        if event_channels is None:
            event_channels = self._channels

        # - Boolean array of which events match selected channels
        include_events = np.isin(event_channels, channels)

        return include_events

    ## -- Magic methods

    def __call__(
        self,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        channels: Optional[Union[int, ArrayLike]] = None,
    ) -> (np.ndarray, np.ndarray):
        """
        ts(...) - Return events in interval between indicated times, ignoring
                  events at `t_stop`.

        :param Optional[float] t_start:     Time from which on events are returned
        :param Optional[float] t_stop:      Time until which events are returned
        :param Optional[Union[int, ArrayLike]] channels:  Channels of which events are returned

        :return:
            np.ndarray  Times of events
            np.ndarray  Channels of events
        """
        # - Get default start and end values from time series data
        if t_start is None:
            t_start: float = self.t_start

        if t_stop is None:
            t_stop: float = self.t_stop

        # - Permit unsorted bounds
        if t_stop < t_start:
            t_start, t_stop = t_stop, t_start

        # - Handle a periodic time series
        if self.periodic:
            # - Repeat events sufficiently often
            all_times = _extend_periodic_times(t_start, t_stop, self)
            num_reps = int(np.round(all_times.size / self.channels.size))
            all_channels = np.tile(self.channels, num_reps)
        else:
            all_times = self.times
            all_channels = self.channels

        # - Events with matching channels
        channel_matches = self._matching_channels(channels, all_channels)

        # - Ignore events from stop time onwards
        choose_events_stop: np.ndarray = all_times < t_stop

        # - Extract matching events and return
        choose_events: np.ndarray = (
            (all_times >= t_start) & (choose_events_stop) & channel_matches
        )
        return all_times[choose_events], all_channels[choose_events]

    def __getitem__(self, ind: Union[ArrayLike, slice, int]) -> "TSEvent":
        """
        ts[tTime1, tTime2, ...] - Index the events of `self` by with the argument
                                  and return TSEvent with corresponding events.
                                  Other attributes, including `num_channels` and `duration`
                                  are the same as in `self`.
        :return:
            np.array of indexed event times
            np.array of indexed event channels
        """
        indexed_times: np.ndarray = np.atleast_1d(self.times[ind])
        indexed_channels: np.ndarray = np.atleast_1d(self.channels[ind])
        # - New TSEvent with the selected events
        new_series = self.copy()
        new_series._times = indexed_times
        new_series._channels = indexed_channels
        return new_series

    def __repr__(self):
        """
        __repr__() - Return a string representation of this object
        :return: str String description
        """
        if self.isempty():
            return "Empty `TSEvent` object `{}` from t={} to t={}.".format(
                self.name, self.t_start, self.t_stop
            )
        else:
            return "{}periodic `TSEvent` object `{}` from t={} to {}. Channels: {}. Events: {}".format(
                int(not self.periodic) * "non-",
                self.name,
                self.t_start,
                self.t_stop,
                self.num_channels,
                self.times.size,
            )

    # - Iteration
    def __iter__(self):
        """
        Yield tuples of event times and channels

        :yield Tuple[float, int]:   Yields a tuple [times, channels]
        """
        for t, ch in zip(self.times, self.channels):
            yield (t, ch)

    ## -- Properties

    @property
    def channels(self):
        """
        (ArrayLike[int]) Event channel indices. A ``Tx1`` vector, where each element ``t`` corresponds to the event time in ``self.times[t]``.
        """
        return self._channels

    @channels.setter
    def channels(self, new_channels: np.ndarray):

        new_channels = np.asarray(new_channels)

        # - Check size of new data
        if np.size(new_channels) != 1 and np.size(new_channels) != np.size(self.times):
            raise ValueError(
                f"TSEvent `{self.name}`: `new_channels` must be the same size as `times`."
            )

        # - Handle scalar channel
        if np.size(new_channels) == 1:
            new_channels = np.repeat(new_channels, np.size(self._times))

        # - Update self.num_channels
        if new_channels.size > 0:
            highest_channel = np.amax(new_channels)
            if self.num_channels <= highest_channel:
                self.num_channels = highest_channel + 1
                # print(
                #     f"TSEvent `{self.name}`: `num_channels` has been increased "
                #     + f"to {self.num_channels}."
                # )
        # - Assign channels
        self._channels = new_channels

    @property
    def num_channels(self):
        """
        (int) The maximum number of channels represented by this :py:class:`TSEvent`
        """
        return self._num_channels

    @num_channels.setter
    def num_channels(self, new_num_ch):
        if self.channels.size > 0:
            min_num_ch = np.amax(self.channels)
        else:
            min_num_ch = 0
        if new_num_ch < min_num_ch:
            raise ValueError(
                f"TSContinuous `{self.name}`: `num_channels` must be at least {min_num_ch}."
            )
        else:
            self._num_channels = new_num_ch

    @property
    def times(self):
        """(ArrayLike[float]) Array of sample times"""
        return self._times

    @times.setter
    def times(self, new_times: ArrayLike):
        # - Check time trace for correct size
        if np.size(new_times) != np.size(self._times):
            raise ValueError(
                f"TSContinuous `{self.name}`: "
                + "New time trace must have the same number of elements as the original trace."
            )

        # - Make sure time trace is sorted
        if (np.diff(new_times) < 0).any():
            raise ValueError(
                f"TSContinuous `{self.name}`: "
                + "The time trace must be sorted and not decreasing"
            )

        # - Store new time trace
        self._times = np.atleast_1d(new_times).flatten().astype(float)

        if np.size(self._times) > 0:
            # - Fix t_start and t_stop
            self._t_start = min(self._t_start, new_times[0])
            res = np.finfo(self._times.dtype).resolution
            self._t_stop = max(self._t_stop, new_times[-1] + res)

    @property
    def t_stop(self) -> float:
        """(float) Stop time of time series (final sample)"""
        return self._t_stop

    @t_stop.setter
    def t_stop(self, new_stop):
        # - Smallest allowed value for new_stop
        res = np.finfo(self._times.dtype).resolution
        min_stop = self._times[-1] + res if self._times.size > 0 else self._t_start
        if new_stop >= min_stop:
            self._t_stop = new_stop
        elif min_stop - new_stop < _TOLERANCE_ABSOLUTE:
            self._t_stop = min_stop
        else:
            raise ValueError(
                "TimeSeries `{}`: t_stop must be greater or equal to {}. It was {}.".format(
                    self.name, min_stop, new_stop
                )
            )


### --- Dict-like object to store TimeSeries on disk
class TSDictOnDisk(collections.abc.MutableMapping):
    """
    Behaves like a dict. However, if a `TimeSeries` is added, it will be stored in a temporary file to reduce main memory usage.
    """

    def __init__(self, data: Union[Dict, "TSDictOnDisk"] = {}):
        """
        TSDictOnDisk - dict-like container that stores TimeSeries in temporary files to save memory.

        :param Union[Dict, TSDictOnDisk] data:   Data with which the object should be instantiatied.
        """

        # - Dict to hold non-`TimeSeries` objects, to emulate behavior of a normal dict
        self._mapping = {}
        # - Dict for handles to temporary files that store `TimeSeries`
        self._mapping_ts = {}
        # - Add provided data to `self`.
        self.update(data)

    def insert(self, data: Union[Dict, "TSDictOnDisk"]):
        """
        insert - Similar to 'self.update'. The difference is that `update` would store any TimeSeries in `data` in a temporary file, whereas `insert` keeps TimeSeries from `data` in the memory if they are not already in a temporary file (e.g. when including a dict).
        :data:  Dict or TSDictOnDisk that is to be inserted.
        """

        # - Make sure existing keys are overwritten
        include_keys = set(data.keys())
        remove_keys = include_keys.intersection(self.keys())
        for k in remove_keys:
            del self[k]

        # - Insert `TSDictOnDisk`
        if isinstance(data, TSDictOnDisk):
            self._mapping.update(data._mapping)
            self._mapping_ts.update(data._mapping_ts)
        # - Insert dict
        else:
            self._mapping.update(data)

    def __getitem__(self, key: Hashable) -> Any:
        """
        dod[key] - Access an object of `self` by its key.
        :return:
            The object to which the `key` corresponds.
        """
        if key in self._mapping_ts:
            # - Load `TimeSeries` from temporary file
            return load_ts_from_file(self._mapping_ts[key])
        else:
            # - Return value stored under `key`.
            return self._mapping[key]

    def __setitem__(self, key: Hashable, value: Any):
        """
        dod[key] = value - Add an object to self together with a (hashable) key.
        """
        if isinstance(value, TimeSeries):
            # - Store `TimeSeries` in a temporary file, whose handle is added as value to `self._mapping_ts`.
            self._mapping_ts[key] = TemporaryFile()
            value.save(self._mapping_ts[key])
            # - Make sure existing keys are overwritten, also in `self._mapping`.
            if key in self._mapping:
                del self._mapping[key]
        else:
            # - Store `value` in `self._mapping`.
            self._mapping[key] = value
            # - Make sure existing keys are overwritten, also in `self._mapping_ts`.
            if key in self._mapping_ts:
                del self._mapping_ts[key]

    def __delitem__(self, key: Hashable):
        """del dod[key] - Delete an object from self by its key."""
        # - Delete the object corresponding to `key` from the correct dict.
        if key in self._mapping_ts:
            del self._mapping_ts[key]
        else:
            del self._mapping[key]

    def __len__(self):
        """len(dod) - Return the total number of objects stored in `self`."""
        return len(self._mapping) + len(self._mapping_ts)

    def __iter__(self):
        """
        for x in dod:... - First iterate over non-`TimeSeries` objects then over
                           `TimeSeries` that are stored in temporary files.
        :yield Any:     The objects stored in `self`.
        """
        for obj in iter(self._mapping):
            yield obj
        for obj in iter(self._mapping_ts):
            yield obj

    def __repr__(self):
        """
        Return a string representation of this object

        :return str: String description
        """
        return (
            f"{type(self).__name__}.\nKeys of stored TimeSeries objects:\n"
            + f"{list(self._mapping_ts)}\nOther keys:\n"
            + f"{list(self._mapping)}"
        )
