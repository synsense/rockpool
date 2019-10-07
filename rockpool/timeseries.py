"""
timeseries.py - Classes to manage time series
"""

import numpy as np
import scipy.interpolate as spint
from warnings import warn
import copy
from typing import Union, List, Tuple, Optional, Iterable
import collections

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
    "set_global_ts_plotting_backend",
    "get_global_ts_plotting_backend",
    "load_ts_from_file",
]

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

### -- Code for setting plotting backend

# - Absolute tolerance, e.g. for comparing float values
_TOLERANCE_ABSOLUTE = 1e-9

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
    Super-class to represent a continuous or event-based time series. You should use the subclasses `.TSContinuous` and `.TSEvent` to represent continuous-time and event-based time series, respectively. See :ref:`/basics/time_series.ipynb` for futher explanation and examples.
    """

    def __init__(
        self,
        times: ArrayLike,
        periodic: bool = False,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        plotting_backend: Optional[str] = None,
        name: str = "unnamed",
    ):
        """
        TimeSeries - Represent a continuous or event-based time series

        :param ArrayLike times:                 [Tx1] vector of time samples
        :param bool periodic:                   Treat the time series as periodic around the end points. Default: ``False``
        :param Optional[float] t_start:         If not ``None``, the series start time is ``t_start``, otherwise ``times[0]``
        :param Optional[float] t_stop:          If not ``None``, the series stop time is ``t_stop``, otherwise ``times[-1]``
        :param Optional[str] plotting_backend:  Determines plotting backend. If ``None``, backend will be chosen automatically based on what is available.
        :param str name:                        Name of the TimeSeries object. Default: "unnamed"
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

    def delay(self, offset: Union[int, float], inplace: bool = False) -> "TimeSeries":
        """
        Return a copy of ``self`` that is delayed by an offset

        For delaying self, use the `inplace` argument, or ``.times += ...`` instead.

        :param float Offset:    Time by which to offset this time series
        :param bool inplace:    If ``True``, conduct operation in-place (Default: ``False``; create a copy)
        :return TimeSeries:     New TimeSeries, delayed
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

    def isempty(self) -> bool:
        """
        Test if this TimeSeries object is empty

        :return bool: ``True`` iff the TimeSeries object contains no samples
        """
        return np.size(self.times) == 0

    def print(self):
        """print() - Print an overview of the time series."""
        print(self.__repr__())

    def set_plotting_backend(self, backend: Union[str, None], verbose: bool = True):
        """
        Set which plotting backend to use with the .plot() method

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

    def copy(self) -> "TimeSeries":
        """
        Return a deep copy of this time series

        :return TimeSeries: copy of `self`
        """
        return copy.deepcopy(self)

    def _modulo_period(
        self, times: Union[ArrayLike, float, int]
    ) -> Union[ArrayLike, float, int]:
        """_modulo_period - Calculate provided times modulo `self.duration`"""
        return self.t_start + np.mod(times - self.t_start, self.duration)

    def __len__(self):
        return self._times.size

    @property
    def times(self):
        """ (ArrayLike[float]) Array of sample times """
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
        """ (float) Start time of time series"""
        return self._t_start

    @t_start.setter
    def t_start(self, new_start):
        try:
            # - Largest allowed value for new_start
            max_start = self._times[0] if self._times.size > 0 else self._t_stop
            if new_start <= max_start:
                self._t_start = float(new_start)
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
        """ (float) Stop time of time series (final sample) """
        return self._t_stop

    @t_stop.setter
    def t_stop(self, new_stop):
        # - Smallest allowed value for new_stop
        min_stop = self._times[-1] if self._times.size > 0 else self._t_start
        if new_stop >= min_stop:
            self._t_stop = new_stop
        else:
            raise ValueError(
                "TimeSeries `{}`: t_stop must be greater or equal to {}. It was {}.".format(
                    self.name, min_stop, new_stop
                )
            )

    @property
    def duration(self) -> float:
        """ (float) Duration of TimeSeries """
        return self._t_stop - self._t_start

    @property
    def plotting_backend(self):
        """ (str) Current plotting backend"""
        return (
            self._plotting_backend
            if self._plotting_backend is not None
            else _global_plotting_backend
        )


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

    def __init__(
        self,
        times: Optional[ArrayLike] = None,
        samples: Optional[ArrayLike] = None,
        num_channels: Optional[int] = None,
        periodic: bool = False,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        name: str = "unnamed",
        interp_kind: str = "linear",
    ):
        """
        TSContinuous - Represents a continuously-sample time series, supporting interpolation and periodicity.

        :param ArrayLike times:             [Tx1] vector of time samples
        :param ArrayLike samples:           [TxM] matrix of values corresponding to each time sample
        :param Optional[in] num_channels:   If `samples` is None, determines the number of channels of ``self``. Otherwise it has no effect at all.
        :param bool periodic:               Treat the time series as periodic around the end points. Default: False
        :param float t_start:               If not None, the series start time is t_start, otherwise times[0]
        :param float t_stop:                If not None, the series stop time is t_stop, otherwise times[-1]
        :param str name:                    Name of the `.TSContinuous` object. Default: "unnamed"
        :param str interp_kind:             Specify the interpolation type. Default: "linear"

        If the time series is not periodic (the default), then NaNs will be returned for any extrapolated values.
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
        self.interp_kind = interp_kind
        self.samples = samples.astype("float")

    ## -- Methods for plotting and printing

    def plot(
        self,
        times: Union[int, float, ArrayLike] = None,
        target: Union["mpl.axes.Axes", "hv.Curve", "hv.Overlay", None] = None,
        channels: Union[ArrayLike, int, None] = None,
        *args,
        **kwargs,
    ):
        """
        Visualise a time series on a line plot

        :param Optional[ArrayLike] times: Time base on which to plot. Default: time base of time series
        :param Optional target:  Axes (or other) object to which plot will be added.
        :param Optional[ArrayLike] channels:  Channels of the time series to be plotted.
        :param args, kwargs:  Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """
        if times is None:
            times = self.times
            samples = self.samples
        else:
            samples = self(times)
        if channels is not None:
            samples = samples[:, channels]

        if target is None:
            # - Determine plotting backend
            if self._plotting_backend is None:
                backend = _global_plotting_backend
            else:
                backend = self._plotting_backend
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
                return plt.plot(times, samples, **kwargs)
            else:
                raise RuntimeError(
                    f"TSContinuous: `{self.name}`: No plotting back-end set."
                )

        else:
            # - Infer current plotting backend from type of `target`
            if isinstance(target, (hv.Curve, hv.Overlay)):
                if _HV_AVAILABLE:
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
                else:
                    raise RuntimeError(
                        f"TSContinuous `{self.name}`: Holoviews is not available."
                    )

            elif isinstance(target, mpl.axes.Axes):
                if _MPL_AVAILABLE:
                    # - Add `self.name` as label only if a label is not already present
                    kwargs["label"] = kwargs.get("label", self.name)
                    target.plot(times, samples, **kwargs)
                    return target
                else:
                    raise RuntimeError(
                        f"TSContinuous `{self.name}`: Holoviews is not available."
                    )
            else:
                raise TypeError(
                    f"TSContinuous: `{self.name}`: Unrecognized type for `target`. "
                    + "It must be matplotlib Axes or holoviews Curve or Overlay."
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

    def save(self, path: str, verbose: bool = False):
        """
        Save this time series as an ``npz`` file using np.savez

        :param str path:    Path to save file
        """

        # - Make sure path is a string (and not a Path object)
        path = str(path)

        # - Some modules add a `trial_start_times` attribute to the object.
        trial_start_times = (
            self.trial_start_times if hasattr(self, "trial_start_times") else None
        )

        # - Write the file
        np.savez(
            path,
            times=self.times,
            samples=self.samples,
            t_start=self.t_start,
            t_stop=self.t_stop,
            interp_kind=self.interp_kind,
            periodic=self.periodic,
            name=self.name,
            str_type="TSContinuous",  # Indicate that this object is TSContinuous
            trial_start_times=trial_start_times,
        )
        missing_ending = path.split(".")[-1] != "npz"  # np.savez will add ending
        if verbose:
            print(
                "TSContinuous `{}` has been stored in `{}`.".format(
                    self.name, path + missing_ending * ".npz"
                )
            )

    ## -- Methods for finding and extracting data

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

    ## -- Methods for combining time series

    def merge(
        self,
        other_series: "TSContinuous",
        remove_duplicates: bool = True,
        inplace: bool = False,
    ) -> "TSContinuous":
        """
        Merge another time series to this one, by interleaving in time. Maintain each time series' time values and channel IDs.

        :param TSContinuous other_series:           time series that is merged to self
        :param Optional[bool] remove_duplicates:    If ``True``, time points in ``other_series.times`` that are also in ``self.times`` are discarded. Otherwise they are included in the new time trace and come after the corresponding points of self.times.
        :param Optional[bool] inplace:              Conduct operation in-place (Default: ``False``; create a copy)

        :return TSContinuous:                       The merged time series
        """

        # - Check other_series
        if not isinstance(other_series, TSContinuous):
            raise TypeError(
                f"TSContinuous `{self.name}`: "
                + "`other_series` must be a TSContinuous object."
            )

        if self.num_channels == 0 and len(self) == 0:
            # - Handle empty `self`
            self._samples = np.zeros((0, other_series.num_channels))

        if other_series.num_channels != self.num_channels:
            raise ValueError(
                f"TSContinuous `{self.name}`: `other_series` must include "
                f"the same number of traces ({self.num_channels})."
            )

        # - Create a new time series, or modify this time series
        if not inplace:
            merged_series = self.copy()
        else:
            merged_series = self

        # - If the other TimeSeries is empty, just return
        if other_series.isempty():
            return merged_series

        # - If remove_duplicates == True and time ranges overlap,  find and remove
        #   time points of other_series that are also included in self (assuming both
        #   TimeSeries have a sorted vTimeTrace)
        if remove_duplicates and not (
            self.t_start > other_series.t_stop or self.t_stop < other_series.t_start
        ):
            # Determine region of overlap
            overlap: np.ndarray = np.where(
                (self.times >= other_series.t_start)
                & (self.times <= other_series.t_stop)
            )[0]
            # Array of bools indicating which sampled time points of other_series do not occur in self
            is_unique = np.array(
                [(t != self.times[overlap]).all() for t in other_series.times]
            )
            # Time trace and samples to be merged into self
            times_other: np.ndarray = other_series.times[is_unique]
            samples_other: np.ndarray = other_series.samples[is_unique]
        else:
            times_other: np.ndarray = other_series.times
            samples_other: np.ndarray = other_series.samples

        # - Merge time traces and samples
        times_new: np.ndarray = np.concatenate((self._times, times_other))
        samples_new: np.ndarray = np.concatenate((self.samples, samples_other), axis=0)

        #  - Indices for sorting new time trace and samples. Use mergesort as stable sorting algorithm.
        viSorted: np.ndarray = np.argsort(times_new, kind="mergesort")

        # - Update data of new time series
        merged_series._times = times_new[viSorted]
        merged_series._samples = samples_new[viSorted]
        merged_series._t_start: float = min(self.t_start, other_series.t_start)
        merged_series._t_stop: float = max(self.t_stop, other_series.t_stop)

        # - Create new interpolator
        merged_series._create_interpolator()

        # - Return merged TS
        return merged_series

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
        other_series: "TSContinuous",
        offset: Optional[float] = None,
        inplace: bool = False,
    ) -> "TSContinuous":
        """
        Append another time series to this one, along the time axis

        :param TSContinuous other_series:   Another time series. Will be tacked on to the end of the called series object. ``other_series`` must have the same number of channels
        :param Optional[float] offset:      If not None, defines distance between last sample of ``self`` and first sample of ``other_series``. Otherwise the offset will be the median of all timestep sizes of ``self.samples``.
        :param bool inplace:                Conduct operation in-place (Default: ``False``; create a copy)

        :return TSContinuous:               Time series containing data from ``self``, with the other series appended in time
        """

        if offset is None:
            # - If ``self`` is empty then append new elements directly. Otherwise leave an offset
            #   corresponding to the median distance between time points in `self._times`.
            offset = np.median(np.diff(self._times)) if self.times.size > 0 else 0

        # - Time by which ``other_series`` has to be delayed
        delay = self.t_stop + offset - other_series.t_start

        # - Let ``self.merge()`` do the rest
        return self.merge(
            other_series.delay(delay), remove_duplicates=False, inplace=inplace
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
                samples[times == self.times[0]] = self.samples[0]
                return samples

            self.interp = single_sample

        else:
            # - Construct interpolator
            self.interp = spint.interp1d(
                self._times,
                self._samples,
                kind=self.interp_kind,
                axis=0,
                assume_sorted=True,
                bounds_error=False,
            )

    def _interpolate(self, times: Union[int, float, ArrayLike]) -> np.ndarray:
        """
        Interpolate the time series to the provided time points

        :param ArrayLike times: Array of ``T`` desired interpolated time points

        :return np.ndarray:     Array of interpolated values. Will have the shape ``TxN``, where ``N`` is the number of channels in ``self``
        """
        # - Enforce periodicity
        if self.periodic and self.duration > 0:
            times = (np.asarray(times) - self._t_start) % self.duration + self._t_start

        samples = self.interp(times)

        # - Handle empty series
        if samples is None:
            return np.zeros((np.size(times), 0))
        else:
            return np.reshape(self.interp(times), (-1, self.num_channels))

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

    def __repr__(self):
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

    def __add__(self, other_samples):
        return self.copy().__iadd__(other_samples)

    def __radd__(self, other_samples):
        return self + other_samples

    def __iadd__(self, other_samples):
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

    def __sub__(self, other_samples):
        return self.copy().__isub__(other_samples)

    def __rsub__(self, other_samples):
        return -(self - other_samples)

    def __isub__(self, other_samples):
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

    def __mul__(self, other_samples):
        return self.copy().__imul__(other_samples)

    def __rmul__(self, other_samples):
        return self * other_samples

    def __imul__(self, other_samples):
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

    def __truediv__(self, other_samples):
        return self.copy().__itruediv__(other_samples)

    def __rtruediv__(self, other_samples):
        self_copy = self.copy()
        self_copy.samples = 1 / self_copy.samples
        return self_copy * other_samples

    def __itruediv__(self, other_samples):
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

    def __floordiv__(self, other_samples):
        return self.copy().__ifloordiv__(other_samples)

    def __rfloordiv__(self, other_samples):
        self_copy = self.copy()
        self_copy.samples = 1 / self_copy.samples
        return self_copy // (1 / other_samples)

    def __ifloordiv__(self, other_samples):
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

    # - Exponentiation

    def __pow__(self, exponent):
        return self.copy().__ipow__(exponent)

    def __rpow__(self, base):
        new_series = self.copy()

        base = new_series._compatible_shape(base)

        # - Propagate NaNs
        is_nan_self = np.isnan(new_series.samples)
        is_nan_other = np.isnan(base)

        # - Perform exponentiation
        new_series.samples = base ** new_series.samples

        # - Fill in nans
        new_series.samples[np.logical_or(is_nan_self, is_nan_other)] = np.nan

        return new_series

    def __ipow__(self, exponent):
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

    def __abs__(self):
        self_copy = self.copy()
        self_copy.samples = np.abs(self_copy.samples)
        return self_copy

    # - Negative

    def __neg__(self):
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
        """ (ArrayLike[float]) Array of sample times """
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
        times: ArrayLike = None,
        channels: Union[int, ArrayLike] = None,
        periodic: bool = False,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        name: str = None,
        num_channels: int = None,
    ):
        """
        Represent discrete events in time

        :param ArrayLike[float] times:     ``Tx1`` vector of event times
        :param ArrayLike[int] channels:     ``Tx1`` vector of event channels (Default: all events are in channel 0)

        :param bool periodic:               Is this a periodic TimeSeries (Default: False; non-periodic)

        :param float t_start:               Explicitly specify the start time of this series. If ``None``, then ``times[0]`` is taken to be the start time
        :param float t_stop:                Explicitly specify the stop time of this series. If ``None``, then ``times[-1]`` is taken to be the stop time

        :param str name:                    Name of the time series (Default: None)

        :param int num_channels:            Total number of channels in the data source. If ``None``, max(channels) is taken to be the total channel number
        """

        # - Default time trace: empty
        if times is None:
            times = np.array([])
        else:
            times = np.atleast_1d(times).flatten().astype(float)

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
                    t_start = self.t_stop if time_limits[1] is None else time_limits[1]
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
                return plt.scatter(times, channels, *args, **kwargs)

            else:
                raise RuntimeError(f"TSEvent: `{self.name}`: No plotting back-end set.")

        else:
            # - Infer current plotting backend from type of `target`
            if isinstance(target, (hv.Curve, hv.Overlay)):
                if _HV_AVAILABLE:
                    target *= (
                        hv.Scatter((times, channels), *args, **kwargs)
                        .redim(x="Time", y="Channel")
                        .relabel(self.name)
                    )
                    return target.relabel(group=self.name)
                else:
                    raise RuntimeError(
                        f"TSEvent: `{self.name}`: Holoviews not available."
                    )
            elif isinstance(target, mpl.axes.Axes):
                if _MPL_AVAILABLE:
                    # - Add `self.name` as label only if a label is not already present
                    kwargs["label"] = kwargs.get("label", self.name)
                    target.scatter(times, channels, *args, **kwargs)
                    return target
                else:
                    raise RuntimeError(
                        f"TSEvent: `{self.name}`: Matplotlib not available."
                    )
            else:
                raise TypeError(
                    f"TSEvent: `{self.name}`: Unrecognized type for `target`. "
                    + "It must be matplotlib Axes or holoviews Curve or Overlay."
                )

    ## -- Methods for manipulating timeseries

    def clip(
        self,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        channels: Union[int, ArrayLike, None] = None,
        include_stop: bool = False,
        remap_channels: bool = False,
        inplace: bool = False,
    ) -> "TSEvent":
        """
        Return a `TSEvent` which is restricted to given time limits and only contains events of selected channels

        If time limits are provided, `.t_start` and `.t_stop` attributes of the new time series will correspond to those. If `remap_channels` is ``True``, channels IDs will be mapped to a continuous sequence of integers starting from 0 (e.g. [1, 3, 6]->[0, 1, 2]). In this case `.num_channels` will be set to the number of different channels in ``channels``. Otherwise `.num_channels` will keep its original values, which is also the case for all other attributes. If `inplace` is True, modify ``self`` accordingly.

        :param Optional[float] t_start:             Time from which on events are returned. Default: `.t_start`
        :param Optional[float] t_stop:              Time until which events are returned. Default: `.t_stop`
        :param Optional[ArrayLike[int]] channels:   Channels of which events are returned. Default: All channels
        :param Optional[bool] include_stop:          If there are events with time `t_stop`, include them or not. Default: ``False``, do not include events at `t_stop`
        :param Optional[bool] remap_channels:        Map channel IDs to continuous sequence starting from 0. Set `num_channels` to largest new ID + 1. Default: ``False``, do not remap channels
        :param Optional[bool] inplace:              Iff ``True``, the operation is performed in place (Default: False)

        :return TSEvent:                            `TSEvent` containing events from the requested channels
        """

        if not inplace:
            new_series = self.copy()
        else:
            new_series = self

        # - Extract matching events
        time_data, channel_data = new_series(t_start, t_stop, channels, include_stop)

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
        Renumber channels in the :py:class:`TSEvent`

        Maps channels 0..``self.num_channels-1`` to the channels in ``channel_map``.

        :param ArrayLike[int] channel_map:  List of channels that existing channels should be mapped to, in order.. Must be of size ``self.num_channels``.
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

        :param float dt:                            Duration of single time step in raster
        :param Optional[float] t_start:             Time where to start raster. Default: None (use ``self.t_start``)
        :param Optional[float] t_stop:              Time where to stop raster. This time point is not included in the raster. Default: ``None`` (use ``self.t_stop``. If ``num_timesteps`` is provided, ``t_stop`` is ignored.
        :param Optional[int] num_timesteps:         Specify number of time steps directly, instead of providing ``t_stop``. Default: ``None`` (use ``t_start``, ``t_stop`` and ``dt`` to determine raster size)
        :param Optional[ArrayLike[int]] channels:   Channels from which data is to be used. Default: ``None`` (use all channels)
        :param Optional[bool] add_events:           If ``True``, return an integer raster containing number of events for each time step and channel. Default: ``False``, merge simultaneous events in a single channel, and return a boolean raster

        :return ArrayLike:  event_raster - Boolean matrix with ``True`` indicating presence of events for each time step and channel. If ``add_events == True``, the raster consists of integers indicating the number of events per time step and channel. First axis corresponds to time, second axis to channel.
        """
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

        # - Work out number of time steps
        if num_timesteps is None:
            series = self.clip(
                t_start=t_start,
                t_stop=t_stop,
                channels=channels_clip,
                remap_channels=False,
            )
            # - Make sure that last point is also included if ``duration`` is a
            #   multiple of dt. Therefore floor(...) + 1
            num_timesteps = int(np.floor((series.duration) / dt)) + 1

        else:
            t_stop = t_start + num_timesteps * dt
            series = self.clip(
                t_start=t_start,
                t_stop=t_stop,
                channels=channels_clip,
                remap_channels=False,
            )

        # - Raster for storing event data
        raster_type = int if add_events else bool
        event_raster = np.zeros((num_timesteps, channels.size), raster_type)

        # - Handle empty series
        if len(series) == 0:
            return event_raster

        # - Select data according to time base
        event_times = series.times
        event_channels = series.channels

        ## -- Convert input events and samples to boolean or integer raster
        # - Only consider rasters that have non-zero length
        if num_timesteps > 0:
            # - Compute indices for times
            time_indices = np.floor((event_times - t_start) / dt).astype(int)
            time_indices = time_indices[time_indices < num_timesteps]

            if add_events:
                # Count events per time step and channel
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
                        + " per time step. Consider smaller dt or setting add_events True."
                    )
                # Mark spiking indices with True
                event_raster[time_indices, event_channels] = True

        return event_raster

    def xraster(
        self,
        dt: float,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        channels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generator which ``yield`` s a rasterized time series data, where each data point represents a time step

        Events are represented in a boolean matrix, where the first axis corresponds to time, the second axis to the channel. Events that happen between time steps are projected to the preceding one. If two events happen during one time step within a single channel, they are counted as one.

        :param float dt:                            Duration of single time step in raster
        :param Optional[float] t_start:             Time where to start raster. Default: ``None`` (use ``self.t_start``)
        :param Optional[float] t_stop:              Time where to stop raster. This time point is not included in the raster. Default: ``None`` (use ``self.t_stop``. If ``num_timesteps`` is provided, ``t_stop`` is ignored.
        :param Optional[int] num_timesteps:         Specify number of time steps directly, instead of providing ``t_stop``. Default: ``None`` (use ``t_start``, ``t_stop`` and ``dt`` to determine raster size.
        :param Optional[ArrayLike[int]] channels:   Channels from which data is to be used. Default: ``None`` (use all channels)

        :yields ArrayLike: event_raster - Boolean matrix with ``True`` indicating presence of events for each time step and channel. If ``add_events == True``, the raster consists of integers indicating the number of events per time step and channel. First axis corresponds to time, second axis to channel.
        """
        event_raster = self.raster(
            dt=dt,
            t_start=t_start,
            t_stop=t_stop,
            num_timesteps=num_timesteps,
            channels=channels,
        )
        yield from event_raster  # Yield one row at a time

    def save(self, path: str, verbose: bool = False):
        """
        Save this :py:`TSEvent` as an ``npz`` file using ``np.savez``

        :param str path: File path to save data
        """

        # - Make sure path is string (and not Path object)
        path = str(path)

        # - Some modules add a `trial_start_times` attribute to the object.
        trial_start_times = (
            self.trial_start_times if hasattr(self, "trial_start_times") else None
        )
        np.savez(
            path,
            times=self.times,
            channels=self.channels,
            t_start=self.t_start,
            t_stop=self.t_stop,
            periodic=self.periodic,
            num_channels=self.num_channels,
            name=self.name,
            str_type="TSEvent",  # Indicate that the object is TSEvent
            trial_start_times=trial_start_times,
        )
        missing_ending = path.split(".")[-1] != "npz"  # np.savez will add ending
        if verbose:
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
        :param Optional[bool] inplace:  Conduct operation in-place (Default: ``False``; create a copy)

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
        offset: float = 0,
        remove_duplicates: bool = False,
        inplace: bool = False,
    ) -> "TSEvent":
        """
        Append another time series to this one along the time axis

        ``t_start`` from ``other_series`` is shifted to ``self.t_stop + offset``.

        :param TSEvent other_series:                :py:class:`TSEvent` or list of :py:class:`TSEvent` that will be appended to ``self`` along the time axis
        :param Optional[float] offset:              Scalar or iterable with at least the same number of elements as ``other_series``. If scalar, use same value for all timeseries. Event times from ``other_series`` will be shifted by ``self.t_stop + offset``. Default: 0
        :param Optional[bool] remove_duplicates:    If ``True``, duplicate events will be removed from the resulting timeseries. Duplicates can occur if ``offset`` is negative. Default: ``False``, do not remove duplicate events.
        :param Optional[bool] inplace:              If ``True``, conduct operation in-place (Default: ``False``; return a copy)

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
            offset_list = [offset] * len(other_series)
        else:
            offset_list = list(offset)
            if len(offset_list) != len(other_series):
                warn(
                    f"TSEvent `{self.name}`: Numbers of provided offsets and TSEvent "
                    + "objects do not match. Will ignore excess elements."
                )

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
        :param Optional[float] delay:   Scalar or iterable with at least the number of elements as other_series. If scalar, use same value for all timeseries. Delay ``other_series`` series by this value before merging.
        :param Optional[bool] remove_duplicates:  If ``True``, remove duplicate events in resulting timeseries. Default: ``False``, do not remove duplicates.
        :param Optional[bool] inplace:  If ``True``, operation will be performed in place (Default: ``False``, return a copy)

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
        self, channels: Union[int, ArrayLike, None] = None
    ) -> np.ndarray:
        """
        Return boolean array of which events match a given channel selection

        :param ArrayLike[int] channels: Channels of which events are to be indicated ``True``. Default: ``None``, use all channels

        :return ArrayLike[bool]:        A matrix ``TxC`` indicating which events match the requested channels
        """

        if channels is None:
            channels = np.arange(self.num_channels)
        else:
            # - Check `channels` for validity
            if not (np.min(channels) >= 0 and np.max(channels) < self.num_channels):
                raise IndexError(
                    f"TSEvent `{self.name}`: `channels` must be between 0 and {self.num_channels}."
                )
        # - Make sure elements in `channels` are unique for better performance
        channels = np.unique(channels)

        # - Boolean array of which events match selected channels
        include_events = np.isin(self._channels, channels)

        return include_events

    ## -- Magic methods

    def __call__(
        self,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        channels: Union[int, ArrayLike, None] = None,
        include_stop: bool = False,
    ) -> (np.ndarray, np.ndarray):
        """
        ts(...) - Return events in interval between indicated times

        :param t_start:     Time from which on events are returned
        :param t_stop:      Time until which events are returned
        :param channels:  Channels of which events are returned
        :param include_stop:  If there are events with time t_stop include them or not
        :return:
            np.ndarray  Times of events
            np.ndarray  Channels of events
        """
        if t_start is None:
            t_start: float = self.t_start
        if t_stop is None:
            t_stop: float = self.t_stop
            include_stop = True
        # - Permit unsorted bounds
        if t_stop < t_start:
            t_start, t_stop = t_stop, t_start
        # - Events with matching channels
        channel_matches = self._matching_channels(channels)

        if self.periodic:
            # - Repeat events sufficiently often
            all_times = _extend_periodic_times(t_start, t_stop, self)
            num_reps = int(np.round(all_times.size / self.channels.size))
            all_channels = np.tile(self.channels, num_reps)
        else:
            all_times = self.times
            all_channels = self.channels

        if include_stop:
            choose_events_stop: np.ndarray = all_times <= t_stop
        else:
            choose_events_stop: np.ndarray = all_times < t_stop

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
        # - Check size of new data
        assert np.size(new_channels) == 1 or np.size(new_channels) == np.size(
            self.times
        ), "`new_channels` must be the same size as `times`."

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


def load_ts_from_file(path: str, expected_type: Optional[str] = None) -> TimeSeries:
    """
    Load a timeseries object from an ``npz`` file

    :param str path:                    Filepath to load file
    :param Optional[str] expected_type: Specify expected type of timeseires (:py:class:`TSContinuous` or py:class:`TSEvent`). Default: ``None``, use whichever type is loaded.

    :return TimeSeries: Loaded time series object
    :raises TypeError:  Unsupported or unexpected type
    """
    # - Make sure path is string (and not Path object)
    path = str(path)

    # - Load npz file from specified path
    dLoaded = np.load(path)

    # - Check for expected type
    try:
        loaded_type = dLoaded["str_type"].item()
    except KeyError:
        loaded_type = dLoaded["strType"].item()

    if expected_type is not None:
        if not loaded_type == expected_type:
            raise TypeError(
                "Timeseries at `{}` is of type `{}`, which does not match expected type `{}`.".format(
                    path, loaded_type, expected_type
                )
            )
    if loaded_type == "TSContinuous":
        return TSContinuous(
            times=dLoaded["times"],
            samples=dLoaded["samples"],
            t_start=dLoaded["t_start"].item(),
            t_stop=dLoaded["t_stop"].item(),
            interp_kind=dLoaded["interp_kind"].item(),
            periodic=dLoaded["periodic"].item(),
            name=dLoaded["name"].item(),
        )
    elif loaded_type == "TSEvent":
        return TSEvent(
            times=dLoaded["times"],
            channels=dLoaded["channels"],
            t_start=dLoaded["t_start"].item(),
            t_stop=dLoaded["t_stop"].item(),
            periodic=dLoaded["periodic"].item(),
            num_channels=dLoaded["num_channels"].item(),
            name=dLoaded["name"].item(),
        )
    else:
        raise TypeError("Type `{}` not supported.".format(loaded_type))
