###
# timeseries.py - Classes to manage time series
###

import numpy as np
import scipy.interpolate as spint
from warnings import warn
import copy
from typing import Union, List, Tuple, Optional, TypeVar, Iterable
import collections
import matplotlib as mpl
from matplotlib import pyplot as plt
import holoviews as hv

# - Define exports
__all__ = [
    "TimeSeries",
    "set_plotting_backend",
    "get_plotting_backend",
    "TSEvent",
    "TSContinuous",
]

# - Type alias for array-like objects and for yet undefined objects
ArrayLike = Union[np.ndarray, List, Tuple]
TSEventType = TypeVar("TSEvent")
TSType = TypeVar("TimeSeries")

### -- Code for setting plotting backend

__bUseHoloviews = False
__bUseMatplotlib = False


# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9


def set_plotting_backend(backend: str):
    global __bUseHoloviews
    global __bUseMatplotlib
    if backend in ("holoviews", "holo", "Holoviews", "HoloViews", "hv"):
        __bUseHoloviews = True
        __bUseMatplotlib = False

    elif backend in ("matplotlib", "mpl", "mp", "pyplot", "plt"):
        __bUseHoloviews = False
        __bUseMatplotlib = True

    else:
        __bUseHoloviews = False
        __bUseMatplotlib = False


def get_plotting_backend():
    return __bUseHoloviews, __bUseMatplotlib


## - Set default plotting backend
set_plotting_backend("matplotlib")


## - Convenience method to return a nan array
def full_nan(vnShape: Union[tuple, int]):
    a = np.empty(vnShape)
    a.fill(np.nan)
    return a


## -- Convenience function to decorate methods to permit modifying owners in place
def optional_inplace(func):
    def inner_func(obj, *args, **kwargs):
        inplace = kwargs.pop("inplace", False)
        if inplace:
            return func(obj, *args, **kwargs)
        else:
            return func(obj.copy(), *args, **kwargs)

    return inner_func


### --- TimeSeries base class


class TimeSeries:
    """
    TimeSeries - Class represent a continuous or event-based time series
    """

    def __init__(
        self,
        vtTimeTrace: ArrayLike,
        bPeriodic: bool = False,
        tStart: Optional[float] = None,
        tStop: Optional[float] = None,
        strName: str = "unnamed",
    ):
        """
        TimeSeries - Class represent a continuous or event-based time series

        :param vtTimeTrace:     [Tx1] vector of time samples
        :param bPeriodic:       bool: Treat the time series as periodic around the end points. Default: False
        :param tStart:          float: If not None, the series start time is tStart, otherwise vtTimeTrace[0]
        :param tStop:           float: If not None, the series stop time is tStop, otherwise vtTimeTrace[-1]
        :param strName:         str: Name of the TimeSeries object. Default: `unnamed`
        """

        # - Convert time trace to numpy arrays
        vtTimeTrace = np.asarray(vtTimeTrace).flatten().astype(float)

        if (np.diff(vtTimeTrace) < 0).any():
            raise ValueError(
                f"TimeSeries `{strName}`: The time trace must be not decreasing"
            )

        # - Assign attributes
        self._vtTimeTrace = vtTimeTrace
        self.bPeriodic = bPeriodic
        self.strName = strName
        self.tStart = (
            (0 if np.size(vtTimeTrace) == 0 else vtTimeTrace[0])
            if tStart is None
            else tStart
        )
        self.tStop = (
            (0 if np.size(vtTimeTrace) == 0 else vtTimeTrace[-1])
            if tStop is None
            else tStop
        )

    def delay(
        self, offset: Union[int, float], inplace: bool = False
    ) -> Union[None, TSType]:
        """
        delay - Return a copy of self that is delayed by an offset.
                For delaying self, use ".vtTimeTrace += ..." instead.

        :param tOffset:     float   Time offset
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: New TimeSeries, delayed
        """
        if not inplace:
            series = self.copy()
        else:
            series = self

        # - Store previous tStart and tStop
        t_start_old = series.tStart
        t_stop_old = series.tStop

        # - Shift time trace
        if not self.isempty():
            series.vtTimeTrace += offset
        # - Shift tStart and tStop
        series._tStart = t_start_old + offset
        series._tStop = t_stop_old + offset

        return series

    def resample(self, vtTimes: Union[int, float, ArrayLike], inplace: bool = False):
        """
        resample - Return a new time series sampled to the supplied time base

        :param vtTimes:     Array-like of T desired time points to resample
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:            TimeSeries object, resampled to new time base
        """
        if not inplace:
            tsResampled = self.copy()
        else:
            tsResampled = self

        # - Resample time series
        tsResampled._mfSamples = self(vtTimes)
        tsResampled._vtTimeTrace = vtTimes
        tsResampled._tStart = vtTimes[0]
        tsResampled._tStop = vtTimes[-1]
        tsResampled.bPeriodic = False
        tsResampled._create_interpolator()
        return tsResampled

    def resample_within(
        self,
        tStart: float = None,
        tStop: float = None,
        tDt: float = None,
        inplace: bool = False,
    ):
        """
        resample_within - Return a new time series sampled between tStart
                          and tStop with step size tDt

        :param tStart:  Start time for sampling - defaults to minimum value
                        of self.vtTimeTrace
        :param tStop:   Stop time for sampling - defaults to maximum value
                        of self.vtTimeTrace
        :param tDt:     Sampling time step - defaults to mean difference
                        between values of self.vtTimeTrace
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:        New TimeSeries object, resampled according to parameters
        """
        # - Determine start time, if not supplied
        tStart = (
            self.tStart
            if tStart is None
            else (
                tStart
                if self.bPeriodic  # - Allow for tStart < self.tStart if self.bPeriodic
                else max(tStart, self.tStart)
            )
        )

        # - Determine stop time, if not supplied
        tStop = (
            self.tStop
            if tStop is None
            else (
                tStop
                if self.bPeriodic  # - Allow for tStop > self.tStop if self.bPeriodic
                else min(tStop, self.tStop)
            )
        )

        # - Determine time step, if not supplied
        tDt = np.mean(np.diff(self.vtTimeTrace)) if tDt is None else tDt

        # - Build a time trace for the resampled time series
        vtSampleTimes = np.arange(tStart, tStop + tDt, tDt)
        vtSampleTimes = vtSampleTimes[vtSampleTimes <= tStop + fTolAbs]

        # - If vtSampleTimes[-1] is close to tStop, correct it, so that
        #   is exactly tStop. This ensures that the returned TimeSeries
        #   is neither too short, nor is the last sample nan
        if np.isclose(vtSampleTimes[-1], tStop, atol=fTolAbs):
            vtSampleTimes[-1] = tStop

        # - Return a resampled time series
        return self.resample(vtSampleTimes, inplace=inplace)

    def merge(self, tsOther, remove_duplicates=True, inplace: bool = False):
        """
        merge - Merge another time series to this one, in time. Maintain
                each time series' time values.
        :param tsOther:             TimeSeries that is merged to self
        :param remove_duplicates:   bool - If true, time points in tsOther.vtTimeTrace
                                           that are also in self.vtTimeTrace are
                                           discarded. Otherwise they are included in
                                           the new time trace and come after the
                                           corresponding points of self.vtTimeTrace.
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:         The merged time series
        """

        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), "`tsOther` must be a TimeSeries object."

        assert tsOther.nNumChannels == self.nNumChannels, (
            "`tsOther` must include the same number of traces ("
            + str(self.nNumChannels)
            + ")."
        )

        # - If the other TimeSeries is empty, just return
        if tsOther.isempty():
            if inplace:
                return self
            else:
                return self.copy()

        # - If remove_duplicates==True and time ranges overlap,  find and remove
        #   time points of tsOther that are also included in self (assuming both
        #   TimeSeries have a sorted vTimeTrace)
        if remove_duplicates and not (
            self.tStart > tsOther.tStop or self.tStop < tsOther.tStart
        ):
            # Determine region of overlap
            viOverlap = np.where(
                (self.vtTimeTrace >= tsOther.tStart)
                & (self.vtTimeTrace <= tsOther.tStop)
            )
            # Array of bools indicating which sampled time points of tsOther do not occur in self
            vbUnique = np.array(
                [(t != self.vtTimeTrace[viOverlap]).all() for t in tsOther.vtTimeTrace]
            )
            # Time trace and samples to be merged into self
            vtTimeTraceOther = tsOther.vtTimeTrace[vbUnique]
            mfSamplesOther = tsOther.mfSamples[vbUnique]
        else:
            vtTimeTraceOther = tsOther.vtTimeTrace
            mfSamplesOther = tsOther.mfSamples

        # - Merge time traces and samples
        vtTimeTraceNew = np.concatenate((self._vtTimeTrace, vtTimeTraceOther))
        mfSamplesNew = np.concatenate((self.mfSamples, mfSamplesOther), axis=0)

        #  - Indices for sorting new time trace and samples. Use mergesort as stable sorting algorithm.
        viSorted = np.argsort(vtTimeTraceNew, kind="mergesort")

        # - Take care of tStart and tStop
        tStart = min(self.tStart, tsOther.tStart)
        tStop = min(self.tStop, tsOther.tStop)

        # - Create a new time series, or modify this time series
        if not inplace:
            merged_series = self.copy()
        else:
            merged_series = self

        merged_series._vtTimeTrace = vtTimeTraceNew[viSorted]
        merged_series._mfSamples = mfSamplesNew[viSorted]
        merged_series._tStart = tStart
        merged_series._tStop = tStop

        # - Determine number of channels
        merged_series._nNumChannels = max(tsOther.nNumChannels, self.nNumChannels)

        # - Create new interpolator
        merged_series._create_interpolator()

        # - Return merged TS
        return merged_series

    def append(self, tsOther, inplace: bool = False):
        """
        append() - Combine another time series into this one, along samples axis

        :param tsOther: Another time series. Will be resampled to the time base of the called series object
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:        Current time series,
        """
        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), "`tsOther` must be a TimeSeries object."

        # - Resample tsOther to own time base
        mfOtherSamples = tsOther(self.vtTimeTrace)

        # - Create a new time series, or modify this time series
        if not inplace:
            tsAppended = self.copy()
        else:
            tsAppended = self

        # - Combine samples
        tsAppended.mfSamples = np.concatenate(
            (np.atleast_2d(tsAppended.mfSamples), mfOtherSamples), 1
        )

        # - Create new interpolator
        tsAppended._create_interpolator()

        # - Return appended TS
        return tsAppended

    def concatenate(self, tsOther, inplace: bool = False):
        """
        concatenate() - Combine another time series with this one, along samples axis

        :param tsOther: Another time series. Will be resampled to the time base of the called series object
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: Time series, with series from both source and other time series
        """
        # - Create a new time series, or modify this time series
        return self.append(tsOther, inplace=inplace)

    def append_t(self, tsOther, inplace: bool = False):
        """
        append_t() - Append another time series to this one, in time

        :param tsOther: Another time series. Will be tacked on to the end of the called series object
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: Time series containing current data, with other TS appended in time
        """

        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), "`tsOther` must be a TimeSeries object."

        assert tsOther.nNumChannels == self.nNumChannels, (
            "`tsOther` must include the same number of traces ("
            + str(self.nNumChannels)
            + ")."
        )

        # - Create a new time series, or modify this time series
        if not inplace:
            tsAppended = self.copy()
        else:
            tsAppended = self

        # - Concatenate time trace and samples
        tsAppended._mfSamples = np.concatenate(
            (tsAppended.mfSamples, tsOther.mfSamples), axis=0
        )

        # - If `self` is empty append new elements directly. Otherwise leafe space
        #   corresponding to median distance between time points in `self._vtTimeTrace`.
        tMedianDT = (
            np.median(np.diff(self._vtTimeTrace)) if self.vtTimeTrace.size > 0 else 0
        )
        tsAppended._vtTimeTrace = np.concatenate(
            (
                tsAppended._vtTimeTrace,
                tsOther.vtTimeTrace + tsAppended.tStop + tMedianDT - tsOther.tStart,
            ),
            axis=0,
        )
        # - Fix tStop
        tsAppended._tStop += tsOther.tDuration + tMedianDT

        # - Recreate interpolator
        tsAppended._create_interpolator()

        # - Return appended time series
        return tsAppended

    def concatenate_t(self, tsOther, inplace: bool = False):
        """
        concatenate_t() - Join together two time series in time

        :param tsOther: Another time series. Will be tacked on to the end of the called series object
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: New concatenated time series
        """
        return self.append_t(tsOther, inplace=inplace)

    def isempty(self):
        """
        isempty() - Is this TimeSeries object empty?

        :return: bool True -> The TimeSeries object contains no samples
        """
        return np.size(self.vtTimeTrace) == 0

    def _create_interpolator(self):
        """
        _create_interpolator - Build an interpolator for the samples in this TimeSeries
        """
        if np.size(self.vtTimeTrace) == 0:
            self.oInterp = lambda o: None
            return

        elif np.size(self.vtTimeTrace) == 1:
            # - Replicate to avoid error in `interp1d`
            vtTimeTrace = np.repeat(self.vtTimeTrace, 2, axis=0)
            mfSamples = np.repeat(self.mfSamples, 2, axis=0)
        else:
            vtTimeTrace = self._vtTimeTrace
            mfSamples = self._mfSamples

        # - Construct interpolator
        self.oInterp = spint.interp1d(
            vtTimeTrace,
            mfSamples,
            kind=self.strInterpKind,
            axis=0,
            assume_sorted=True,
            bounds_error=False,
        )

    def print(
        self,
        full: bool = False,
        num_first: int = 4,
        num_last: int = 4,
        limit_shorten: int = 10,
    ):
        """
        print - Print an overview of the time series and its values.

        :param full:           Print all samples of `self`, no matter how long it is
        :param limit_shorten:  Print shortened version of self if it comprises more
                               than `limit_shorten` time points and `full` is False
        :param num_first:      Shortened version of printout contains samples at first
                               `num_first` points in `self.vtTimeTrace`
        :param num_last:       Shortened version of printout contains samples at last
                               `num_last` points in `self.vtTimeTrace`
        """

        s = "\n"
        if len(self.vtTimeTrace) <= 10 or full:
            summary = s.join(
                [
                    "{}: \t {}".format(t, samples)
                    for t, samples in zip(self.vtTimeTrace, self.mfSamples)
                ]
            )
        else:
            summary0 = s.join(
                [
                    "{}: \t {}".format(t, samples)
                    for t, samples in zip(
                        self.vtTimeTrace[:num_first], self.mfSamples[:num_first]
                    )
                ]
            )
            summary1 = s.join(
                [
                    "{}: \t {}".format(t, samples)
                    for t, samples in zip(
                        self.vtTimeTrace[-num_last:], self.mfSamples[-num_last:]
                    )
                ]
            )
            summary = summary0 + "\n\t...\n" + summary1
        print(self.__repr__() + "\n" + summary)

    def clip(self, vtNewBounds: ArrayLike, inplace: bool = False):
        """
        clip - Clip a TimeSeries to data only within a new set of time bounds (exclusive end points)

        :param vtNewBounds: ArrayLike   [tStart tStop] defining new bounds
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)

        :return: tsClip, vbIncludeSamples)
                tsClip:             New TimeSeries clipped to bounds
                vbIncludeSamples:   boolean ndarray indicating which original samples are included
        """
        # - Create a new time series, or modify this time series
        if not inplace:
            tsClipped = self.copy()
        else:
            tsClipped = self

        # - Get first sample
        vfFirstSample = np.atleast_1d(tsClipped(vtNewBounds[0]))

        # - Get number of traces
        nNumChannels = tsClipped.nNumChannels

        # - For periodic time series, resample the series
        if tsClipped.bPeriodic:
            tsClipped, _ = tsClipped._clip_periodic(vtNewBounds, inplace=inplace)
        else:
            tsClipped, _ = tsClipped._clip(vtNewBounds, inplace=inplace)

        # - Insert initial time point
        tsClipped._vtTimeTrace = np.concatenate(
            ([vtNewBounds[0]], tsClipped._vtTimeTrace)
        )

        # - Insert initial samples

        tsClipped._mfSamples = np.concatenate(
            (
                np.reshape(vfFirstSample, (-1, nNumChannels)),
                np.reshape(tsClipped._mfSamples, (-1, nNumChannels)),
            ),
            axis=0,
        )

        # - Update tStart to contain initial time point
        tsClipped._tStart = min(tsClipped._tStart, vtNewBounds[0])

        return tsClipped

    def _clip(self, vtNewBounds: ArrayLike, inplace: bool = False):
        """
        clip - Clip a TimeSeries to data only within a new set of time bounds (exclusive end points)

        :param vtNewBounds: ArrayLike   [tStart tStop] defining new bounds
        :param inplace:    bool        Conduct operation in-place (Default: False; create a copy)
        :return:  TimeSeries clipped to bounds
        """
        # - Create a new time series, or use self
        if not inplace:
            tsClipped = self.copy()
        else:
            tsClipped = self

        # - Find samples included in new time bounds
        vtNewBounds = np.sort(vtNewBounds)
        vbIncludeSamples = np.logical_and(
            tsClipped.vtTimeTrace >= vtNewBounds[0],
            tsClipped.vtTimeTrace < vtNewBounds[-1],
        )

        # - Build and return TimeSeries
        tsClipped._vtTimeTrace = tsClipped._vtTimeTrace[vbIncludeSamples]

        if np.size(vbIncludeSamples) == 0:
            # - Handle empty data
            tsClipped._mfSamples = np.zeros((0, tsClipped.mfSamples.shape[1]))
        else:
            tsClipped._mfSamples = np.reshape(
                tsClipped._mfSamples, (np.size(vbIncludeSamples), -1)
            )[vbIncludeSamples, :]

        # - Update tStart and tStop
        tsClipped._tStart, tsClipped._tStop = vtNewBounds

        return tsClipped, vbIncludeSamples

    def _clip_periodic(self, vtNewBounds: ArrayLike, inplace: bool = False):
        """
        _clip_periodic - Clip a periodic TimeSeries
        :param vtNewBounds: ArrayLike   [tStart tStop] defining new bounds
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:
        """
        # - Ensure time bounds are sorted
        vtNewBounds = np.sort(vtNewBounds)
        tDuration = np.diff(vtNewBounds)

        # - Create a new time series, or use self
        if not inplace:
            tsClipped = self.copy()
        else:
            tsClipped = self

        # - Catch sinlgeton time point
        if vtNewBounds[0] == vtNewBounds[1]:
            return tsClipped.resample(vtNewBounds[0], inplace=inplace)

        # - Map time bounds to periodic bounds
        vtNewBoundsPeriodic = copy.deepcopy(vtNewBounds)
        vtNewBoundsPeriodic[0] = (
            np.asarray(vtNewBoundsPeriodic[0]) - tsClipped._tStart
        ) % tsClipped.tDuration + tsClipped._tStart
        vtNewBoundsPeriodic[1] = vtNewBoundsPeriodic[0] + tDuration

        # - Build new time trace
        vtNewTimeTrace = copy.deepcopy(tsClipped._vtTimeTrace)
        vtNewTimeTrace = vtNewTimeTrace[vtNewTimeTrace >= vtNewBoundsPeriodic[0]]

        # - Keep appending copies of periodic time base until required duration is reached
        while vtNewTimeTrace[-1] < vtNewBoundsPeriodic[1]:
            vtNewTimeTrace = np.concatenate(
                (vtNewTimeTrace, tsClipped._vtTimeTrace + vtNewTimeTrace[-1])
            )

        # - Trim new time base to end point
        vtNewTimeTrace = vtNewTimeTrace[vtNewTimeTrace <= vtNewBoundsPeriodic[1]]

        # - Restore to original time base
        vtNewTimeTrace = vtNewTimeTrace - vtNewTimeTrace[0] + vtNewBounds[0]

        # - Update tStart and tStop
        tsClipped._tStart = vtNewTimeTrace[0]
        tsClipped._tStop = vtNewTimeTrace[-1]

        # - Return a new clipped time series
        tsClip = tsClipped.resample(vtNewTimeTrace, inplace=inplace)
        return tsClip, None

    def __add__(self, other):
        return self.copy().__iadd__(other)

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        # - Should we handle TimeSeries addition?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self._compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self._compatibleShape(other)

        # - Treat NaNs as zero
        mbIsNanSelf = np.isnan(self.mfSamples)
        mbIsNanOther = np.isnan(mfOtherSamples)
        self.mfSamples[mbIsNanSelf] = 0
        mfOtherSamples[mbIsNanOther] = 0

        # - Perform addition
        mfNewSamples = self.mfSamples + mfOtherSamples
        self.mfSamples = mfNewSamples

        # - Fill in nans
        self.mfSamples[np.logical_and(mbIsNanSelf, mbIsNanOther)] = np.nan

        # - Re-create interpolator
        self._create_interpolator()
        return self

    def __mul__(self, other):
        return self.copy().__imul__(other)

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self._compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self._compatibleShape(other)

        # - Propagate NaNs
        mbIsNanSelf = np.isnan(self.mfSamples)
        mbIsNanOther = np.isnan(mfOtherSamples)

        # - Perform multiplication
        self.mfSamples *= mfOtherSamples

        # - Fill in nans
        self.mfSamples[np.logical_or(mbIsNanSelf, mbIsNanOther)] = np.nan

        # - Re-create interpolator
        self._create_interpolator()
        return self

    def __truediv__(self, other):
        return self.copy().__idiv__(other)

    def __rdiv__(self, other):
        tsCopy = self.copy()
        tsCopy.mfSamples = 1 / tsCopy.mfSamples
        return tsCopy * other

    def __idiv__(self, other):
        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self._compatibleShape(
                np.reshape(other(self.vtTimeTrace), (np.size(self.vtTimeTrace), -1))
            )
        else:
            mfOtherSamples = self._compatibleShape(other)

        # - Propagate NaNs
        mbIsNanSelf = np.isnan(self.mfSamples)
        mbIsNanOther = np.isnan(mfOtherSamples)

        # - Perform division
        self.mfSamples /= mfOtherSamples

        # - Fill in nans
        self.mfSamples[np.logical_or(mbIsNanSelf, mbIsNanOther)] = np.nan

        # - Re-create interpolator
        self._create_interpolator()
        return self

    def __floordiv__(self, other):
        return self.copy().__ifloordiv__(other)

    def __rfloordiv__(self, other):
        tsCopy = self.copy()
        tsCopy.mfSamples = 1 / tsCopy.mfSamples
        return tsCopy // (1 / other)

    def __ifloordiv__(self, other):
        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self._compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self._compatibleShape(other)

        # - Propagate NaNs
        mbIsNanSelf = np.isnan(self.mfSamples)
        mbIsNanOther = np.isnan(mfOtherSamples)

        # - Perform division
        self.mfSamples //= mfOtherSamples

        # - Fill in nans
        self.mfSamples[np.logical_or(mbIsNanSelf, mbIsNanOther)] = np.nan

        # - Re-create interpolator
        self._create_interpolator()
        return self

    def max(self):
        return np.nanmax(self.mfSamples)

    def min(self):
        return np.nanmin(self.mfSamples)

    def __abs__(self):
        tsCopy = self.copy()
        tsCopy.mfSamples = np.abs(tsCopy.mfSamples)
        tsCopy._create_interpolator()
        return tsCopy

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def __rsub__(self, other):
        return -(self - other)

    def __isub__(self, other):
        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self._compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self._compatibleShape(other)

        # - Treat NaNs as zero
        mbIsNanSelf = np.isnan(self.mfSamples)
        mbIsNanOther = np.isnan(mfOtherSamples)
        self.mfSamples[mbIsNanSelf] = 0
        mfOtherSamples[mbIsNanOther] = 0

        # - Perform subtraction
        self.mfSamples -= mfOtherSamples

        # - Fill in nans
        self.mfSamples[np.logical_and(mbIsNanSelf, mbIsNanOther)] = np.nan

        # - Re-create interpolator
        self._create_interpolator()
        return self

    def __neg__(self):
        tsCopy = self.copy()
        tsCopy.mfSamples = -tsCopy.mfSamples
        tsCopy._create_interpolator()
        return tsCopy

    @property
    def vtTimeTrace(self):
        return self._vtTimeTrace

    @vtTimeTrace.setter
    def vtTimeTrace(self, vtNewTrace: ArrayLike):
        # - Check time trace for correct size
        if np.size(vtNewTrace) != np.size(self._vtTimeTrace):
            raise ValueError(
                f"TSContinuous `{self.strName}`: "
                + "New time trace must have the same number of elements as the original trace."
            )

        # - Make sure time trace is sorted
        if (np.diff(vtNewTrace) < 0).any():
            raise ValueError(
                f"TSContinuous `{self.strName}`: "
                + "The time trace must be sorted and not decreasing"
            )

        # - Store new time trace
        self._vtTimeTrace = np.reshape(vtNewTrace, -1)

        if np.size(self._vtTimeTrace) > 0:
            # - Fix tStart and tStop
            self._tStart = min(self._tStart, vtNewTrace[0])
            self._tStop = max(self._tStop, vtNewTrace[-1])

    def choose(self, vnTraces: Union[int, ArrayLike], inplace: bool = False):
        """
        choose() - Select from one of several sub-traces; return a TimeSeries containing these traces

        :param vnTraces:    array-like of indices within source TimeSeries
        :param inplace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:            TimeSeries containing only the selected traces
        """
        # - Convert to a numpy array and check extents
        vnTraces = np.atleast_1d(vnTraces)
        assert (
            min(vnTraces) >= 0 and max(vnTraces) <= self.nNumChannels
        ), "`vnTraces` must be between 0 and " + str(self.nNumChannels)

        if not inplace:
            tsChosen = self.copy()
        else:
            tsChosen = self

        # - Return a TimeSeries with the subselected traces
        tsChosen.mfSamples = tsChosen.mfSamples[:, vnTraces]
        tsChosen._create_interpolator()
        return tsChosen

    def copy(self):
        """
        copy() - Return a deep copy of this time series
        :return: tsCopy
        """
        return copy.deepcopy(self)

    @property
    def tDuration(self) -> float:
        """
        .tDuration: float Duration of TimeSeries
        """
        return self._tStop - self._tStart

    @property
    def tStart(self) -> float:
        """
        .tStart: float Start time
        """
        return self._tStart

    @tStart.setter
    def tStart(self, tNewStart):
        if np.size(self._vtTimeTrace) == 0 or tNewStart <= self._vtTimeTrace[0]:
            self._tStart = tNewStart
        else:
            raise ValueError(
                "TimeSeries `{}`: tStart must be less or equal to {}. It was {}.".format(
                    self.strName, self._vtTimeTrace[0], tNewStart
                )
            )

    @property
    def tStop(self) -> float:
        """
        .tStop: float Stop time
        """
        return self._tStop

    @tStop.setter
    def tStop(self, tNewStop):
        if np.size(self._vtTimeTrace) == 0 or tNewStop >= self._vtTimeTrace[-1]:
            self._tStop = tNewStop
        else:
            raise ValueError(
                "TimeSeries `{}`: tStop must be greater or equal to {}.".format(
                    self.strName, self._vtTimeTrace[-1]
                )
            )

    def _compatibleShape(self, other) -> np.ndarray:
        try:
            if np.size(other) == 1:
                return copy.copy(np.broadcast_to(other, self.mfSamples.shape))

            elif other.shape[0] == self.mfSamples.shape[0]:
                return np.reshape(other, self.mfSamples.shape)

            else:
                raise Exception

        except Exception:
            raise ValueError("Input data must have shape " + str(self.mfSamples.shape))

    def _modulo_period(
        self, times: Union[ArrayLike, float, int]
    ) -> Union[ArrayLike, float, int]:
        """_modulo_period - Calculate provided times modulo `self.tDuration`"""
        return self.t_start + np.modulo(times - self.t_start, self.tDuration)


### --- Continuous-valued time series


class TSContinuous(TimeSeries):
    """
    TSContinuous - Class represent a multi-series time series, with temporal interpolation
                   and periodicity supported
    """

    def __init__(
        self,
        vtTimeTrace: ArrayLike,
        mfSamples: ArrayLike,
        bPeriodic: bool = False,
        tStart: Optional[float] = None,
        tStop: Optional[float] = None,
        strName: str = "unnamed",
        strInterpKind: str = "linear",
    ):
        """
        TSContinuous - Class represent a multi-series time series, with temporal interpolation and periodicity supported

        :param vtTimeTrace:     [Tx1] vector of time samples
        :param mfSamples:       [TxM] matrix of values corresponding to each time sample
        :param bPeriodic:       bool: Treat the time series as periodic around the end points. Default: False
        :param tStart:          float: If not None, the series start time is tStart, otherwise vtTimeTrace[0]
        :param tStop:           float: If not None, the series stop time is tStop, otherwise vtTimeTrace[-1]
        :param strName:         str: Name of the TSContinuous object. Default: `unnamed`
        :param strInterpKind:   str: Specify the interpolation type. Default: 'linear'

        If the time series is not periodic (the default), then NaNs will be returned for any extrapolated values.
        """

        # - Convert everything to numpy arrays
        vtTimeTrace = np.asarray(vtTimeTrace).flatten().astype(float)
        mfSamples = np.atleast_1d(mfSamples).astype(float)

        # - Check arguments
        assert np.size(vtTimeTrace) == mfSamples.shape[0], (
            f"TSContinuous `{strName}`: The number of time samples must be equal to the"
            " first dimension of `mfSamples`"
        )
        assert np.all(
            np.diff(vtTimeTrace) >= 0
        ), f"TSContinuous `{strName}`: The time trace must be sorted and not decreasing"

        # - Initialize superclass
        super().__init__(
            vtTimeTrace=vtTimeTrace,
            bPeriodic=bPeriodic,
            tStart=tStart,
            tStop=tStop,
            strName=strName,
        )

        # - Assign attributes
        self.strInterpKind = strInterpKind
        self._mfSamples = mfSamples.astype("float")
        # - Interpolator for samples
        self._create_interpolator()

    ## -- Methods for plotting and printing

    def plot(
        self,
        vtTimes: Union[int, float, ArrayLike] = None,
        target: Union[mpl.axes.Axes, hv.Curve, hv.Overlay, None] = None,
        channels: Union[ArrayLike, int, None] = None,
        *args,
        **kwargs,
    ):
        """
        plot - Visualise a time series on a line plot

        :param vtTimes: Optional. Time base on which to plot. Default: time base of time series
        :param target:  Optional. Object to which plot will be added.
        :param channels:  Optional. Channels that are to be plotted.
        :param args, kwargs:  Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """
        if vtTimes is None:
            vtTimes = self.vtTimeTrace
            mfSamples = self.mfSamples
        else:
            mfSamples = self(vtTimes)
        if channels is not None:
            mfSamples = mfSamples[:, channels]

        if target is None:
            # - Get current plotting backend from global settings
            _bUseHoloviews, _bUseMatplotlib = GetPlottingBackend()

            if _bUseHoloviews:
                if kwargs == {}:
                    vhCurves = [
                        hv.Curve((vtTimes, vfData)).redim(x="Time")
                        for vfData in mfSamples.T
                    ]
                else:
                    vhCurves = [
                        hv.Curve((vtTimes, vfData))
                        .redim(x="Time")
                        .options(*args, **kwargs)
                        for vfData in mfSamples.T
                    ]

                if len(vhCurves) > 1:
                    return hv.Overlay(vhCurves).relabel(group=self.strName)
                else:
                    return vhCurves[0].relabel(self.strName)

            elif _bUseMatplotlib:
                return plt.plot(vtTimes, mfSamples, label=self.strName, **kwargs)

            else:
                raise RuntimeError(
                    f"TSContinuous: `{self.strName}`: No plotting back-end detected."
                )

        else:
            # - Infer current plotting backend from type of `target`
            if isinstance(target, (hv.Curve, hv.Overlay)):
                if kwargs == {}:
                    for vfData in mfSamples.T:
                        target *= hv.Curve((vtTimes, vfData)).redim(x="Time")
                else:
                    for vfData in mfSamples.T:
                        target *= (
                            hv.Curve((vtTimes, vfData))
                            .redim(x="Time")
                            .options(*args, **kwargs)
                        )
                return target.relabel(group=self.strName)
            elif isinstance(target, mpl.axes.Axes):
                target.plot(vtTimes, mfSamples, label=self.strName, **kwargs)
                return target
            else:
                raise TypeError(
                    f"TSContinuous: `{self.strName}`: Unrecognized type for `target`. "
                    + "It must be matplotlib Axes or holoviews Curve or Overlay."
                )

    def save(self, strPath: str):
        """
        save - Save TSContinuous as npz file using np.savez
        :param strPath:     str  Path to save file
        """
        np.savez(
            strPath,
            vtTimeTrace=self.vtTimeTrace,
            mfSamples=self.mfSamples,
            tStart=self.tStart,
            tStop=self.tStop,
            strInterpKind=self.strInterpKind,
            bPeriodic=self.bPeriodic,
            strName=self.strName,
            strType="TSContinuous",  # Indicate that this object is TSContinuous
        )
        bPathMissesEnding = strPath.split(".")[-1] != "npz"  # np.savez will add ending
        print(
            "TSContinuous `{}` has been stored in `{}`.".format(
                self.strName, strPath + bPathMissesEnding * ".npz"
            )
        )

    ## -- Methods for finding and extracting data

    def contains(self, vtTimeTrace: Union[int, float, ArrayLike]) -> bool:
        """
        contains - Does the time series contain the time range specified in the given time trace?

        :param vtTimeTrace: Array-like containing time points
        :return:            boolean: All time points are contained within this time series
        """
        return (
            True
            if self.tStart <= np.min(vtTimeTrace) and self.tStop >= np.max(vtTimeTrace)
            else False
        )

    # REDUNDANT?
    def find(self, times: ArrayLike) -> (np.ndarray, np.ndarray):
        """find - Convenience function that returns arrays with given times and corresponding
                  samples.
            :param times:  Times of samples that should be returned
            :return:
                np.ndarray  Provided times
                np.ndarray  Corresponding samples
        """
        return np.asarray(vtTime), self(times)

    ## -- Internal methods

    def interpolate(self, vtTimes: Union[int, float, ArrayLike]):
        """
        interpolate - Interpolate the time series to the provided time points

        :param vtTimes: np.ndarray of T desired interpolated time points
        :return:        np.ndarray of interpolated values. Will have the shape TxN
        """
        # - Enforce periodicity
        if self.bPeriodic:
            vtTimes = (
                np.asarray(vtTimes) - self._tStart
            ) % self.tDuration + self._tStart

        return np.reshape(self.oInterp(vtTimes), (-1, self.nNumChannels))

    ## -- Magic methods

    def __call__(self, vtTimes: Union[int, float, ArrayLike]):
        """
        ts(tTime1, tTime2, ...) - Interpolate the time series to the provided time points

        :param tTime: Scalar, list or np.array of T desired interpolated time points
        :return:      np.array of interpolated values. Will have the shape TxN
        """
        return self.interpolate(vtTimes)

    def __getitem__(self, vtTimes: Union[ArrayLike, slice]) -> np.ndarray:
        """
        ts[tTime1, tTime2, ...] - Interpolate the time series to the provided time points
        NOTE that ts[:] uses as (fixed) step size the mean step size of self.vtTimeTrace
        and thus can return different values than those in ts.mfSamples!
        :param vtTimes: Slice, scalar, list or np.array of T desired interpolated time points
        :return:        np.array of interpolated values. Will have the shape TxN
        """
        if isinstance(vtTimes, slice):
            fStep: np.float = (
                np.mean(np.diff(self._vtTimeTrace))
                if vtTimes.step is None
                else vtTimes.step
            )
            fStart: float = self._vtTimeTrace[
                0
            ] if vtTimes.tStart is None else vtTimes.tStart
            fStop: float = (
                self._vtTimeTrace[-1] + abs(fStep)
                if vtTimes.tStop is None
                else vtTimes.tStop
            )

            if fStart < self._vtTimeTrace[0]:
                raise ValueError(
                    f"TSContinuous `{self.strName}`: "
                    f"This TimeSeries only starts at t={self._vtTimeTrace[0]}"
                )
            if fStop > self._vtTimeTrace[-1] + abs(fStep):
                raise ValueError(
                    f"TSContinuous `{self.strName}`: "
                    f"This TimeSeries already ends at t={self._vtTimeTrace[-1]}"
                )

            # - Determine time points at which series is sampled
            time_points: np.ndarray = np.arange(fStart, fStop, abs(fStep))
            # - Invert order if fStep is negative
            if fStep < 0:
                time_points = time_points[::-1]

            return self.interpolate(vTimeIndices)

        else:
            return self.interpolate(vtTimes)

    def __repr__(self):
        """
        __repr__() - Return a string representation of this object
        :return: str String description
        """
        if self.isempty():
            return f"Empty TSContinuous object `{self.strName}`"
        else:
            return "{}periodic TSContinuous object `{}` from t={} to {}. Shape: {}".format(
                int(not self.bPeriodic) * "non-",
                self.__class__.__name__,
                self.strName,
                self.tStart,
                self.tStop,
                self.mfSamples.shape,
            )

    ## -- Properties

    @property
    def mfSamples(self):
        return self._mfSamples

    @mfSamples.setter
    def mfSamples(self, mfNewSamples: ArrayLike):
        # - Make sure that if assigned empty samples array, number of traces is implicityly
        #   with as second dimension of `mfNewSamples`
        if np.size(mfNewSamples) == 0 and np.ndim(mfNewSamples) > 2:
            raise ValueError(
                f"TSContinuous `{self.strName}`: "
                + "Empty mfSampels object must be 2D to allow infering the number of channels."
            )

        # - Promote to 2d
        mfNewSamples = np.atleast_2d(mfNewSamples)

        # - Permit a one-dimensional sample input, promote to 2d
        if (mfNewSamples.shape[0] == 1) and (np.size(self.vtTimeTrace) > 1):
            mfNewSamples = np.reshape(mfNewSamples, (np.size(self.vtTimeTrace), 1))

        # - Check samples for correct size
        if mfNewSamples.shape[0] != np.size(self.vtTimeTrace):
            raise ValueError(
                f"TSContinuous `{self.strName}`: "
                + "New samples matrix must have the same number of samples as `.vtTimeTrace`."
            )

        # - Store new time trace
        self._mfSamples = mfNewSamples

        # - Create a new interpolator
        self._create_interpolator()

    # - Extend setter of vtTimeTrace to update interpolator
    @property
    def vtTimeTrace(self):
        return super().vtTimeTrace

    @vtTimeTrace.setter
    def vtTimeTrace(self, vtNewTrace: ArrayLike):
        super().vtTimeTrace.fset(self, vtNewTrace)
        # - Create a new interpolator
        self._create_interpolator()

    @property
    def nNumTraces(self):
        """nNumTraces - Synonymous to nNumChannels"""
        return self.mfSamples.shape[1]

    @property
    def nNumChannels(self):
        """nNumChannels: int Number of channels (dimension of sample vectors) in this TimeSeries object"""
        return self.mfSamples.shape[1]


### --- Event time series


class TSEvent(TimeSeries):
    def __init__(
        self,
        vtTimeTrace: ArrayLike,
        vnChannels: Union[int, ArrayLike] = None,
        bPeriodic: bool = False,
        tStart: Optional[float] = None,
        tStop: Optional[float] = None,
        strName: str = None,
        nNumChannels: int = None,
    ):
        """
        TSEvent - Represent discrete events in time

        :param vtTimeTrace:     np.array float Tx1 vector of event times
        :param vnChannels:      np.array int Tx1 vector of event channels (Default: channel 0)

        :param bPeriodic:       bool Is this a periodic TimeSeries (Default: False)

        :param tStart:          float: If not None, the series start time is tStart, otherwise vtTimeTrace[0]
        :param tStop:           float: If not None, the series stop time is tStop, otherwise vtTimeTrace[-1]

        :param strName:         str Name of the time series (Default: None)

        :param nNumChannels:    int Total number of channels in the data source. If None,
                                    it is inferred from the max channel ID in vnChannels
        """

        # - Default time trace: empty
        if vtTimeTrace is None:
            vtTimeTrace = np.array([])
        else:
            vtTimeTrace = np.array(vtTimeTrace, "float").flatten()

        # - Default channel: zero
        if vnChannels is None or np.size(vnChannels) == 0:
            vnChannels = np.zeros(np.size(vtTimeTrace))
            nMinNumChannels = min(np.size(vtTimeTrace), 1)
        # - Handle scalar channel
        elif isinstance(vnChannels, int):
            nMinNumChannels = vnChannels + 1
            vnChannels = np.array([vnChannels for _ in vtTimeTrace])
        # - Array-like of channels
        else:
            if np.size(vnChannels) != np.size(vtTimeTrace):
                # - Make sure item sizes match
                raise ValueError(
                    f"TSEvent `{strName}`: `vnChannels` must have the same number of "
                    + "elements as `vtTimeTrace`, be an integer or None."
                )
            else:
                nMinNumChannels = np.amax(vnChannels) + 1

        if nNumChannels is None:
            # - Infer number of channels from maximum channel id in vnChannels
            nNumChannels = nMinNumChannels
        else:
            if nNumChannels < nMinNumChannels:
                raise ValueError(
                    f"TSEvent `{strName}`: nNumChannels must be None or greater than the highest channel ID."
                )

        # - Initialize superclass
        super().__init__(
            vtTimeTrace=vtTimeTrace,
            bPeriodic=bPeriodic,
            tStart=tStart,
            tStop=tStop,
            strName=strName,
        )

        # - Store channels
        self.vnChannels = np.array(vnChannels, "int").flatten()

        # - Store total number of channels
        self._nNumChannels = int(nNumChannels)

    def print(
        self,
        full: bool = False,
        num_first: int = 4,
        num_last: int = 4,
        limit_shorten: int = 10,
    ):
        """
        print - Print an overview of the time series and its values.

        :param full:           Print all samples of `self`, no matter how long it is
        :param limit_shorten:  Print shortened version of self if it comprises more
                               than `limit_shorten` time points and `full` is False
        :param num_first:      Shortened version of printout contains samples at first
                               `num_first` points in `self.vtTimeTrace`
        :param num_last:       Shortened version of printout contains samples at last
                               `num_last` points in `self.vtTimeTrace`
        """

        s = "\n"
        if len(self.vtTimeTrace) <= 10 or full:
            summary = s.join(
                [f"{t}: \t {ch}" for t, ch in zip(self.vtTimeTrace, self.vnChannels)]
            )
        else:
            summary0 = s.join(
                [
                    f"{t}: \t {ch}"
                    for t, ch in zip(
                        self.vtTimeTrace[:num_first], self.vnChannels[:num_first]
                    )
                ]
            )
            summary1 = s.join(
                [
                    f"{t}: \t {ch}"
                    for t, ch in zip(
                        self.vtTimeTrace[-num_last:], self.vnChannels[-num_last:]
                    )
                ]
            )
            summary = summary0 + "\n\t...\n" + summary1
        print(self.__repr__() + "\nTime \t Ch.-ID" + "\n" + summary)

    ## -- Methods for plotting and printing

    def plot(
        self,
        time_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        target: Union[mpl.axes.Axes, hv.Scatter, hv.Overlay, None] = None,
        channels: Union[ArrayLike, int, None] = None,
        *args,
        **kwargs,
    ):
        """
        plot - Visualise a time series on a line plot

        :param time_limits: Optional. Tuple with times between which to plot
        :param target:  Optional. Object to which plot will be added.
        :param channels:  Optional. Channels that are to be plotted.
        :param args, kwargs:  Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """
        # - Filter spikes by time
        if time_limits is None:
            t_start = self.tStart
            t_stop = self.tStop
        else:
            execption_limits = (
                f"TSEvent `{self.strName}`: `time_limits` must be None or tuple "
                + "of length 2."
            )
            try:
                # - Make sure `time_limits` has correct length
                if len(time_limits) != 2 or not isin:
                    raise ValueError(execption_limits)
                else:
                    t_start = self.tStart if time_limits[0] is None else time_limits[0]
                    t_start = self.tStop if time_limits[1] is None else time_limits[1]
            except TypeError:
                raise TypeError(exception_limits)
        # - Choose matching events
        times, channels = self(t_start, t_stop, channels)

        if target is None:
            # - Get current plotting backend from global settings
            _bUseHoloviews, _bUseMatplotlib = GetPlottingBackend()

            if _bUseHoloviews:
                return (
                    hv.Scatter((times, channels), *args, **kwargs)
                    .redim(x="Time", y="Channel")
                    .relabel(self.strName)
                )

            elif _bUseMatplotlib:
                return plt.scatter(times, channels, label=self.strName, *args, **kwargs)

            else:
                raise RuntimeError(
                    f"TSEvent: `{self.strName}`: No plotting back-end detected."
                )

        else:
            # - Infer current plotting backend from type of `target`
            if isinstance(target, (hv.Curve, hv.Overlay)):
                target *= (
                    hv.Scatter((times, channels), *args, **kwargs)
                    .redim(x="Time", y="Channel")
                    .relabel(self.strName)
                )
                return target.relabel(group=self.strName)
            elif isinstance(target, mpl.axes.Axes):
                target.scatter(times, channels, label=self.strName, *args, **kwargs)
                return target
            else:
                raise TypeError(
                    f"TSEvent: `{self.strName}`: Unrecognized type for `target`. "
                    + "It must be matplotlib Axes or holoviews Curve or Overlay."
                )

    ## -- Methods for finding and extracting data

    # REDUNDANT?
    def find(
        self, vtTimeBounds: Optional[ArrayLike] = None
    ) -> (np.ndarray, np.ndarray):
        """
        find - Return events that fall within a range

        :param vtTimeBounds: array-like [tStart tFinish]

        :return:    (vtMatchingTimes, vnMatchingChannels, vfMatchingSamples)
                vtMatchingTimes:    np.ndarray Event times falling within time bounds (exclusive end time)
                vnMatchingChannels: np.ndarray Channels corresponding to matching times
                vfMatchingSamples:  np.ndarray Samples corresponding to matching times
        """
        warn(
            DeprecationWarning(
                f"TSContinuous `{self.strName}`: The `find` method is deprecated. "
                + "Call the object as a function instead, with (optional) arguments "
                + "`t_start`, `t_stop`, `channels`."
            )
        )
        return self(*(None if vtTimeBounds is None else vtTimeBounds))

    def clip(
        self,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
        channels: Union[int, ArrayLike, None] = None,
        include_stop: bool = False,
        compress_channels: bool = False,
        inplace: bool = False,
    ) -> TSEventType:
        """
        clip - Return a TSEvent which is restricted to given time limits and only
                 contains events of selected channels. If time limits are provided,
                 tStart and tStop attributes will correspond to those. If
                 `compress_channels` is true, channels IDs will be mapped to continuous
                 sequence of integers starting from 0 (e.g. [1,3,6]->[0,1,2]). In this
                 case `nNumChannels` will be set to the largest new channel ID + 1.
                 Otherwise it will keep its original values, which is also the case for
                 all other attributes.
                 If `inplace` is True, modify `self` accordingly.

        :param t_start:       Time from which on events are returned
        :param t_stop:        Time until which events are returned
        :param channels:      Channels of which events are returned
        :param include_stop:  If there are events with time t_stop include them or not
        :param compress_channels:  Map channel IDs to continuous sequence startign from 0.
                                   Set `nNumChannels` to largest new ID + 1.
        :param inplase:       Specify whether operation should be performed in place (Default: False)
        :return: TSEvent containing events from the requested channels
        """

        if not inplace:
            new_series = self.copy()
        else:
            new_series = self

        # - Extract matching events
        times, channels = new_series(t_start, t_stop, channels, include_stop)

        # - Update new timeseries
        new_series._vtTimeTrace = times
        if t_start is not None:
            new_series._tStart = t_start
        if t_stop is not None:
            new_series._tStop = t_stop
        if compress_channels and channels.size > 0:
            # - Set channel IDs to sequence starting from 0
            unique_channels, channel_indices = np.unique(channels, return_inverse=True)
            nNumChannels = unique_channels.size
            new_series._vnChannels = np.arange(nNumChannels)[channel_indices]
            new_series.nNumChannels = nNumChannels
        else:
            new_series._vnChannels = channels

        return new_series

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
        raster - Return rasterized time series data, where each data point
                 represents a time step. Events are represented in a boolean
                 matrix, where the first axis corresponds to time, the second
                 axis to the channel.
                 Events that happen between time steps are projected to the
                 preceding one. If two events happen during one time step
                 within a single channel, they are counted as one, unless
                 add_events is True.
        :param dt:      Length of single time step in raster
        :param tStart:  Time where to start raster - Will start at self.tStart if None
        :param t_stop:  Time where to stop raster. This time point is not included anymore.
                        If None, will use all points until (and including) self.tStop.
                        If num_timesteps is set, t_stop is ignored.
        :param num_timesteps: Can be used to determine number of time steps directly,
                              directly, instead of providing t_stop
        :param channels:      Array-like Channels, from which data is to be used.
        :param add_events:    bool If True, return integer raster containing number of
                              events for each time step and channel

        :return
            event_raster    Boolean matrix with True indicating presence of events
                            for each time step and channel. If add_events == True,
                            the raster consists of integers, indicating the number
                            of events per time step and channel.
                            First axis corresponds to time, second axis to channel.
        """
        # - Filter time and channels
        t_start = self.tStart if t_start is None else t_start
        if num_timesteps is None:
            series = self.clip(
                t_start=t_start,
                t_stop=t_stop,
                channels=channels,
                compress_channels=True,
            )
            # - Make sure that last point is also included if `tDuration` is a
            #   multiple of dt. Therefore floor(...) + 1
            num_timesteps = int(np.floor((series.tDuration) / dt)) + 1
        else:
            t_stop = t_start + num_timesteps * dt
            series = self.clip(
                t_start=t_start,
                t_stop=t_stop,
                channels=channels,
                compress_channels=True,
            )

        # - Raster for storing event data
        raster_type = int if add_events else bool
        event_raster = np.zeros((num_timesteps, series.nNumChannels), raster_type)

        # - Handle empty series
        if len(series) == 0:
            return event_raster

        # - Select data according to time base
        event_times = series.vtTimeTrace
        event_channels = series.vnChannels

        ## -- Convert input events and samples to boolen or integer raster
        # - Only consider rasters that have non-zero length
        if num_timesteps > 0:
            # Compute indices for times
            time_indices = np.floor((event_times - t_start) / dt).astype(int)
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
                        f"TSEvent `{self.strName}`: There are channels with multiple events"
                        + " per time step. Consider smaller dt or setting add_events True."
                    )
                # Mark spiking indices with True
                event_raster[time_indices, event_channels] = True

        return event_raster

    def xraster(
        self,
        dt: float,
        tStart: Optional[float] = None,
        tStop: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        channels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        xraster - Yields a rasterized time series data, where each data point
                 represents a time step. Events are represented in a boolean
                 matrix, where the first axis corresponds to time, the second
                 axis to the channel.
                 Events that happen between time steps are projected to the
                 preceding one. If two events happen during one time step
                 within a single channel, they are counted as one.
        :param dt:      Length of single time step in raster
        :param tStart:  Time where to start raster - Will start at self.tStart if None
        :param t_stop:  Time where to stop raster. This time point is not included anymore.
                        If None, will use all points until (and including) self.tStop.
                        If num_timesteps is set, t_stop is ignored.
        :param num_timesteps: Can be used to determine number of time steps directly,
                              directly, instead of providing t_stop
        :param channels:      Array-like Channels, from which data is to be used.

        :yields
            vbEventsRaster  Boolean matrix with True indicating event axis corresponds to channel
        """
        event_raster = self.raster(
            dt=dt,
            t_start=t_start,
            t_stop=t_stop,
            num_timesteps=num_timesteps,
            channels=channels,
        )
        yield from event_raster  # Yield one row at a time

    def save(self, path: str):
        """
        save - Save TSEvent as npz file using np.savez
        :param strPath:     str  Path to save file
        """
        np.savez(
            path,
            vtTimeTrace=self.vtTimeTrace,
            vnChannels=self.vnChannels,
            tStart=self.tStart,
            tStop=self.tStop,
            bPeriodic=self.bPeriodic,
            nNumChannels=self.nNumChannels,
            strName=self.strName,
            strType="TSEvent",  # Indicate that the object is TSEvent
        )
        missing_ending = path.split(".")[-1] != "npz"  # np.savez will add ending
        print(
            "TSEvent `{}` has been stored in `{}`.".format(
                self.strName, path + missing_ending * ".npz"
            )
        )

    ## -- Methods for manipulating or combining time series

    def append_c(self, other_series: TSEventType, inplace: bool = False) -> TSEventType:
        """
        append_c - Spatially append another time series to this one, so that the other
                   series' channel IDs are shifted by `self.nNumChannels`. The event
                   times remain the same.

        :param other_series: TSEvent or list thereof that will be included in `self`.
        :param inplace:      Conduct operation in-place (Default: False; create a copy)
        :return: TSEvent containing current data, with other TS appended spatially
        """

        # - Create a new time series, or modify this time series
        if not inplace:
            appended_series = self.copy()
        else:
            appended_series = self

        # - Ensure we have a list of timeseries to work on
        if not isinstance(other_series, collections.abc.Iterable):
            series_list = [appended_series, other_series]
        else:
            series_list = [appended_series] + list(other_series)

        # - Check series class
        if not all(isinstance(series, TSEvent) for series in series_list):
            raise TypeError(
                f"TSEvent `{self.strName}`: Can only append `TSEvent` objects."
            )

        # - Determine tStart and tStop
        t_start_new = min(series.tStart for series in series_list)
        t_stop_new = max(series.tStop for series in series_list)

        # - Determine number of channels for each series
        nums_channels = [series.nNumChannels for series in series_list]
        print(nums_channels)
        # - Shift for each TSEvent's vnChannels
        channel_shifts = np.cumsum([0] + nums_channels[:-1])
        print(channel_shifts)

        # - Stop if no non-empty series is left
        if not series_list:
            return appended_series

        # - Merge all samples
        times_new = np.concatenate([series.vtTimeTrace for series in series_list])
        channels_new = np.concatenate(
            [
                series.vnChannels + shift
                for series, shift in zip(series_list, channel_shifts)
            ]
        )
        print(*(series.vnChannels for series in series_list))
        print(channels_new)

        # - Sort on time and merge
        sort_indices = np.argsort(times_new)
        appended_series._vtTimeTrace = times_new[sort_indices]
        appended_series._vnChannels = channels_new[sort_indices].astype(int)
        appended_series._tStart = t_start_new
        appended_series._tStop = t_stop_new
        appended_series._nNumChannels = int(np.sum(nums_channels))

        return appended_series

    def append_t(
        self,
        other_series: Union[TimeSeries, Iterable[TimeSeries]],
        offset: float = 0,
        remove_duplicates: bool = False,
        inplace: bool = False,
    ) -> TSEventType:
        """
        append_t - Append another time series to this one, in time, so that the other
                   series' `tStart` is shifted to `tStop+offset` of `self`.

        :param other_series: TSEvent or list thereof that will be tacked on to the end of `self`
        :param offset:       Scalar or iterable with at least the same number of elements as
                             other_series. If scalar, use same value for all timeseries.
                             Shift `tStart` of corresponding series from `self.tStop` by this value.
        :param remove_duplicates:  Remove duplicate events in resulting timeseries
        :param inplace:      Conduct operation in-place (Default: False; create a copy)
        :return: TSEvent containing current data, with other TS appended in time
        """

        # - Ensure we have a list of timeseries to work on
        if not isinstance(other_series, collections.abc.Iterable):
            other_series = [other_series]
        else:
            other_series = list(other_series)
        # - Same for offsets
        if not isinstance(offset, collections.abc.Iterable):
            offset_list = [offset] * len(other_series)
        else:
            offset_list = list(offset)
            if len(offset_list) != len(other_series):
                warn(
                    f"TSEvent `{self.strName}`: Numbers of provided offsets and TSEvent "
                    + "objects do not match. Will ignore excess elements."
                )

        # - Translate offsets so that they correspond to indiviual delays for each series
        # Delay for first appended series:
        delay1 = offset_list[0] + self.tStop - other_series[0].tStart
        delay_list = [delay1]
        # Add delays for other lists
        for prev_series, curr_series, offset in zip(
            other_series[:-1], other_series[1:], offset_list[1:]
        ):
            # Time at which previous series stops
            stop_previous = delay_list[-1] + prev_series.tStop
            # Delay for current series
            delay_list.append(stop_previous + offset - curr_series.tStart)
        print(delay_list)
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
                f"TSEvent `{self.strName}`: Can only append `TSEvent` objects."
            )

    def merge(
        self,
        other_series: Union[TimeSeries, Iterable[TimeSeries]],
        delay: Union[float, Iterable[float]] = 0,
        remove_duplicates: bool = False,
        inplace: bool = False,
    ) -> TSEventType:
        """
        merge - Merge another TSEvent into this one so that they may overlap in time
        :param other_series:       TimeSeries (or list of TimeSeries) to merge into this one
        :param delay:             Scalar or iterable with at least the number of elements
                                   as other_series. If scalar, use same value for all
                                   timeseries. Delay corresponding series by this value.
        :param remove_duplicates:  Remove duplicate events in resulting timeseries
        :param inplace:            Specify whether operation should be performed in place (Default: False)
        :return: self with new samples included
        """

        # - Create a new time series, or modify this time series
        if not inplace:
            merged_series = self.copy()
        else:
            merged_series = self

        # - Ensure we have a list of timeseries to work on
        if not isinstance(other_series, collections.abc.Iterable):
            series_list = [merged_series, other_series]
        else:
            series_list = [merged_series] + list(other_series)
        # - Same for offsets
        if not isinstance(delay, collections.abc.Iterable):
            delay_list = [0] + [delay] * (len(series_list) - 1)
        else:
            delay_list = [0] + list(delay)
            if len(delay_list) != len(series_list):
                warn(
                    f"TSEvent `{self.strName}`: Numbers of provided offsets and TSEvent "
                    + "objects do not match. Will ignore excess elements."
                )

        # - Check series class
        if not all(isinstance(series, TSEvent) for series in series_list):
            raise TypeError(
                f"TSEvent `{self.strName}`: Can only merge with `TSEvent` objects."
            )

        # - Determine number of channels
        merged_series._nNumChannels = int(
            np.amax([series.nNumChannels for series in series_list])
        )
        # - Determine tStart and tStop
        t_start_new = min(series.tStart for series in series_list)
        t_stop_new = max(series.tStop for series in series_list)

        # - Filter out empty series and apply delay
        series_list = [
            series.delay(delay)
            for series, delay in zip(series_list, delay_list)
            if not series.isempty()
        ]

        # - Stop if no non-empty series is left
        if not series_list:
            return merged_series

        # - Merge all samples
        times_new = np.concatenate([series.vtTimeTrace for series in series_list])
        channels_new = np.concatenate([series.vnChannels for series in series_list])

        # - Remove events with same times and channels
        if remove_duplicates:
            times_new, channels_new = np.unique((times_new, channels_new), axis=1)

        # - Sort on time and merge
        sort_indices = np.argsort(times_new)
        merged_series._vtTimeTrace = times_new[sort_indices]
        merged_series._vnChannels = channels_new[sort_indices].astype(int)
        merged_series._tStart = t_start_new
        merged_series._tStop = t_stop_new

        return merged_series

    ## -- Internal methods

    def _modulo_period(
        self, times: Union[ArrayLike, float, int]
    ) -> Union[ArrayLike, float, int]:
        """_modulo_period - Calculate provided times modulo `self.tDuration`"""
        return self.t_start + np.modulo(times - self.t_start, self.tDuration)

    def _matching_channels(
        self, channels: Union[int, ArrayLike, None] = None
    ) -> np.ndarray:
        """
        _matching_channels - Return boolean array of which events match channel selection
        :param channels:  Channels of which events are to be indicated True
        :return: (vtTimeTrace, vnChannels) containing events form the requested channels
        """

        if channels is None:
            channels = np.arange(self.nNumChannels)
        else:
            # - Check `channels` for validity
            if not (np.min(channels) >= 0 and np.max(channels) < self.nNumChannels):
                raise IndexError(
                    f"TSEvent `{self.strName}`: `channels` must be between 0 and {self.nNumChannels}."
                )
        # - Make sure elements in `channels` are unique for better performance
        channels = np.unique(channels)

        # - Boolean array of which events match selected channels
        include_events = np.isin(self._vnChannels, channels)

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
        ts(t_start, t_stop) - Return events in interval between indicated times

        :param t_start:     Time from which on events are returned
        :param t_stop:      Time until which events are returned
        :param channels:  Channels of which events are returned
        :param include_stop:  If there are events with time t_stop include them or not
        :return:
            np.ndarray  Times of events
            np.ndarray  Channels of events
        """
        # - Handle empty TSEvent
        if self.isempty():
            return np.array([]), np.array([], int)

        if t_start is None:
            t_start: float = self.tStart
        if t_stop is None:
            t_stop: float = self.tStop
            include_stop = True
        # - Permit unsorted bounds
        if t_stop < t_start:
            t_start, t_stop = t_stop, t_start
        # - Events with matching channels
        channel_matches = self._matching_channels(channels)

        if self.bPeriodic:
            # - Repeat events sufficiently often
            # Number of additional required repetitions to append before and after
            num_reps_after = (
                int(np.ceil((t_stop - self.tStart) / self.tDuration))
                if t_stop > self.tStop
                else 1
            )
            num_reps_before = (
                int(np.ceil((self.tStart - t_start) / self.tDuration))
                if t_start < self.tStart
                else 0
            )
            num_reps_total = num_reps_before + num_reps_after
            all_times = np.tile(self.vtTimeTrace, num_reps_total)
            # - Correct times so they extend over the prolongued period and do not repeat
            # Enumerate periods so that originally defined period is 0
            periods = np.arange(num_reps_total) - num_reps_before
            all_times += self.tDuration * np.repeat(periods, self.vtTimeTrace.size)
            all_channels = np.tile(self.vnChannels, num_reps_total)
        else:
            all_times = self.vtTimeTrace
            all_channels = self.vnChannels

        if include_stop:
            choose_events_stop: np.ndarray = all_times <= t_stop
        else:
            choose_events_stop: np.ndarray = all_times < t_stop

        choose_events: np.ndarray = (
            (all_times >= t_start) & (choose_events_stop) & channel_matches
        )
        return all_times[choose_events], all_channels[choose_events]

    def __getitem__(self, ind: Union[ArrayLike, slice, int]) -> TSEventType:
        """
        ts[tTime1, tTime2, ...] - Index the events of `self` by with the argument
                                  and return TSEvent with corresponding events.
                                  Other attributes, including `nNumChannels` and `tDuration`
                                  are the same as in `self`.
        :return:
            np.array of indexed event times
            np.array of indexed event channels
        """
        indexed_times: np.ndarray = np.atleast_1d(self.vtTimeTrace[ind])
        indexed_channels: np.ndarray = np.atleast_1d(self.vnChannels[ind])
        # - New TSEvent with the selected events
        new_series = self.copy()
        new_series._vtTimeTrace = indexed_times
        new_series._vnChannels = indexed_channels
        return new_series

    def __repr__(self):
        """
        __repr__() - Return a string representation of this object
        :return: str String description
        """
        if self.isempty():
            return f"Empty `TSEvent` object `{self.strName}`"
        else:
            return "{}periodic `TSEvent` object `{}` from t={} to {}. Channels: {}. Events: {}".format(
                int(not self.bPeriodic) * "non-",
                self.strName,
                self.tStart,
                self.tStop,
                self.nNumChannels,
                self.vtTimeTrace.size,
            )

    def __len__(self):
        return self._vtTimeTrace.size

    ## -- Properties

    @property
    def vnChannels(self):
        return self._vnChannels

    @vnChannels.setter
    def vnChannels(self, vnNewChannels):
        # - Check size of new data
        assert np.size(vnNewChannels) == 1 or np.size(vnNewChannels) == np.size(
            self.vtTimeTrace
        ), "`vnNewChannels` must be the same size as `vtTimeTrace`."

        # - Handle scalar channel
        if np.size(vnNewChannels) == 1:
            vnNewChannels = np.repeat(vnNewChannels, np.size(self._vtTimeTrace))

        # - Assign vnChannels
        self._vnChannels = vnNewChannels

    @property
    def nNumChannels(self):
        return self._nNumChannels

    @nNumChannels.setter
    def nNumChannels(self, nNewNumChannels):
        nMinNumChannels = np.amax(self.vnChannels)
        if nNewNumChannels < nMinNumChannels:
            raise ValueError(
                f"TSContinuous `{self.strName}`: `nNumChannels` must be at least {nMinNumChannels}."
            )
        else:
            self._nNumChannels = nNewNumChannels


def load_ts_from_file(
    strPath: str, strExpectedType: Optional[str] = None
) -> TimeSeries:
    """
    load_ts_from_file - Load a timeseries object from an npz file.
    :param strPath:     str Filepath to load file
    :param strExpectedType:   str  Specify expected type of timeseires (TSContinuous or TSEvent)
    :return:
        Loaded time series object
    """
    # - Load npz file from specified path
    dLoaded = np.load(strPath)
    # - Check for expected type
    strLoadedType = dLoaded["strType"].item()
    if strExpectedType is not None:
        if not strLoadedType == strExpectedType:
            raise TypeError(
                "Timeseries at `{}` is of type `{}`, which does not match expected type `{}`.".format(
                    strPath, strLoadedType, strExpectedType
                )
            )
    if strLoadedType == "TSContinuous":
        return TSContinuous(
            vtTimeTrace=dLoaded["vtTimeTrace"],
            mfSamples=dLoaded["mfSamples"],
            tStart=dLoaded["tStart"].item(),
            tStop=dLoaded["tStop"].item(),
            strInterpKind=dLoaded["strInterpKind"].item(),
            bPeriodic=dLoaded["bPeriodic"].item(),
            strName=dLoaded["strName"].item(),
        )
    elif strLoadedType == "TSEvent":
        return TSEvent(
            vtTimeTrace=dLoaded["vtTimeTrace"],
            vnChannels=dLoaded["vnChannels"],
            tStart=dLoaded["tStart"].item(),
            tStop=dLoaded["tStop"].item(),
            bPeriodic=dLoaded["bPeriodic"].item(),
            nNumChannels=dLoaded["nNumChannels"].item(),
            strName=dLoaded["strName"].item(),
        )
    else:
        raise TypeError("Type `{}` not supported.".format(strLoadedType))
