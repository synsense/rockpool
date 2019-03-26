###
# timeseries.py - Classes to manage time series
###

import numpy as np
import scipy.interpolate as spint
from warnings import warn
import copy
from typing import Union, List, Tuple, Optional, TypeVar
import collections
from functools import reduce

# - Define exports
__all__ = [
    "TimeSeries",
    "SetPlottingBackend",
    "GetPlottingBackend",
    "TSEvent",
    "TSContinuous",
]

# - Type alias for array-like objects and for yet undefined objects
ArrayLike = Union[np.ndarray, List, Tuple]
TSEventType = TypeVar("TSEvent")

### -- Code for setting plotting backend

__bHoloviewsDetected = False
__bMatplotlibDetected = False
__bUseHoloviews = False
__bUseMatplotlib = False

try:
    import holoviews as hv

    __bHoloviewsDetected = True

except Exception:
    pass

try:
    import matplotlib.pyplot as plt

    __bMatplotlibDetected = True

except Exception:
    pass

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9


def SetPlottingBackend(strBackend):
    global __bHoloviewsDetected
    global __bMatplotlibDetected
    global __bUseHoloviews
    global __bUseMatplotlib
    if (
            strBackend in ("holoviews", "holo", "Holoviews", "HoloViews", "hv")
            and __bHoloviewsDetected
    ):
        __bUseHoloviews = True
        __bUseMatplotlib = False

    elif (
            strBackend in ("matplotlib", "mpl", "mp", "pyplot", "plt")
            and __bMatplotlibDetected
    ):
        __bUseHoloviews = False
        __bUseMatplotlib = True

    else:
        __bUseHoloviews = False
        __bUseMatplotlib = False


def plotting_backend(strBackend):
    SetPlottingBackend(strBackend)


def GetPlottingBackend():
    return __bUseHoloviews, __bUseMatplotlib


## - Set default plotting backend
if __bHoloviewsDetected:
    SetPlottingBackend("holoviews")

elif __bMatplotlibDetected:
    SetPlottingBackend("matplotlib")


### - Convenience method to return a nan array
def full_nan(vnShape: Union[tuple, int]):
    a = np.empty(vnShape)
    a.fill(np.nan)
    return a


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

    def __getitem__(self, vtTimes: Union[ArrayLike, slice]):
        """
        ts[tTime1, tTime2, ...] - Interpolate the time series to the provided time points
        NOTE that ts[:] uses as (fixed) step size the mean step size of self.vtTimeTrace
        and thus can return different values than those in ts.mfSamples!
        :param vtTimes: Slice, scalar, list or np.array of T desired interpolated time points
        :return:        np.array of interpolated values. Will have the shape TxN
        """
        if isinstance(vtTimes, slice):
            fStep = (
                np.mean(np.diff(self._vtTimeTrace))
                if vtTimes.step is None
                else vtTimes.step
            )
            fStart = self._vtTimeTrace[0] if vtTimes.start is None else vtTimes.start
            fStop = (
                self._vtTimeTrace[-1] + abs(fStep)
                if vtTimes.stop is None
                else vtTimes.stop
            )

            assert (
                    fStart >= self._vtTimeTrace[0]
            ), "This TimeSeries only starts at t={}".format(self._vtTimeTrace[0])
            assert fStop <= self._vtTimeTrace[-1] + abs(
                fStep
            ), "This TimeSeries already ends at t={}".format(self._vtTimeTrace[-1])

            vTimeIndices = np.arange(fStart, fStop, abs(fStep))[:: int(np.sign(fStep))]
            return self.interpolate(vTimeIndices)
        else:
            return self.interpolate(vtTimes)

    def delay(self, tOffset: Union[int, float], bInPlace: bool = False):
        """
        delay - Return a copy of self that is delayed by an offset.
                For delaying self, use ".vtTimeTrace += ..." instead.

        :param tOffset:     float   Time offset
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: New TimeSeries, delayed
        """
        if not bInPlace:
            tsObj = self.copy()
        else:
            tsObj = self

        # - Store previous tStart and tStop
        tStartOld = tsObj.tStart
        tStopOld = tsObj.tStop

        # - Shift time trace
        tsObj.vtTimeTrace += tOffset
        # - Shift tStart and tStop
        tsObj.tStart = tStartOld + tOffset
        tsObj.tStop = tStopOld + tOffset

        return tsObj

    def plot(self, vtTimes: Union[int, float, ArrayLike] = None, **kwargs):
        """
        plot - Visualise a time series on a line plot

        :param vtTimes: Optional. Time base on which to plot. Default: time base of time series
        :param kwargs:  Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """
        if vtTimes is None:
            vtTimes = self.vtTimeTrace

        # - Get current plotting backend
        _bUseHoloviews, _bUseMatplotlib = GetPlottingBackend()

        if _bUseHoloviews:
            mfData = np.atleast_2d(self(vtTimes)).reshape((np.size(vtTimes), -1))
            if kwargs == {}:
                vhCurves = [
                    hv.Curve((vtTimes, vfData)).redim(x="Time") for vfData in mfData.T
                ]
            else:
                vhCurves = [
                    hv.Curve((vtTimes, vfData)).redim(x="Time").options(**kwargs)
                    for vfData in mfData.T
                ]

            if len(vhCurves) > 1:
                return hv.Overlay(vhCurves).relabel(group=self.strName)
            else:
                return vhCurves[0].relabel(self.strName)

        elif _bUseMatplotlib:
            return plt.plot(vtTimes, self(vtTimes), label=self.strName, **kwargs)

        else:
            warn("No plotting back-end detected.")

    def contains(self, vtTimeTrace: Union[int, float, ArrayLike]):
        """
        contains - Does the time series contain all points in the specified time trace?

        :param vtTimeTrace: Array-like containing time points
        :return:            boolean: All time points are contained within this time series
        """
        return (
            True
            if self.tStart <= np.min(vtTimeTrace) and self.tStop >= np.max(vtTimeTrace)
            else False
        )

    def resample(self, vtTimes: Union[int, float, ArrayLike], bInPlace: bool = False):
        """
        resample - Return a new time series sampled to the supplied time base

        :param vtTimes:     Array-like of T desired time points to resample
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:            TimeSeries object, resampled to new time base
        """
        if not bInPlace:
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
            bInPlace: bool = False,
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
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
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
        return self.resample(vtSampleTimes, bInPlace=bInPlace)

    def merge(self, tsOther, bRemoveDuplicates=True, bInPlace: bool = False):
        """
        merge - Merge another time series to this one, in time. Maintain
                each time series' time values.
        :param tsOther:             TimeSeries that is merged to self
        :param bRemoveDuplicates:   bool - If true, time points in tsOther.vtTimeTrace
                                           that are also in self.vtTimeTrace are
                                           discarded. Otherwise they are included in
                                           the new time trace and come after the
                                           corresponding points of self.vtTimeTrace.
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:         The merged time series
        """

        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), "`tsOther` must be a TimeSeries object."

        assert tsOther.nNumTraces == self.nNumTraces, (
                "`tsOther` must include the same number of traces ("
                + str(self.nNumTraces)
                + ")."
        )

        # - If the other TimeSeries is empty, just return
        if tsOther.isempty():
            if bInPlace:
                return self
            else:
                return self.copy()

        # - If bRemoveDuplicates==True and time ranges overlap,  find and remove
        #   time points of tsOther that are also included in self (assuming both
        #   TimeSeries have a sorted vTimeTrace)
        if bRemoveDuplicates and not (
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
        if not bInPlace:
            tsMerged = self.copy()
        else:
            tsMerged = self

        tsMerged._vtTimeTrace = vtTimeTraceNew[viSorted]
        tsMerged._mfSamples = mfSamplesNew[viSorted]
        tsMerged._tStart = tStart
        tsMerged._tStop = tStop

        # - Determine number of channels
        tsMerged._nNumChannels = max(tsOther.nNumChannels, self.nNumChannels)

        # - Create new interpolator
        tsMerged._create_interpolator()

        # - Return merged TS
        return tsMerged

    def append(self, tsOther, bInPlace: bool = False):
        """
        append() - Combine another time series into this one, along samples axis

        :param tsOther: Another time series. Will be resampled to the time base of the called series object
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:        Current time series,
        """
        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), "`tsOther` must be a TimeSeries object."

        # - Resample tsOther to own time base
        mfOtherSamples = tsOther(self.vtTimeTrace)

        # - Create a new time series, or modify this time series
        if not bInPlace:
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

    def concatenate(self, tsOther, bInPlace: bool = False):
        """
        concatenate() - Combine another time series with this one, along samples axis

        :param tsOther: Another time series. Will be resampled to the time base of the called series object
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: Time series, with series from both source and other time series
        """
        # - Create a new time series, or modify this time series
        return self.append(tsOther, bInPlace=bInPlace)

    def append_t(self, tsOther, bInPlace: bool = False):
        """
        append_t() - Append another time series to this one, in time

        :param tsOther: Another time series. Will be tacked on to the end of the called series object
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: Time series containing current data, with other TS appended in time
        """

        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), "`tsOther` must be a TimeSeries object."

        assert tsOther.nNumTraces == self.nNumTraces, (
                "`tsOther` must include the same number of traces ("
                + str(self.nNumTraces)
                + ")."
        )

        # - Create a new time series, or modify this time series
        if not bInPlace:
            tsAppended = self.copy()
        else:
            tsAppended = self

        # - Concatenate time trace and samples
        tsAppended._mfSamples = np.concatenate(
            (tsAppended.mfSamples, tsOther.mfSamples), axis=0
        )

        tMedianDT = np.median(np.diff(self._vtTimeTrace))
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

    def concatenate_t(self, tsOther, bInPlace: bool = False):
        """
        concatenate_t() - Join together two time series in time

        :param tsOther: Another time series. Will be tacked on to the end of the called series object
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: New concatenated time series
        """
        return self.append_t(tsOther, bInPlace=bInPlace)

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
            self, bFull: bool = False, nFirst: int = 4, nLast: int = 4, nShorten: int = 10
    ):
        """
        print - Print an overview of the time series and its values.

        :param bFull:     Boolean - Print all samples of self, no matter how long it is
        :param nShorten:  Integer - Print shortened version of self if it comprises more
                          than nShorten time points and bFull is False
        :param nFirst:    Integer - Shortened version of printout contains samples at first
                          nFirst points in self.vtTimeTrace
        :param nLast:     Integer - Shortened version of printout contains samples at last
                          nLast points in self.vtTimeTrace
        """

        s = "\n"
        if len(self.vtTimeTrace) <= 10 or bFull:
            strSummary = s.join(
                [
                    "{}: \t {}".format(t, vSamples)
                    for t, vSamples in zip(self.vtTimeTrace, self.mfSamples)
                ]
            )
        else:
            strSummary0 = s.join(
                [
                    "{}: \t {}".format(t, vSamples)
                    for t, vSamples in zip(
                    self.vtTimeTrace[:nFirst], self.mfSamples[:nFirst]
                )
                ]
            )
            strSummary1 = s.join(
                [
                    "{}: \t {}".format(t, vSamples)
                    for t, vSamples in zip(
                    self.vtTimeTrace[-nLast:], self.mfSamples[-nLast:]
                )
                ]
            )
            strSummary = strSummary0 + "\n\t...\n" + strSummary1
        print(self.__repr__() + "\n" + strSummary)

    def clip(self, vtNewBounds: ArrayLike, bInPlace: bool = False):
        """
        clip - Clip a TimeSeries to data only within a new set of time bounds (exclusive end points)

        :param vtNewBounds: ArrayLike   [tStart tStop] defining new bounds
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)

        :return: tsClip, vbIncludeSamples)
                tsClip:             New TimeSeries clipped to bounds
                vbIncludeSamples:   boolean ndarray indicating which original samples are included
        """
        # - Create a new time series, or modify this time series
        if not bInPlace:
            tsClipped = self.copy()
        else:
            tsClipped = self

        # - Get first sample
        vfFirstSample = np.atleast_1d(tsClipped(vtNewBounds[0]))

        # - Get number of traces
        nNumTraces = tsClipped.nNumTraces

        # - For periodic time series, resample the series
        if tsClipped.bPeriodic:
            tsClipped, _ = tsClipped._clip_periodic(vtNewBounds, bInPlace=bInPlace)
        else:
            tsClipped, _ = tsClipped._clip(vtNewBounds, bInPlace=bInPlace)

        # - Insert initial time point
        tsClipped._vtTimeTrace = np.concatenate(
            ([vtNewBounds[0]], tsClipped._vtTimeTrace)
        )

        # - Insert initial samples

        tsClipped._mfSamples = np.concatenate(
            (
                np.reshape(vfFirstSample, (-1, nNumTraces)),
                np.reshape(tsClipped._mfSamples, (-1, nNumTraces)),
            ),
            axis=0,
        )

        # - Update tStart to contain initial time point
        tsClipped._tStart = min(tsClipped._tStart, vtNewBounds[0])

        return tsClipped

    def _clip(self, vtNewBounds: ArrayLike, bInPlace: bool = False):
        """
        clip - Clip a TimeSeries to data only within a new set of time bounds (exclusive end points)

        :param vtNewBounds: ArrayLike   [tStart tStop] defining new bounds
        :param bInPlace:    bool        Conduct operation in-place (Default: False; create a copy)
        :return:  TimeSeries clipped to bounds
        """
        # - Create a new time series, or use self
        if not bInPlace:
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

    def _clip_periodic(self, vtNewBounds: ArrayLike, bInPlace: bool = False):
        """
        _clip_periodic - Clip a periodic TimeSeries
        :param vtNewBounds: ArrayLike   [tStart tStop] defining new bounds
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:
        """
        # - Ensure time bounds are sorted
        vtNewBounds = np.sort(vtNewBounds)
        tDuration = np.diff(vtNewBounds)

        # - Create a new time series, or use self
        if not bInPlace:
            tsClipped = self.copy()
        else:
            tsClipped = self

        # - Catch sinlgeton time point
        if vtNewBounds[0] == vtNewBounds[1]:
            return tsClipped.resample(vtNewBounds[0], bInPlace=bInPlace)

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
        tsClip = tsClipped.resample(vtNewTimeTrace, bInPlace=bInPlace)
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
        # - Get nan mask
        mbIsNan = np.isnan(self.mfSamples)

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
        assert np.size(vtNewTrace) == np.size(
            self._vtTimeTrace
        ), "New time trace must have the same number of elements as the original trace."

        # - Store new time trace
        self._vtTimeTrace = np.reshape(vtNewTrace, -1)

        if np.size(self._vtTimeTrace) > 0:
            # - Fix tStart and tStop
            self._tStart = min(self._tStart, vtNewTrace[0])
            self._tStop = max(self._tStop, vtNewTrace[-1])

        # - Create a new interpolator
        self._create_interpolator()

    def choose(self, vnTraces: Union[int, ArrayLike], bInPlace: bool = False):
        """
        choose() - Select from one of several sub-traces; return a TimeSeries containing these traces

        :param vnTraces:    array-like of indices within source TimeSeries
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return:            TimeSeries containing only the selected traces
        """
        # - Convert to a numpy array and check extents
        vnTraces = np.atleast_1d(vnTraces)
        assert (
                min(vnTraces) >= 0 and max(vnTraces) <= self.nNumTraces
        ), "`vnTraces` must be between 0 and " + str(self.nNumTraces)

        if not bInPlace:
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
        self._mfSamples = mfSamples.astype("float")
        self.strInterpKind = strInterpKind
        # - Interpolator for samples
        self._create_interpolator()

    def find(self, ttTimes: ArrayLike):
        vtTime = list(ttTimes)
        mfSamples = self(vtTime)
        return vtTime, mfSamples

    def save(self, strPath):
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

        return np.reshape(self.oInterp(vtTimes), (-1, self.nNumTraces))

    def __call__(self, vtTimes: Union[int, float, ArrayLike]):
        """
        ts(tTime1, tTime2, ...) - Interpolate the time series to the provided time points

        :param tTime: Scalar, list or np.array of T desired interpolated time points
        :return:      np.array of interpolated values. Will have the shape TxN
        """
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

    @property
    def mfSamples(self):
        return np.reshape(self._mfSamples, (np.size(self.vtTimeTrace), -1))

    @mfSamples.setter
    def mfSamples(self, mfNewSamples: ArrayLike):
        # - Promote to 1d
        mfNewSamples = np.atleast_1d(mfNewSamples)

        # - Permit a one-dimensional sample input, promote to 2d
        if (mfNewSamples.shape[0] == 1) and (np.size(self.vtTimeTrace) > 1):
            mfNewSamples = np.reshape(mfNewSamples, (np.size(self.vtTimeTrace), 1))

        # - Check samples for correct size
        assert mfNewSamples.shape[0] == np.size(
            self.vtTimeTrace
        ), "New samples matrix must have the same number of samples as `.vtTimeTrace`."

        # - Store new time trace
        self._mfSamples = mfNewSamples

        # - Create a new interpolator
        self._create_interpolator()

    @property
    def nNumTraces(self):
        """
        .nNumTraces: int Number of traces in this TimeSeries object
        """
        try:
            return self.mfSamples.shape[1]
        # If mfSamples is 1d:
        except IndexError:
            return 1

    @property
    def nNumChannels(self):
        if hasattr(self, "_nNumChannels") and self._nNumChannels is not None:
            return self._nNumChannels
        else:
            return self.nNumTraces

    @nNumChannels.setter
    def nNumChannels(self, nNewNumChannels):
        if self.nNumTraces == 0:
            if nNewNumChannels < 0:
                raise ValueError(f"TSContinuous `{self.strName}`: `nNumChannels` cannot be negative.")
        elif nNewNumChannels < self.nNumTraces:
            raise ValueError(
                f"TSContinuous `{self.strName}`: `nNumChannels` must be at least {self.nNumTraces}."
            )
        else:
            self._nNumChannels = nNewNumChannels


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

        # - Handle empty TSEvent
        if len(self.vtTimeTrace) == 0:
            return np.array([]), np.array([], int)

        if vtTimeBounds is None:
            # - Return all samples
            return self.vtTimeTrace, self.vnChannels
        else:
            vtTimeBounds = np.array(vtTimeBounds)
            # - Map None to time trace bounds
            if vtTimeBounds[0] is None:
                vtTimeBounds[0] = self.vtTimeTrace[0]
            if vtTimeBounds[-1] is None:
                vtTimeBounds[-1] = self.vtTimeTrace[-1]
                bIncludeFinal = True
            else:
                bIncludeFinal = False

            # - Permit unsorted bounds
            vtTimeBounds = np.sort(vtTimeBounds)

            # - Find matching times
            vbMatchingTimes = np.logical_and(
                self.vtTimeTrace >= vtTimeBounds[0],
                np.logical_or(self.vtTimeTrace < vtTimeBounds[-1], bIncludeFinal),
            )

            # - Return matching samples
            return self.vtTimeTrace[vbMatchingTimes], self.vnChannels[vbMatchingTimes]

    def clip(self, vtNewBounds: ArrayLike, bInPlace: bool = False) -> TSEventType:
        """
        clip - Clip a time series in time, to a specified start and stop time

        :param vtNewBounds:     Array-like  Bounds between object is clipped
        :param bInPlace:        bool Specify whether operation should be performed in place (Default: False)
        :return:                TSEvent Including only clipped events
        """
        # - Call super-class clipper
        tsClip, vbIncludeSamples = super()._clip(vtNewBounds, bInPlace=bInPlace)

        # - Fix up channels variable
        tsClip._vnChannels = self._vnChannels[vbIncludeSamples]

        # - Return new TimeSeries
        return tsClip

    def _choose(self, vnSelectChannels: ArrayLike) -> (np.ndarray, np.ndarray):
        """
        _choose - Select and return raw event data for the requested channels

        :param vnSelectChannels: array-like of channel indices
        :return: (vtTimeTrace, vnChannels) containing events form the requested channels
        """

        # - Check vnSelectChannels
        if not (
                np.min(vnSelectChannels) >= 0
                and np.max(vnSelectChannels) < self.nNumChannels
        ):
            raise IndexError(
                f"TSEvent `{self.strName}`: `vnSelectChannels` must be between 0 and {self.nNumChannels}."
            )

        # - Make sure elements in vnSelectChannels are unique for better performance
        vnSelectChannels = np.unique(vnSelectChannels)

        # - Find samples to return
        vbIncludeSamples = np.isin(self._vnChannels, vnSelectChannels)

        # - Return events for those samples
        return self._vtTimeTrace[vbIncludeSamples], self._vnChannels[vbIncludeSamples]

    def choose(self, vnSelectChannels: ArrayLike, bInPlace: bool = False) -> TSEventType:
        """
        choose - Select and return only the requested channels

        :param vnSelectChannels: array-like of channel indices
        :param bInPlace:         bool Specify whether operation should be performed in place (Default: False)
        :return: TSEvent containing events from the requested channels
        """

        # - Build new TS with only those samples from requested channels
        if not bInPlace:
            tsChosen = self.copy()
        else:
            tsChosen = self

        tsChosen._vtTimeTrace, tsChosen._vnChannels = tsChosen._choose(vnSelectChannels)

        return tsChosen

    def choose_times(self, vnSelectChannels: ArrayLike) -> np.ndarray:
        """
        choose_times - like choose but only return vtTimeTrace instead of time series

        :param vnSelectChannels: array-like of channel indices
        :return: np.ndarray with the time values corresponding to the given channel indices
        """

        # - Use `_choose` to return time trace
        vtTimes, __ = self._choose(vnSelectChannels)
        return vtTimes

    def plot(self, vtTimes: Optional[ArrayLike] = None, *args, **kwargs):
        """
        plot - Visualise a time series on plot

        :param vtTimes:         Optional. Time base on which to plot. Default: time base of time series
        :param args, kwargs:    Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """

        # - Get current plotting backend
        _bUseHoloviews, _bUseMatplotlib = GetPlottingBackend()

        # - Get samples
        vtTimes, vnChannels = self.find(vtTimes)

        if _bUseHoloviews:
            return (
                hv.Scatter((vtTimes, vnChannels), *args, **kwargs)
                    .redim(x="Time", y="Channel")
                    .relabel(self.strName)
            )

        elif _bUseMatplotlib:
            return plt.scatter(vtTimes, vnChannels, *args, **kwargs)

        else:
            warn(f"TSEvent `{self.strName}`: No plotting back-end detected.")

    def resample(self, vtTimes: np.ndarray, bInPlace: bool = False) -> TSEventType:
        return self.clip(vtTimes, bInPlace=bInPlace)

    def resample_within(
            self,
            tStart: Optional[float] = None,
            tStop: Optional[float] = None,
            tDt: Optional[float] = None,
            bInPlace: bool = False,
    ) -> TSEventType:
        """
        resample_within - Resample a TimeSeries, within bounds

        :param tStart:      float Start time (inclusive)
        :param tStop:       float Stop time (inclusive)
        :param tDt:         Unused for event TimeSeries
        :param bInPlace:    bool Specify whether operation should be performed in place (Default: False)

        :return:        New TimeSeries containing events within [tStart, tStop]
        """
        if tStart is None:
            tStart = self.tStart

        if tStop is None:
            tStop = self.tStop + np.finfo(float).eps

        return self.clip([tStart, tStop], bInPlace=bInPlace)

    def append_t(self, tsOther: TSEventType, bInPlace: bool = False) -> TSEventType:
        """
        append_t() - Append another time series to this one, in time

        :param tsOther: Another time series. Will be tacked on to the end of the called series object
        :param bInPlace:    bool    Conduct operation in-place (Default: False; create a copy)
        :return: TSEvent containing current data, with other TS appended in time
        """

        # - Check tsOther
        if not isinstance(tsOther, TSEvent):
            raise TypeError(f"TSEvent `{self.strName}`: `tsOther` must be a TSEvent object.")

        if tsOther.nNumTraces != self.nNumTraces:
            raise ValueError(
                f"TSEvent `{self.strName}`: `tsOther` must include the same number of"
                + f"traces ({self.nNumTraces}))."
            )

        # - Create a new time series, or modify this time series
        if not bInPlace:
            tsAppended = self.copy()
        else:
            tsAppended = self

        # - Concatenate time trace and channels
        tsAppended._vnChannels = np.concatenate(
            (tsAppended.vnChannels, tsOther.vnChannels), axis=0
        )
        tsAppended._vtTimeTrace = np.concatenate(
            (
                tsAppended._vtTimeTrace,
                tsOther.vtTimeTrace + tsAppended.tStop - tsOther.tStart,
            ),
            axis=0,
        )

        # - Fix tStop
        tsAppended._tStop += tsOther.tDuration

        # - Return appended time series
        return tsAppended

    def merge(
            self,
            ltsOther: Union[TimeSeries, List[TimeSeries]],
            bRemoveDuplicates: Optional[bool] = None,
            bInPlace: bool = False,
    ) -> TSEventType:
        """
        merge - Merge another TSEvent into this one
        :param ltsOther:            TimeSeries (or list of TimeSeries) to merge into this one
        :param bRemoveDuplicates:   bool Remove duplicate events (unused in TSEvent)
        :param bInPlace:            bool Specify whether operation should be performed in place (Default: False)
        :return: self with new samples included
        """

        if bInPlace:
            tsMerged = self
        else:
            tsMerged = self.copy()

        # - Ensure we have a list of objects to work on
        if not isinstance(ltsOther, collections.abc.Iterable):
            ltsOther = [tsMerged, ltsOther]
        else:
            ltsOther = [tsMerged] + list(ltsOther)

        # - Determine number of channels
        tsMerged._nNumChannels = np.amax([tsOther.nNumChannels for tsOther in ltsOther])

        # - Check tsOther class
        assert all(
            map(lambda tsOther: isinstance(tsOther, TSEvent), ltsOther)
        ), f"TSEvent `{self.strName}`: `tsOther` must be a `TSEvent` object."

        # - Filter out empty series
        ltsOther = list(filter(lambda ts: not ts.isempty(), ltsOther))

        # - Stop if no non-empty series is left
        if not ltsOther:
            return tsMerged

        # - Merge all samples
        vtNewTimeBase = np.concatenate([tsOther.vtTimeTrace for tsOther in ltsOther])
        vnNewChannels = np.concatenate([tsOther.vnChannels for tsOther in ltsOther])
        tNewStart = min(tsOther.tStart for tsOther in ltsOther)
        tNewStop = max(tsOther.tStop for tsOther in ltsOther)

        # - Sort on time and merge
        vnOrder = np.argsort(vtNewTimeBase)
        tsMerged._vtTimeTrace = vtNewTimeBase[vnOrder]
        tsMerged._vnChannels = vnNewChannels[vnOrder]
        tsMerged._tStart = tNewStart
        tsMerged._tStop = tNewStop

        return tsMerged

    def _compatibleShape(self, other: Union[ArrayLike, int, float]) -> np.ndarray:
        try:
            if np.size(other) == 1:
                return copy.copy(np.broadcast_to(other, self.vtTimeTrace.shape))

            elif other.shape[0] == self.vtTimeTrace.shape[0]:
                if len(other.shape) > 1 and other.shape[1] == 1:
                    return other.flatten()
                else:
                    return np.reshape(other, self.vtTimeTrace.shape)

        except:
            raise ValueError(
                f"TSEvent `{self.strName}`: Input data must have shape {self.vtTimeTrace.shape}."
            )

    def raster(
            self,
            tDt: float,
            tStart: float = None,
            tStop: float = None,
            nNumTimeSteps: int = None,
            vnSelectChannels: np.ndarray = None,
            bAddEvents: bool = False,
    ) -> (np.ndarray, np.ndarray, np.ndarray):

        """
        raster - Return rasterized time series data, where each data point
                 represents a time step. Events are represented in a boolean
                 matrix, where the first axis corresponds to time, the second
                 axis to the channel.
                 Events that happen between time steps are projected to the
                 preceding one. If two events happen during one time step
                 within a single channel, they are counted as one, unless
                 bAddEvents is True.
        :param tDt:     float Length of single time step in raster
        :param tStart:  float Time where to start raster - Will start
                              at self.vtTImeTrace[0] if None
        :param tStop:   float Time where to stop raster. This time point is
                              not included anymore. - If None, will use all
                              points until (and including) self.vtTImeTrace[-1]
                              (unless nTimeSteps is set)
        :param nNumTimeSteps: int Can be used to determine number of time steps
                                  directly, instead of providing tStop
        :param vnSelectedChannels: Array-like Channels, from which data is to be used.
        :param bAddEvents:    bool If True, return integer raster containing number of
                              events for each time step and channel

        :return
            vtTimeBase:     Time base of rasterized data
            vnSelectChannels Channel ids corresponding to columns in mEventsRaster
            mEventsRaster   Boolean matrix with True indicating presence of events
                            for each time step and channel. If bAddEvents == True,
                            the raster consists of integers, indicating the number
                            of events per time step and channel.
                            First axis corresponds to time, second axis to channel.
        """
        # - Get data from selected channels and time range
        if vnSelectChannels is not None:
            tsSelected = self.choose(vnSelectChannels)
        else:
            tsSelected = self.copy()
            vnSelectChannels = np.arange(self.nNumChannels)

        # - Generate time base
        if self.tStart is None and tStart is None:
            raise ValueError(
                f"TSEvent `{self.strName}`: Cannot determine tStart. Provide as argument."
            )
        if self.tStop is None and tStop is None and nNumTimeSteps is None:
            raise ValueError(
                f"TSEvent `{self.strName}`: Cannot determine tStop or nNumTimeSteps. "
                + f"Provide one of them as argument."
            )
        tStartBase = self.tStart if tStart is None else tStart
        if nNumTimeSteps is None:
            tStopBase = self.tStop if tStop is None else tStop
            nNumTimeSteps = int(np.ceil((tStopBase - tStartBase) / tDt))
            # - Make sure tStopBase is multiple of nNumTimeSteps
            tStopBase = tStartBase + nNumTimeSteps * tDt
        else:
            tStopBase = tStartBase + nNumTimeSteps * tDt
        vtTimeBase = np.arange(nNumTimeSteps) * tDt + tStartBase

        # - Raster for storing event data
        dtypeRaster = int if bAddEvents else bool
        mEventsRaster = np.zeros((nNumTimeSteps, len(vnSelectChannels)), dtypeRaster)

        # - Handle empty series
        if len(self.vtTimeTrace) == 0:
            return vtTimeBase, vnSelectChannels, mEventsRaster

        # - Select data according to time base
        vbChoose = (tsSelected.vtTimeTrace >= tStartBase) & (
                tsSelected.vtTimeTrace < tStopBase
        )
        vtEventTimes = tsSelected.vtTimeTrace[vbChoose]
        vnEventChannels = tsSelected.vnChannels[vbChoose]

        ## -- Convert input events and samples to boolen or integer raster
        # - Only consider rasters that have non-zero length
        if nNumTimeSteps > 0:
            # Compute indices for times
            viTimeIndices_Raster = np.floor((vtEventTimes - tStartBase) / tDt).astype(
                int
            )
            if bAddEvents:
                # Count events per time step and channel
                for iTime, iChannel in zip(viTimeIndices_Raster, vnEventChannels):
                    mEventsRaster[iTime, iChannel] += 1
            else:
                # - Print a warning if there are multiple spikes in one time step and channel
                if (
                        (
                                np.diff(np.c_[viTimeIndices_Raster, vnEventChannels], axis=0)
                                == np.zeros(2)
                        )
                                .all(axis=1)
                                .any(axis=0)
                ):
                    print(
                        "TSEvent `{}`: There are channels with multiple events".format(
                            self.strName
                        )
                        + " per time step. Consider smaller tDt or setting bAddEvents True."
                    )
                # Mark spiking indices with True
                mEventsRaster[viTimeIndices_Raster, vnEventChannels] = True

        return vtTimeBase, np.array(vnSelectChannels), mEventsRaster

    def xraster(
            self,
            tDt: float,
            tStart: Optional[float] = None,
            tStop: Optional[float] = None,
            nNumTimeSteps: int = None,
    ) -> np.ndarray:

        """
        xraster - Yields a rasterized time series data, where each data point
                 represents a time step. Events are represented in a boolean
                 matrix, where the first axis corresponds to time, the second
                 axis to the channel.
                 Events that happen between time steps are projected to the
                 preceding one. If two events happen during one time step
                 within a single channel, they are counted as one.
        :param tDt:     float Length of single time step in raster
        :param tStart:  float Time where to start raster - Will start
                              at self.vtTImeTrace[0] if None
        :param tStop:   float Time where to stop raster. This time point is
                              not included anymore. - If None, will use all
                              points until (and including) self.vtTImeTrace[-1]
                              (unless nTimeSteps is set)
        :param nBatchSize: int      Process one nBatchSize time steps at a time.
                              This parameter will determine the speed vs latency for this process
        :param nNumTimeSteps: int Can be used to determine number of time steps
                                  directly, instead of providing tStop

        :yields
            vbEventsRaster  Boolean matrix with True indicating event axis corresponds to channel
        """
        tsSelected = self
        vnSelectChannels = np.arange(self.nNumChannels)

        # - Generate time base
        if self.tStart is None and tStart is None:
            raise ValueError(
                f"TSEvent `{self.strName}`: Cannot determine tStart. Provide as argument."
            )
        if self.tStop is None and tStop is None and nNumTimeSteps is None:
            raise ValueError(
                f"TSEvent `{self.strName}`: Cannot determine tStop or nNumTimeSteps. "
                + f"Provide one of them as argument."
            )
        tStartBase = self.tStart if tStart is None else tStart
        tStopBase = self.tStop + tDt if tStop is None else tStop
        nNumTimeSteps = (
            int(np.floor((tStopBase - tStartBase) / tDt))
            if nNumTimeSteps is None
            else nNumTimeSteps
        )

        vtEventTimes, vnEventChannels = tsSelected.find([tStartBase, tStopBase])

        # - Convert input events and samples to boolen raster

        mbEventsRaster = np.zeros((nNumTimeSteps, len(vnSelectChannels)), bool)

        # - Only consider rasters that have non-zero length
        if nNumTimeSteps > 0:
            # Compute indices for times
            viTimeIndices_Raster = ((vtEventTimes - tStartBase) / tDt).astype(int)
            viRowIndices_Raster = vnEventChannels
            # Mark spiking indices with True
            mbEventsRaster[viTimeIndices_Raster, viRowIndices_Raster] = True

        yield from mbEventsRaster  # Yield one row at a time

    def print(
            self, bFull: bool = False, nFirst: int = 4, nLast: int = 4, nShorten: int = 10
    ):
        """
        print - Print an overview of the time series and its values.

        :param bFull:     Boolean - Print all samples of self, no matter how long it is
        :param nShorten:  Integer - Print shortened version of self if it comprises more
                          than nShorten time points and bFull is False
        :param nFirst:    Integer - Shortened version of printout contains samples at first
                          nFirst points in self.vtTimeTrace
        :param nLast:     Integer - Shortened version of printout contains samples at last
                          nLast points in self.vtTimeTrace
        """

        s = "\n"
        if len(self.vtTimeTrace) <= 10 or bFull:
            strSummary = s.join(
                [
                    f"{t}: \t {nChannel}"
                    for t, nChannel in zip(self.vtTimeTrace, self.vnChannels)
                ]
            )
        else:
            strSummary0 = s.join(
                [
                    f"{t}: \t {nChannel}"
                    for t, nChannel, fSample in zip(self.vtTimeTrace[:nFirst], self.vnChannels[:nFirst])
                ]
            )
            strSummary1 = s.join(
                [
                    f"{t}: \t {nChannel}"
                    for t, nChannel in zip(self.vtTimeTrace[-nLast:], self.vnChannels[-nLast:])
                ]
            )
            strSummary = strSummary0 + "\n\t...\n" + strSummary1
        print(self.__repr__() + "\nTime \t Ch.-ID" + "\n" + strSummary)

    def save(self, strPath: str):
        """
        save - Save TSEvent as npz file using np.savez
        :param strPath:     str  Path to save file
        """
        np.savez(
            strPath,
            vtTimeTrace=self.vtTimeTrace,
            vnChannels=self.vnChannels,
            tStart=self.tStart,
            tStop=self.tStop,
            bPeriodic=self.bPeriodic,
            nNumChannels=self.nNumChannels,
            strName=self.strName,
            strType="TSEvent",  # Indicate that the object is TSEvent
        )
        bPathMissesEnding = strPath.split(".")[-1] != "npz"  # np.savez will add ending
        print(
            "TSEvent `{}` has been stored in `{}`.".format(
                self.strName, strPath + bPathMissesEnding * ".npz"
            )
        )

    def __repr__(self):
        """
        __repr__() - Return a string representation of this object
        :return: str String description
        """
        if self.isempty():
            return f"Empty TSEvent object `{self.strName}`"
        else:
            return "{}periodic TSEvent object `{}` from t={} to {}. Channels: {}. Events: {}".format(
                int(not self.bPeriodic) * "non-",
                self.__class__.__name__,
                self.strName,
                self.tStart,
                self.tStop,
                self.vnChannels,
                self.vtTimeTrace.size
            )

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
