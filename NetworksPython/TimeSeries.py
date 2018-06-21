###
# TimeSeries.py - Classes to manage time series
###

import numpy as np
import scipy.interpolate as spint
from warnings import warn
import copy
from typing import Union, List
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
    TimeSeries - Class represent a multi-series time series, with temporal interpolation and periodicity supported
    ts = TimeSeries(vtTimeTrace, mfSamples <, strInterpKind, bPeriodic>)

    ts[tInterpTime]:
    ts(tInterpTime):
    ts.interpolate(tInterpTime): Interpolate to a time point `tInterpTime`

    ts.oInterp:         scipy.interpolate.interp1d object, interpolator
    """

    def __init__(
        self,
        vtTimeTrace: np.ndarray,
        mfSamples: np.ndarray,
        strInterpKind: str = "linear",
        bPeriodic: bool = False,
        strName=None,
    ):
        """
        TimeSeries - Class represent a multi-series time series, with temporal interpolation and periodicity supported

        :param vtTimeTrace:     [Tx1] vector of time samples
        :param mfSamples:       [TxM] matrix of values corresponding to each time sample
        :param strInterpKind:   str: Specify the interpolation type. Default: 'linear'
        :param bPeriodic:       bool: Treat the time series as periodic around the end points. Default: False

        If the time series is not periodic (the default), then NaNs will be returned for any extrapolated values.
        """

        # - Convert everything to numpy arrays
        vtTimeTrace = np.atleast_1d(np.asarray(vtTimeTrace)).astype(float)
        mfSamples = np.atleast_1d(mfSamples).astype(float)

        # - Permit a one-dimensional sample input
        if (mfSamples.shape[0] == 1) and (np.size(vtTimeTrace) > 1):
            mfSamples = np.transpose(mfSamples)

        # - Check arguments
        assert (
            np.size(vtTimeTrace) == mfSamples.shape[0]
        ), "The number of time samples must be equal to the first " "dimension of `mfSamples`"
        assert np.all(
            np.diff(vtTimeTrace) >= 0
        ), "The time trace must be sorted and not decreasing"

        # - Assign attributes
        self._vtTimeTrace = vtTimeTrace
        self._mfSamples = mfSamples.astype("float")
        self.strInterpKind = strInterpKind
        self.bPeriodic = bPeriodic
        self.strName = strName

        self._bUseHoloviews, self._bUseMatplotlib = GetPlottingBackend()

        if bPeriodic:
            self._tDuration = vtTimeTrace[-1] - vtTimeTrace[0]
            self._tStart = vtTimeTrace[0]

        self._create_interpolator()

    def __getitem__(self, vtTimes):
        """
        ts[tTime1, tTime2, ...] - Interpolate the time series to the provided time points
        NOTE that ts[:] uses as (fixed) step size the mean step size of self.vtTimeTrace
        and thus can return different values than those in ts.mfSamples!
        :param vtTimes: Slice, scalar, list or np.array of T desired interpolated time points
        :return:      np.array of interpolated values. Will have the shape TxN
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

    def __call__(self, vtTimes):
        """
        ts(tTime1, tTime2, ...) - Interpolate the time series to the provided time points

        :param tTime: Scalar, list or np.array of T desired interpolated time points
        :return:      np.array of interpolated values. Will have the shape TxN
        """
        return np.reshape(self.interpolate(vtTimes), (-1, self.nNumTraces))

    def interpolate(self, vtTimes: np.ndarray):
        """
        interpolate - Interpolate the time series to the provided time points

        :param vtTimes: np.ndarray of T desired interpolated time points
        :return:        np.ndarray of interpolated values. Will have the shape TxN
        """
        # - Enforce periodicity
        if self.bPeriodic:
            vtTimes = (
                np.asarray(vtTimes) - self._tStart
            ) % self._tDuration + self._tStart

        return self.oInterp(vtTimes)

    def delay(self, tOffset):
        """
        delay - Return a copy of self that is delayed by an offset. 
                For delaying self, use ".vtTimeTrace += ..." instead.
        :param tOffset: float Time offset
        :return: New TimeSeries, delayed
        """
        tsCopy = self.copy()
        tsCopy.vtTimeTrace += tOffset
        return tsCopy

    def plot(self, vtTimes: np.ndarray = None, **kwargs):
        """
        plot - Visualise a time series on a line plot

        :param vtTimes: Optional. Time base on which to plot. Default: time base of time series
        :param kwargs:  Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """
        if vtTimes is None:
            vtTimes = self.vtTimeTrace

        if self._bUseHoloviews:
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

        elif self._bUseMatplotlib:
            return plt.plot(vtTimes, self(vtTimes), **kwargs)

        else:
            warn("No plotting back-end detected.")

    def contains(self, vtTimeTrace: np.ndarray):
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

    def resample(self, vtTimes: np.ndarray):
        """
        resample - Return a new time series sampled to the supplied time base

        :param vtTimes: Array-like of T desired time points to resample
        :return:        New TimeSeries object, resampled to new time base
        """

        # - Return a new time series
        tsResampled = self.copy()
        tsResampled._vtTimeTrace = vtTimes
        tsResampled._mfSamples = self(vtTimes)
        tsResampled.bPeriodic = False
        tsResampled._create_interpolator()
        return tsResampled

    def resample_within(
        self, tStart: float = None, tStop: float = None, tDt: float = None
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
        :return:        New TimeSeries object, resampled according to parameters
        """
        tStart = (
            min(self.vtTimeTrace)
            if tStart is None
            else max(tStart, min(self.vtTimeTrace))
        )
        tStop = (
            max(self.vtTimeTrace)
            if tStop is None
            else min(tStop, max(self.vtTimeTrace))
        )
        tDt = np.mean(np.diff(self.vtTimeTrace)) if tDt is None else tDt

        vtSampleTimes = np.arange(tStart, tStop + tDt, tDt)
        vtSampleTimes = vtSampleTimes[vtSampleTimes <= tStop]

        # - Return a new time series
        return self.resample(vtSampleTimes)

    def merge(self, tsOther, bRemoveDuplicates=True):
        """
        merge - Merge another time series to this one, in time. Maintain
                each time series' time values.
        :param tsOther:             TimeSeries that is merged to self
        :param bRemoveDuplicates:   bool - If true, time points in tsOther.vtTimeTrace
                                           that are also in self.vtTimeTrace are
                                           discarded. Otherwise they are included in
                                           the new time trace and come after the
                                           corresponding points of self.vtTimeTrace.
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
            return self

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
        self._vtTimeTrace = vtTimeTraceNew[viSorted]
        self.mfSamples = mfSamplesNew[viSorted]

        # - Fix up periodicity, if the time trace is periodic
        if self.bPeriodic:
            self._tDuration = vtTimeTraceNew[-1] - vtTimeTraceNew[0]
            self._tStart = vtTimeTraceNew[0]

        # - Create new interpolator
        self._create_interpolator()

        # - Return merged TS
        return self

    def append(self, tsOther):
        """
        append() - Combine another time series into this one, along samples axis

        :param tsOther: Another time series. Will be resampled to the time base of the called series object
        :return:        Current time series,
        """
        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), "`tsOther` must be a TimeSeries object."

        # - Resample tsOther to own time base
        mfOtherSamples = tsOther(self.vtTimeTrace)

        # - Combine samples
        self.mfSamples = np.concatenate((self.mfSamples, mfOtherSamples), 1)

        # - Create new interpolator
        self._create_interpolator()

        # - Return merged TS
        return self

    def concatenate(self, tsOther):
        """
        concatenate() - Combine another time series with this one, along samples axis

        :param tsOther: Another time series. Will be resampled to the time base of the called series object
        :return: New time series, with series from both source and other time series
        """
        return self.copy().append(tsOther)

    def append_t(self, tsOther):
        """
        append_t() - Append another time series to this one, in time

        :param tsOther: Another time series. WIll be tacked on to the end of the called series object
        :return: Self, with other TS appended
        """

        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), "`tsOther` must be a TimeSeries object."

        assert tsOther.nNumTraces == self.nNumTraces, (
            "`tsOther` must include the same number of traces ("
            + str(self.nNumTraces)
            + ")."
        )

        # - Concatenate time trace and samples
        tMedianDT = np.median(np.diff(self._vtTimeTrace))
        self._vtTimeTrace = np.concatenate(
            (
                self._vtTimeTrace,
                tsOther.vtTimeTrace
                + self._vtTimeTrace[-1]
                + tMedianDT
                - tsOther.vtTimeTrace[0],
            ),
            axis=0,
        )

        self.mfSamples = np.concatenate((self.mfSamples, tsOther.mfSamples), axis=0)

        # - Check and correct periodicity
        if self.bPeriodic:
            self._tDuration = self._vtTimeTrace[-1]

        # - Recreate interpolator
        self._create_interpolator()

        # - Return self
        return self

    def concatenate_t(self, tsOther):
        """
        concatenate_t() - Join together two time series in time

        :param tsOther: Another time series. Will be tacked on to the end of the called series object
        :return: New concatenated time series
        """
        return self.copy().append_t(tsOther)

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

    def __repr__(self):
        """
        __repr__() - Return a string representation of this object
        :return: str String description
        """
        if self.isempty():
            return "Empty {} object".format(self.__class__.__name__)
        else:
            return "{}periodic {} object from t={} to {}. Shape: {}".format(
                int(not self.bPeriodic) * "non-",
                self.__class__.__name__,
                self.tStart,
                self.tStop,
                self.mfSamples.shape,
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

    def clip(self, vtNewBounds):
        """
        clip - Clip a TimeSeries to data only within a new set of time bounds (exclusive end)
        :param vtNewBounds:

        :return: tsClip, vbIncludeSamples)
                tsClip:             New TimeSeries clipped to bounds
                vbIncludeSamples:   boolean ndarray indicating which original samples are included
        """
        # - For periodic time series, resample the series
        if self.bPeriodic:
            tsClip, _ = self._clip_periodic(vtNewBounds)
        else:
            tsClip, _ = self._clip(vtNewBounds)

        # - Insert initial time point
        tsClip._vtTimeTrace = np.concatenate(([vtNewBounds[0]], tsClip._vtTimeTrace))

        # - Insert initial samples
        vfFirstSample = np.atleast_1d(self(vtNewBounds[0]))

        tsClip._mfSamples = np.concatenate(
                (np.reshape(vfFirstSample, (-1, self.nNumTraces)),
                 np.reshape(tsClip._mfSamples, (-1, self.nNumTraces))),
                axis = 0,
        )

        return tsClip

    def _clip(self, vtNewBounds):
        """
        clip - Clip a TimeSeries to data only within a new set of time bounds (exclusive end points)
        :param vtNewBounds:
        :return: New TimeSeries clipped to bounds
        """
        # - Find samples included in new time bounds
        vtNewBounds = np.sort(vtNewBounds)
        vbIncludeSamples = np.logical_and(
            self.vtTimeTrace >= vtNewBounds[0], self.vtTimeTrace < vtNewBounds[-1]
        )

        # - Build and return new TimeSeries
        tsClip = self.copy()
        tsClip._vtTimeTrace = self.vtTimeTrace[vbIncludeSamples]
        tsClip._mfSamples = self.mfSamples[vbIncludeSamples]

        return tsClip, vbIncludeSamples

    def _clip_periodic(self, vtNewBounds):
        """
        _clip_periodic - Clip a periodic TimeSeries
        :param vtNewBounds:
        :return:
        """
        # - Ensure time bounds are sorted
        vtNewBounds = np.sort(vtNewBounds)
        tDuration = np.diff(vtNewBounds)

        # - Catch sinlgeton time point
        if vtNewBounds[0] == vtNewBounds[1]:
            return self(vtNewBounds[0]).copy()

        # - Map time bounds to periodic bounds
        vtNewBoundsPeriodic = copy.deepcopy(vtNewBounds)
        vtNewBoundsPeriodic[0] = (np.asarray(vtNewBoundsPeriodic[0]) - self._tStart
        ) % self._tDuration + self._tStart
        vtNewBoundsPeriodic[1] = vtNewBoundsPeriodic[0] + tDuration

        # - Build new time trace
        vtNewTimeTrace = copy.deepcopy(self._vtTimeTrace)
        vtNewTimeTrace = vtNewTimeTrace[vtNewTimeTrace >= vtNewBoundsPeriodic[0]]

        # - Keep appending copies of periodic time base until required duration is reached
        while vtNewTimeTrace[-1] < vtNewBoundsPeriodic[1]:
            vtNewTimeTrace = np.concatenate((vtNewTimeTrace, self._vtTimeTrace + vtNewTimeTrace[-1]))

        # - Trim new time base to end point
        vtNewTimeTrace = vtNewTimeTrace[vtNewTimeTrace <= vtNewBoundsPeriodic[1]]

        # - Restore to original time base
        vtNewTimeTrace = vtNewTimeTrace - vtNewTimeTrace[0] + vtNewBounds[0]

        # - Return a new clipped time series
        tsClip = self.resample(vtNewTimeTrace)
        return tsClip, None


    def __add__(self, other):
        return self.copy().__iadd__(other)

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
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

        # - Perform addition
        self.mfSamples += mfOtherSamples

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
        return tsCopy // (1/other)

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
    def vtTimeTrace(self, vtNewTrace):
        # - Check time trace for correct size
        assert np.size(vtNewTrace) == np.size(
            self._vtTimeTrace
        ), "New time trace must have the same number of elements as the original trace."

        # - Store new time trace
        self._vtTimeTrace = np.reshape(vtNewTrace, -1)

        # - Fix up periodicity, if the time trace is periodic
        if self.bPeriodic:
            self._tDuration = vtNewTrace[-1] - vtNewTrace[0]
            self._tStart = vtNewTrace[0]

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

    def choose(self, vnTraces):
        """
        choose() - Select from one of several sub-traces; return a new TimeSeries containing these traces

        :param vnTraces:    array-like of indices within source TimeSeries
        :return:            TimeSeries containing only the selected traces
        """
        # - Convert to a numpy array and check extents
        vnTraces = np.atleast_1d(vnTraces)
        assert (
            min(vnTraces) >= 0 and max(vnTraces) <= self.nNumTraces
        ), "`vnTraces` must be between 0 and " + str(self.nNumTraces)

        # - Return a new TimeSeries with the subselected traces
        tsCopy = self.copy()
        tsCopy.mfSamples = tsCopy.mfSamples[:, vnTraces]
        tsCopy._create_interpolator()
        return tsCopy

    def copy(self):
        """
        copy() - Return a deep copy of this time series
        :return: tsCopy
        """
        return copy.deepcopy(self)

    @property
    def mfSamples(self):
        return self._mfSamples

    @mfSamples.setter
    def mfSamples(self, mfNewSamples):
        # - Promote to 1d
        mfNewSamples = np.atleast_1d(mfNewSamples)

        # - Permit a one-dimensional sample input
        if (mfNewSamples.shape[0] == 1) and (np.size(self.vtTimeTrace) > 1):
            mfNewSamples = np.transpose(mfNewSamples)

        # - Check samples for correct size
        assert mfNewSamples.shape[0] == np.size(
            self.vtTimeTrace
        ), "New samples matrix must have the same number of samples as `.vtTimeTrace`."

        # - Store new time trace
        self._mfSamples = mfNewSamples

        # - Create a new interpolator
        self._create_interpolator()

    @property
    def tDuration(self) -> float:
        """
        .tDuration: float Duration of TimeSeries
        """
        return self._vtTimeTrace[-1] - self._vtTimeTrace[0]

    @property
    def tStart(self) -> float:
        """
        .tStart: float Start time
        """
        return self._vtTimeTrace[0]

    @property
    def tStop(self) -> float:
        """
        .tStop: float Stop time
        """
        return self._vtTimeTrace[-1]

    def _compatibleShape(self, other) -> np.ndarray:
        try:
            if np.size(other) == 1:
                return copy.copy(np.broadcast_to(other, self.mfSamples.shape))

            elif other.shape[0] == self.mfSamples.shape[0]:
                return np.reshape(other, self.mfSamples.shape)

        except Exception:
            raise ValueError("Input data must have shape " + str(self.mfSamples.shape))


### --- Continuous-valued time series


class TSContinuous(TimeSeries):
    pass


### --- Event time series


class TSEvent(TimeSeries):
    def __init__(
        self,
        vtTimeTrace: np.ndarray,
        vnChannels: np.ndarray = None,
        vfSamples: np.ndarray = None,
        strInterpKind = "linear",
        bPeriodic: bool = False,
        strName = None,
    ):
        """
        TSEvent - Represent discrete events in time

        :param vtTimeTrace:     np.array float Tx1 vector of event times
        :param vnChannels:      np.array int Tx1 vector of event channels (Default: channel 0)
        :param vfSamples:       np.array float Tx1 vector of event values (Default: nan)

        :param strInterpKind:   str Interpolation kind for event values (Default: "linear")
        :param bPeriodic:       bool Is this a periodic TimeSeries (Default: False)

        :param strName:         str Name of the time series (Default: None)
        """

        # - Only 1D samples and channels are supported
        assert (np.ndim(vfSamples) <= 1) or (
            np.sum(vfSamples.shape > 1) == 1
        ), "`vfSamples` must be 1-dimensional"

        assert (np.ndim(vnChannels) <= 1) or (
            np.sum(vnChannels.shape > 1) == 1
        ), "`vnChannels` must be 1-dimensional"

        # - Provide default inputs
        if vtTimeTrace is None:
            vtTimeTrace = np.array([])

        if vfSamples is None:
            vfSamples = full_nan((np.size(vtTimeTrace), 1))

        if vnChannels is None:
            vnChannels = np.zeros(np.size(vtTimeTrace))

        # - Check size of inputs
        assert (np.size(vnChannels) == 1) or (
            np.size(vtTimeTrace) == np.size(vnChannels)
        ), "`vnChannels` must match the size of `vtTimeTrace`"

        # - Initialise superclass
        super().__init__(
            vtTimeTrace,
            vfSamples,
            strInterpKind=strInterpKind,
            bPeriodic=bPeriodic,
            strName=strName,
        )

        # - Store channels
        self.vnChannels = np.array(vnChannels).flatten()

    def interpolate(self, vtTimes: np.ndarray):
        """
        interpolate - Return sample values interpolated to new time base

        :param vtTimes: np.ndarray Time points to interpolate to
        :return:        np.ndarray Interpolated sample values
        """
        return super().interpolate(vtTimes).flatten()

    def find(self, vtTimeBounds: Union[list, np.ndarray] = None):
        """
        find - Return events that fall within a range

        :param vtTimeBounds: array-like [tStart tFinish]

        :return:    (vtMatchingTimes, vnMatchingChannels, vfMatchingSamples)
                vtMatchingTimes:    np.ndarray Event times falling within time bounds (exclusive end time)
                vnMatchingChannels: np.ndarray Channels corresponding to matching times
                vfMatchingSamples:  np.ndarray Samples corresponding to matching times
        """

        if vtTimeBounds is not None:
            # - Map None to time trace bounds
            vtTimeBounds = np.array(vtTimeBounds)
            if vtTimeBounds[0] is None:
                vtTimeBounds[0] = self.vtTimeTrace[0]
            if vtTimeBounds[-1] is None:
                vtTimeBounds[-1] = self.vtTimeTrace[-1]
                bIncludeFinal = True
            else: bIncludeFinal = False

            # - Permit unsorted bounds
            vtTimeBounds = np.sort(vtTimeBounds)

            # - Find matching times
            vbMatchingTimes = np.logical_and(
                self.vtTimeTrace >= vtTimeBounds[0],
                np.logical_or(self.vtTimeTrace < vtTimeBounds[-1], bIncludeFinal)
            )

            # - Return matching samples
            return (
                self.vtTimeTrace[vbMatchingTimes],
                self.vnChannels[vbMatchingTimes],
                (self.mfSamples[vbMatchingTimes]).flatten(),
            )

        else:
            # - Return all samples
            return self.vtTimeTrace, self.vnChannels, self.mfSamples.flatten()

    def clip(self, vtNewBounds):
        # - Call super-class clipper
        tsClip, vbIncludeSamples = super()._clip(vtNewBounds)

        # - Fix up channels variable
        tsClip._vnChannels = self._vnChannels[vbIncludeSamples]

        # - Return new TimeSeries
        return tsClip

    def choose(self, vnSelectChannels):
        """
        choose - Select and return only the requested channels

        :param vnSelectChannels: array-like of channel indices
        :return: new TSEvent containing events from the requested channels
        """

        # - Check vnSelectChannels
        assert np.min(vnSelectChannels) >= 0 and np.max(vnSelectChannels) <= np.max(
            self.vnChannels
        ), "`vnSelectChannels` must be between 0 and {}".format(np.max(self.vnChannels))

        # - Find samples to return
        vbIncludeSamples = np.any(
            np.concatenate(
                [
                    np.atleast_2d(self._vnChannels == i)
                    for i in np.array(vnSelectChannels).flatten()
                ]
            ),
            axis=0,
        )

        # - Build new TS with only those samples
        tsChosen = self.copy()
        tsChosen._vtTimeTrace = tsChosen._vtTimeTrace[vbIncludeSamples]
        tsChosen._vnChannels = tsChosen._vnChannels[vbIncludeSamples]
        tsChosen._mfSamples = tsChosen._mfSamples[vbIncludeSamples]

        return tsChosen

    def plot(self, vtTimes: np.ndarray = None, **kwargs):
        """
        plot - Visualise a time series on plot

        :param vtTimes: Optional. Time base on which to plot. Default: time base of time series
        :param kwargs:  Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """

        # - Get samples
        vtTimes, vnChannels, _ = self.find(vtTimes)

        if self._bUseHoloviews:
            return (
                hv.Scatter((vtTimes, vnChannels))
                .redim(x="Time", y="Channel")
                .relabel(self.strName)
            )

        elif self._bUseMatplotlib:
            return plt.plot(vtTimes, self(vtTimes), **kwargs)

        else:
            warn("No plotting back-end detected.")

    def resample(self, vtTimes: np.ndarray):
        return self.clip(vtTimes)

    def resample_within(
        self, tStart: float = None, tStop: float = None, tDt: float = None
    ):
        """
        resample_within - Resample a TimeSeries, within bounds

        :param tStart:  float Start time (inclusive)
        :param tStop:   float Stop time (inclusive)
        :param tDt:     Unused for event TimeSeries

        :return:        New TimeSeries containing events within [tStart, tStop]
        """
        if tStart is None:
            tStart = self.tStart

        if tStop is None:
            tStop = self.tStop + np.finfo(float).eps

        return self.clip([tStart, tStop])

    def merge(self, ltsOther: Union[TimeSeries, List[TimeSeries]]):
        """
        merge - Merge another TimeSeries into this one
        :param ltsOther: TimeSeries (or list of TimeSeries) to merge into this one
        :return: self with new samples included
        """

        # - Ensure we have a list of objects to work on
        if not isinstance(ltsOther, collections.Iterable):
            ltsOther = [self, ltsOther]
        else:
            ltsOther = [self] + list(ltsOther)

        # - Check tsOther class
        assert all(map(lambda tsOther: isinstance(tsOther, TSEvent), ltsOther)),\
            "`tsOther` must be a `TSEvent` object."

        # - Filter out empty series
        ltsOther = list(filter(lambda ts: not ts.isempty(), ltsOther))

        # - Merge all samples
        vtNewTimeBase = np.concatenate([tsOther.vtTimeTrace for tsOther in ltsOther])
        vnNewChannels = np.concatenate([tsOther.vnChannels for tsOther in ltsOther])
        vfNewSamples = np.concatenate([tsOther.mfSamples for tsOther in ltsOther])

        # - Sort on time and merge
        vnOrder = np.argsort(vtNewTimeBase)
        self._vtTimeTrace = vtNewTimeBase[vnOrder]
        self._vnChannels = vnNewChannels[vnOrder]
        self._mfSamples = vfNewSamples[vnOrder]

        return self

    def _compatibleShape(self, other) -> np.ndarray:
        try:
            if np.size(other) == 1:
                return copy.copy(np.broadcast_to(other, self.vtTimeTrace.shape))

            elif other.shape[0] == self.vtTimeTrace.shape[0]:
                if len(other.shape) > 0:
                    if other.shape[1] == 1:
                        return other
                    else:
                        return np.reshape(other, self.vtTimeTrace.shape)

        except Exception:
            raise ValueError(
                "Input data must have shape " + str(self.vtTimeTrace.shape)
            )

    def raster(self,
               tDt: float,
               tStart: float = None,
               tStop: float = None,
               vnSelectChannels: np.ndarray = None,
               bSamples: bool = False,
        ) -> (np.ndarray, np.ndarray, np.ndarray):

        """
        raster - Return rasterized time series data, where each data point
                 represents a time step. Events are represented in a boolen
                 matrix, where the first axis corresponds to time, the second
                 axis to the channel. If bSamples is True, Samples are
                 returned in a tuple of lists (see description below).
                 Events that happen between time steps are projected to the
                 preceding one. If two events happen during one time step
                 within a single channel, they are counted as one.
        :param tDt:     float Length of single time step in raster
        :param tStart:  float Time where to start raster - Will start
                              at self.vtTImeTrace[0] if None
        :param tStop:   float Time where to stop raster. This time point is 
                              not included anymore. - If None, will use all 
                              points until (and including) self.vtTImeTrace[-1]
        :vnSelectedChannels: Array-like Channels, from which data is to be used.
        :bSamples:      bool If True, tplSamples is returned, otherwise None.

        :return
            vtTimeBase:     Time base of rasterized data
            mbEventsRaster  Boolean matrix with True indicating event
                            First axis corresponds to time, second axis to channel.
            tplSamples      Tuple with one list per time step. For each event 
                            corresponding to a time step the list contains a tuple
                            whose first entry is the channel, the second entry is
                            the sample.
                            If bSamples is False, then None is returned.
        """
        
        # - Get data from selected channels and time range
        if vnSelectChannels is not None:
            tsSelected = self.choose(vnSelectChannels)
        else:
            tsSelected = self.copy()

        vtEventTimes, vnEventChannels, vfSamples = tsSelected.find([tStart, tStop])
        
        # - Generate time base
        tStartBase = (self.tStart if tStart is None else tStart)
        tStopBase = (self.tStop + tDt if tStop is None else tStop)

        vtTimeBase = np.arange(tStartBase, tStopBase, tDt)

        # - Convert input events and samples to boolen raster
        
        nNumChannels = np.amax(tsSelected.vnChannels + 1)
        
        mbEventsRaster = np.zeros((vtTimeBase.size, nNumChannels), bool)
        
        if bSamples:
            tplSamples = tuple(([] for i in range(vtTimeBase.size)))
        else:
            tplSamples = None

        #   Iterate over channel indices and create their event raster
        for channel in range(nNumChannels):

            # Times with event in current channel
            viEventIndices_Channel = np.where(vnEventChannels == channel)[0]
            vtEventTimes_Channel = vtEventTimes[viEventIndices_Channel]

            # Indices of vtTimeBase corresponding to these times
            viEventIndices_Raster = ((vtEventTimes_Channel-vtTimeBase[0]) / tDt).astype(int)

            # Set event  and sample raster for current channel
            mbEventsRaster[viEventIndices_Raster, channel] = True
            
            # Add samples
            if bSamples:
                for iRasterIndex, iTimeIndex in zip(viEventIndices_Raster, viEventIndices_Channel):
                    tplSamples[iRasterIndex].append((channel, vfSamples[iTimeIndex]))
        
        return vtTimeBase, mbEventsRaster, tplSamples

    @property
    def vnChannels(self):
        return self._vnChannels

    @vnChannels.setter
    def vnChannels(self, vnNewChannels):
        # - Check size of new data
        assert np.size(vnNewChannels) == 1 or \
                np.size(vnNewChannels) == np.size(self.vtTimeTrace), \
            "`vnNewChannels` must be the same size as `vtTimeTrace`."

        # - Handle scalar channel
        if np.size(vnNewChannels) == 1:
            vnNewChannels = np.repeat(vnNewChannels, np.size(self._vtTimeTrace))

        # - Assign vnChannels
        self._vnChannels = vnNewChannels
