import numpy as np
import scipy.interpolate as spint
from warnings import warn
import copy

# - Define import *
__all__ = ["TimeSeries", "SetPlottingBackend", "GetPlottingBackend"]

## -- Detect what plotting back-end to use
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

class TimeSeries:
    """
    TimeSeries - Class represent a multi-series time series, with temporal interpolation and periodicity supported
    ts = TimeSeries(vtTimeTrace, mfSamples <, strInterpKind, bPeriodic>)

    ts[tInterpTime]:
    ts(tInterpTime):
    ts.interpolate(tInterpTime): Interpolate to a time point `tInterpTime`

    ts.oInterp:         scipy.interpolate.interp1d object, interpolator
    """
    def __init__(self,
                 vtTimeTrace: np.ndarray,
                 mfSamples: np.ndarray,
                 strInterpKind: str = 'linear',
                 bPeriodic: bool = False,
                 strName = None):
        """
        TimeSeries - Class represent a multi-series time series, with temporal interpolation and periodicity supported

        :param vtTimeTrace:     [Tx1] vector of time samples
        :param mfSamples:       [TxM] matrix of values corresponding to each time sample
        :param strInterpKind:   str: Specify the interpolation type. Default: 'linear'
        :param bPeriodic:       bool: Treat the time series as periodic around the end points. Default: False

        If the time series is not periodic (the default), then NaNs will be returned for any extrapolated values.
        """

        # - Convert everything to numpy arrays
        vtTimeTrace = np.atleast_1d(np.asarray(vtTimeTrace))
        mfSamples = np.atleast_1d(mfSamples)

        # - Permit a one-dimensional sample input
        if (mfSamples.shape[0] == 1) and (np.size(vtTimeTrace) > 1):
            mfSamples = np.transpose(mfSamples)

        # - Check arguments
        assert np.size(vtTimeTrace) == mfSamples.shape[0], 'The number of time samples must be equal to the first ' \
                                                            'dimension of `mfSamples`'
        assert np.all(np.diff(vtTimeTrace) > 0), 'The time trace must be sorted and always increasing'

        # - Assign attributes
        self.__vtTimeTrace = vtTimeTrace
        self.__mfSamples = mfSamples.astype('float')
        self.strInterpKind = strInterpKind
        self.bPeriodic = bPeriodic
        self.strName = strName

        self.__bUseHoloviews, self.__bUseMatplotlib = GetPlottingBackend()

        if bPeriodic:
            self._tDuration = vtTimeTrace[-1] - vtTimeTrace[0]
            self._tStart = vtTimeTrace[0]

        self.__create_interpolator()

    def __getitem__(self, vtTimes):
        """
        ts[tTime1, tTime2, ...] - Interpolate the time series to the provided time points
        NOTE that ts[:] uses as (fixed) step size the mean step size of self.vtTimeTrace
        and thus can return different values than those in ts.mfSamples!
        :param vtTimes: Slice, scalar, list or np.array of T desired interpolated time points
        :return:      np.array of interpolated values. Will have the shape TxN
        """
        if isinstance(vtTimes, slice):
            fStep = (np.mean(np.diff(self.__vtTimeTrace)) if vtTimes.step is None else vtTimes.step)
            fStart = (self.__vtTimeTrace[0] if vtTimes.start is None else vtTimes.start)
            fStop = (self.__vtTimeTrace[-1]+abs(fStep) if vtTimes.stop is None else vtTimes.stop)
            
            assert fStart >= self.__vtTimeTrace[0],\
                   'This TimeSeries only starts at t={}'.format(self.__vtTimeTrace[0])
            assert fStop <= self.__vtTimeTrace[-1]+abs(fStep),\
                   'This TimeSeries already ends at t={}'.format(self.__vtTimeTrace[-1])
            
            vTimeIndices = np.arange(fStart, fStop, abs(fStep))[::int(np.sign(fStep))]
            return self.interpolate(vTimeIndices)
        else:
            return self.interpolate(vtTimes)

    def __call__(self, vtTimes):
        """
        ts(tTime1, tTime2, ...) - Interpolate the time series to the provided time points

        :param tTime: Scalar, list or np.array of T desired interpolated time points
        :return:      np.array of interpolated values. Will have the shape TxN
        """
        return self.interpolate(vtTimes)

    def interpolate(self, vtTimes: np.ndarray):
        """
        interpolate - Interpolate the time series to the provided time points

        :param vtTimes: np.ndarray of T desired interpolated time points
        :return:        np.ndarray of interpolated values. Will have the shape TxN
        """

        # - Enforce periodicity
        if self.bPeriodic:
            vtTimes = (np.asarray(vtTimes) - self._tStart) % self._tDuration + self._tStart

        return np.reshape(self.oInterp(vtTimes), (np.size(vtTimes), -1))

    def delay(self, tOffset):
        warn('DEPRECATED')
        tsDelayed = TimeSeries(self.vtTimeTrace + tOffset, self.mfSamples, self.strInterpKind, self.bPeriodic)
        return tsDelayed

    def plot(self, vtTimes: np.ndarray = None, **kwargs):
        """
        plot - Visualise a time series on a line plot

        :param vtTimes: Optional. Time base on which to plot. Default: time base of time series
        :param kwargs:  Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """
        if vtTimes is None:
            vtTimes = self.vtTimeTrace

        if self.__bUseHoloviews:
            mfData = self(vtTimes)
            if kwargs == {}:
                vhCurves = [hv.Curve((vtTimes, vfData)).redim(x = 'Time')
                            for vfData in mfData.T]
            else:
                vhCurves = [hv.Curve((vtTimes, vfData)).redim(x = 'Time').options(**kwargs)
                            for vfData in mfData.T]

            if len(vhCurves) > 1:
                return hv.Overlay(vhCurves).relabel(group = self.strName)
            else:
                return vhCurves[0].relabel(self.strName)

        elif self.__bUseMatplotlib:
            return plt.plot(vtTimes, self(vtTimes), **kwargs)

        else:
            warn('No plotting back-end detected.')

    def contains(self, vtTimeTrace: np.ndarray):
        """
        contains - Does the time series contain all points in the specified time trace?

        :param vtTimeTrace: Array-like containing time points
        :return:            boolean: All time points are contained within this time series
        """
        return (True if self.tStart <= np.min(vtTimeTrace) and self.tStop >= np.max(vtTimeTrace)
                     else False)

    def resample(self, vtTimes: np.ndarray):
        """
        resample - Return a new time series sampled to the supplied time base

        :param vtTimes: Array-like of T desired time points to resample
        :return:        New TimeSeries object, resampled to new time base
        """

        # - Return a new time series
        return TimeSeries(vtTimes, self(vtTimes))

    def resample_within(self, tStart: float=None, tStop: float=None, tDt: float=None):
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
        tStart = (min(self.vtTimeTrace) if tStart is None else max(tStart, min(self.vtTimeTrace)))
        tStop = (max(self.vtTimeTrace) if tStop is None else min(tStop, max(self.vtTimeTrace)))
        tDt = (np.mean(np.diff(self.vtTimeTrace)) if tDt is None else tDt)

        vtSampleTimes = np.arange(tStart, tStop+tDt, tDt)
        vtSampleTimes = vtSampleTimes[vtSampleTimes <= tStop]

        # - Return a new time series
        return self.resample(vtSampleTimes)

    def append(self, tsOther):
        """
        append() - Combine another time series into this one, along samples axis

        :param tsOther: Another time series. Will be resampled to the time base of the called series object
        :return:        Current time series,
        """
        # - Check tsOther
        assert isinstance(tsOther, TimeSeries), \
            '`tsOther` must be a TimeSeries object.'

        # - Resample tsOther to own time base
        mfOtherSamples = tsOther(self.vtTimeTrace)

        # - Combine samples
        self.mfSamples = np.concatenate((self.mfSamples, mfOtherSamples), 1)

        # - Create new interpolator
        self.__create_interpolator()

        # - Return merged TS
        return self


    def concatenate(self, tsOther):
        """
        concatenate() - Combine two time series another time series into this one, along samples axis

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
        assert isinstance(tsOther, TimeSeries), \
            '`tsOther` must be a TimeSeries object.'

        assert tsOther.nNumTraces == self.nNumTraces, \
            '`tsOther` must include the same number of traces (' + str(self.nNumTraces) + ').'

        # - Concatenate time trace and samples
        tMedianDT = np.median(np.diff(self.__vtTimeTrace))
        self.__vtTimeTrace = np.concatenate((self.__vtTimeTrace,
                                             tsOther.vtTimeTrace + self.__vtTimeTrace[-1] + tMedianDT -
                                             tsOther.vtTimeTrace[0]),
                                            axis = 0)

        self.mfSamples = np.concatenate((self.mfSamples, tsOther.mfSamples), axis = 0)

        # - Check and correct periodicity
        if self.bPeriodic:
            self._tDuration = self.__vtTimeTrace[-1]

        # - Recreate interpolator
        self.__create_interpolator()

        # - Return self
        return self

    def concatenate_t(self, tsOther):
        """
        concatenate_t() - Join together two time series in time

        :param tsOther: Another time series. Will be tacked on to the end of the called series object
        :return: New concatenated time series
        """
        return self.copy().append_t(tsOther)

    def __create_interpolator(self):
        # - Construct interpolator
        self.oInterp = spint.interp1d(self.vtTimeTrace, self.mfSamples,
                                      kind = self.strInterpKind, axis = 0, assume_sorted = True,
                                      bounds_error = False)


    def __repr__(self):
        return 'TimeSeries object ' + str(self.mfSamples.shape)

    def print(self, bForceAll: bool=False, nFirst: int=4, nLast: int=4, nShorten: int=10):
        """
        print - Print an overview of the time series and its values.
            
        :param bForceAll: Boolean - Print all samples of self, no matter how long it is
        :param nShorten:  Integer - Print shortened version of self if it comprises more
                          than nShorten time points and bForceAll is False
        :param nFirst:    Integer - Shortened version of printout contains samples at first
                          nFirst points in self.vtTimeTrace
        :param nLast:     Integer - Shortened version of printout contains samples at last
                          nLast points in self.vtTimeTrace
        """

        s = '\n'
        if len(self.vtTimeTrace) <= 10 or bForceAll:
            strSummary = s.join(['{}: \t {}'.format(t, vSamples)
                                for t, vSamples in zip(self.vtTimeTrace, self.mfSamples)])
        else:
            strSummary0 = s.join(['{}: \t {}'.format(t, vSamples)
                                  for t, vSamples in zip(self.vtTimeTrace[:nFirst],
                                                         self.mfSamples[:nFirst])])
            strSummary1 = s.join(['{}: \t {}'.format(t, vSamples)
                                  for t, vSamples in zip(self.vtTimeTrace[-nLast:],
                                                         self.mfSamples[-nLast:])])
            strSummary = strSummary0 + '\n\t...\n' + strSummary1
        print(self.__repr__() + '\n' + strSummary)
            

    def __add__(self, other):
        return self.copy().__iadd__(other)

    def __iadd__(self, other):
        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self.__compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self.__compatibleShape(other)

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
        self.__create_interpolator()
        return self

    def __mul__(self, other):
        return self.copy().__imul__(other)

    def __imul__(self, other):
        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self.__compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self.__compatibleShape(other)

        # - Propagate NaNs
        mbIsNanSelf = np.isnan(self.mfSamples)
        mbIsNanOther = np.isnan(mfOtherSamples)

        # - Perform multiplication
        self.mfSamples *= mfOtherSamples

        # - Fill in nans
        self.mfSamples[np.logical_or(mbIsNanSelf, mbIsNanOther)] = np.nan

        # - Re-create interpolator
        self.__create_interpolator()
        return self

    def __truediv__(self, other):
        return self.copy().__idiv__(other)

    def __idiv__(self, other):
        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self.__compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self.__compatibleShape(other)

        # - Propagate NaNs
        mbIsNanSelf = np.isnan(self.mfSamples)
        mbIsNanOther = np.isnan(mfOtherSamples)

        # - Perform division
        self.mfSamples /= mfOtherSamples

        # - Fill in nans
        self.mfSamples[np.logical_or(mbIsNanSelf, mbIsNanOther)] = np.nan

        # - Re-create interpolator
        self.__create_interpolator()
        return self

    def __floordiv__(self, other):
        return self.copy().__ifloordiv__(other)

    def __ifloordiv__(self, other):
        # - Get nan mask
        mbIsNan = np.isnan(self.mfSamples)

        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self.__compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self.__compatibleShape(other)

        # - Propagate NaNs
        mbIsNanSelf = np.isnan(self.mfSamples)
        mbIsNanOther = np.isnan(mfOtherSamples)

        # - Perform division
        self.mfSamples //= mfOtherSamples

        # - Fill in nans
        self.mfSamples[np.logical_or(mbIsNanSelf, mbIsNanOther)] = np.nan

        # - Re-create interpolator
        self.__create_interpolator()
        return self

    def max(self):
        return np.nanmax(self.mfSamples)

    def min(self):
        return np.nanmin(self.mfSamples)

    def __abs__(self):
        tsCopy = self.copy()
        tsCopy.mfSamples = np.abs(tsCopy.mfSamples)
        tsCopy.__create_interpolator()
        return tsCopy

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def __isub__(self, other):
        # - Should we handle TimeSeries subtraction?
        if isinstance(other, TimeSeries):
            mfOtherSamples = self.__compatibleShape(other(self.vtTimeTrace))
        else:
            mfOtherSamples = self.__compatibleShape(other)

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
        self.__create_interpolator()
        return self

    def __neg__(self):
        tsCopy = self.copy()
        tsCopy.mfSamples = -tsCopy.mfSamples
        tsCopy.__create_interpolator()
        return tsCopy

    @property
    def vtTimeTrace(self):
        return self.__vtTimeTrace

    @vtTimeTrace.setter
    def vtTimeTrace(self, vtNewTrace):
        # - Check time trace for correct size
        assert np.size(vtNewTrace) == np.size(self.__vtTimeTrace), \
            'New time trace must have the same number of elements as the original trace.'

        # - Store new time trace
        self.__vtTimeTrace = np.reshape(vtNewTrace, -1)

        # - Fix up periodicity, if the time trace is periodic
        if self.bPeriodic:
            self._tDuration = vtNewTrace[-1] - vtNewTrace[0]
            self._tStart = vtNewTrace[0]

        # - Create a new interpolator
        self.__create_interpolator()    

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
        assert min(vnTraces) >= 0 and max(vnTraces) <= self.nNumTraces, \
            '`vnTraces` must be between 0 and ' + str(self.nNumTraces)

        # - Return a new TimeSeries with the subselected traces
        tsCopy = self.copy()
        tsCopy.mfSamples = tsCopy.mfSamples[:, vnTraces]
        tsCopy.__create_interpolator()
        return tsCopy


    def copy(self):
        """
        copy() - Return a deep copy of this time series
        :return: tsCopy
        """
        return copy.deepcopy(self)


    @property
    def mfSamples(self):
        return self.__mfSamples

    @mfSamples.setter
    def mfSamples(self, mfNewSamples):
        # - Promote to 1d
        mfNewSamples = np.atleast_1d(mfNewSamples)

        # - Permit a one-dimensional sample input
        if (mfNewSamples.shape[0] == 1) and (np.size(self.vtTimeTrace) > 1):
            mfNewSamples = np.transpose(mfNewSamples)

        # - Check samples for correct size
        assert mfNewSamples.shape[0] == np.size(self.vtTimeTrace), \
            'New samples matrix must have the same number of samples as `.vtTimeTrace`.'

        # - Store new time trace
        self.__mfSamples = mfNewSamples

        # - Create a new interpolator
        self.__create_interpolator() 

    @property
    def tDuration(self) -> float:
        """
        .tDuration: float Duration of TimeSeries
        """
        return self.__vtTimeTrace[-1] - self.__vtTimeTrace[0]

    @property
    def tStart(self) -> float:
        """
        .tStart: float Start time
        """
        return self.__vtTimeTrace[0]

    @property
    def tStop(self) -> float:
        """
        .tStop: float Stop time
        """
        return self.__vtTimeTrace[-1]

    def __compatibleShape(self, other) -> np.ndarray:
        try:
            if np.size(other) == 1:
                return copy.copy(np.broadcast_to(other, self.mfSamples.shape))

            elif other.shape[0] == self.mfSamples.shape[0]:
                if len(other.shape) > 0:
                    if other.shape[1] == 1:
                        return other
                    else:
                        return np.reshape(other, self.mfSamples.shape)

        except Exception:
            raise ValueError('Input data must have shape ' + str(self.mfSamples.shape))


## -- Code for setting plotting backend

def SetPlottingBackend(strBackend):
    global __bHoloviewsDetected
    global __bMatplotlibDetected
    global __bUseHoloviews
    global __bUseMatplotlib
    if strBackend == 'holoviews' and __bHoloviewsDetected:
        __bUseHoloviews = True
        __bUseMatplotlib = False

    elif strBackend == 'matplotlib' and __bMatplotlibDetected:
        __bUseHoloviews = False
        __bUseMatplotlib = True

def GetPlottingBackend():
    return __bUseHoloviews, __bUseMatplotlib

## - Set default plotting backend
if __bHoloviewsDetected:
    SetPlottingBackend('holoviews')

elif __bMatplotlibDetected:
    SetPlottingBackend('matplotlib')