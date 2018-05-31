import numpy as np
import scipy.interpolate as spint
from warnings import warn
import copy

# - Define import *
__all__ = ["TimeSeries"]

# - Detect what platting back-end to use
TS_bUseHoloviews = False
TS_bUseMatplotlib = False

try:
    import holoviews as hv
    TS_bUseHoloviews = True

except Exception:
    try:
        import matplotlib.pyplot as plt
        TS_bUseMatplotlib = True

    except Exception:
        pass

def plotting_backend(strBackend):
    global TS_bUseMatplotlib, TS_bUseHoloviews
    if strBackend in ('holoviews', 'holo', 'Holoviews', 'hv'):
        try:
            import holoviews as hv
            TS_bUseMatplotlib = False
            TS_bUseHoloviews = True
            print('Using holoviews as plotting back-end.')
        except ModuleNotFoundError:
            print('Cannot import holoviews')
    elif strBackend in ('matplotlib', 'mpl', 'mp', 'pyplot', 'plt'):
        try:
            import matplotlib.pyplot as plt
            TS_bUseHoloviews = False
            TS_bUseMatplotlib = True
            print('Using matplotlib as plotting back-end.')
        except ModuleNotFoundError:
            print('Cannot import matplotlib.pyplot.')
    elif strBackend is None or strBackend == 'none':
        TS_bUseMatplotlib = False
        TS_bUseHoloviews = False
        print('No plotting back-end set.')
    else:
        print('Backend unkkown')


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

        if TS_bUseHoloviews:
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

        elif TS_bUseMatplotlib:
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

        return self.resample(vtSampleTimes)


        # - Return a new time series
        return TimeSeries(vtTimes, self(vtTimes))
    def __create_interpolator(self):
        # - Construct interpolator
        self.oInterp = spint.interp1d(self.vtTimeTrace, self.mfSamples,
                                      kind = self.strInterpKind, axis = 0, assume_sorted = True,
                                      bounds_error = False)


    def __repr__(self):
        return 'TimeSeries object ' + str(self.mfSamples.shape)

    def print(self, bForceAll: bool=False, nFirst: int=4, nLast: int=4, nShorten: int=10):
        """
        print - Print an overview over the time series and its values.
            
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
        tsCopy = copy.deepcopy(self)
        tsCopy += other
        return tsCopy

    def __iadd__(self, other):
        self.mfSamples += other
        self.__create_interpolator()
        return self

    def __mul__(self, other):
        tsCopy = copy.deepcopy(self)
        tsCopy *= other
        return tsCopy

    def __imul__(self, other):
        self.mfSamples *= other
        self.__create_interpolator()
        return self

    def __truediv__(self, other):
        tsCopy = copy.deepcopy(self)
        tsCopy /= other
        return tsCopy

    def __idiv__(self, other):
        self.mfSamples /= other
        self.__create_interpolator()
        return self

    def __floordiv__(self, other):
        tsCopy = copy.deepcopy(self)
        tsCopy //= other
        return tsCopy

    def __ifloordiv__(self, other):
        self.mfSamples //= other
        self.__create_interpolator()
        return self

    def max(self):
        return np.max(self.mfSamples)

    def min(self):
        return np.min(self.mfSamples)

    def __abs__(self):
        tsCopy = copy.deepcopy(self)
        tsCopy.mfSamples = np.abs(tsCopy.mfSamples)
        tsCopy.__create_interpolator()
        return tsCopy

    def __sub__(self, other):
        tsCopy = copy.deepcopy(self)
        tsCopy.mfSamples -= other
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
        try:
            return self.mfSamples.shape[1]
        # If mfSamples is 1d:
        except IndexError:
            return 1

    def choose(self, vnTraces):
        # - Convert to a numpy array and check extents
        vnTraces = np.atleast_1d(vnTraces)
        assert min(vnTraces) >= 0 and max(vnTraces) <= self.nNumTraces, \
            '`vnTraces` must be between 0 and ' + str(self.nNumTraces)

        # - Return a new TimeSeries with the subselected traces
        tsCopy = copy.deepcopy(self)
        tsCopy.mfSamples = tsCopy.mfSamples[:, vnTraces]
        tsCopy.__create_interpolator()
        return tsCopy

    @property
    def mfSamples(self):
        return self.__mfSamples

    @mfSamples.setter
    def mfSamples(self, mfNewSamples):
        # - Check samples for correct size
        assert len(mfNewSamples) == len(self.__mfSamples), \
            'New samples matrix must have the same shape as the original matrix.'

        # - Store new time trace
        self.__mfSamples = mfNewSamples

        # - Create a new interpolator
        self.__create_interpolator() 

    @property
    def tDuration(self):
        return self.__vtTimeTrace[-1] - self.__vtTimeTrace[0]

    @property
    def tStart(self):
        return self.__vtTimeTrace[0]

    @property
    def tStop(self):
        return self.__vtTimeTrace[-1]