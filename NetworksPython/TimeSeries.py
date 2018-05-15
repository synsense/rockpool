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
        if mfSamples.shape[0] == 1:
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

        :param vtTimes: Scalar, list or np.array of T desired interpolated time points
        :return:      np.array of interpolated values. Will have the shape TxN
        """
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
            return plt.plot(vtTimes, self(vtTimes), kwargs)

        else:
            warn('No plotting back-end detected.')

    def __create_interpolator(self):
        # - Construct interpolator
        self.oInterp = spint.interp1d(self.vtTimeTrace, self.mfSamples,
                                      kind = self.strInterpKind, axis = 0, assume_sorted = True,
                                      bounds_error = False)


    def __repr__(self):
        return 'TimeSeries object ' + str(self.mfSamples.shape)

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
        return self.mfSamples.shape[1]

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
        assert mfNewSamples.shape == self.__mfSamples.shape, \
            'New samples matrix must have the same shape as the original matrix.'

        # - Store new time trace
        self.__mfSamples = mfNewSamples

        # - Create a new interpolator
        self.__create_interpolator() 