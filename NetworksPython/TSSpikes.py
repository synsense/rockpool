
### --- Imports
import numpy as np
from typing import Union
import copy
from warnings import warn

from TimeSeries import TimeSeries, GetPlottingBackend

# - Try to import plotting backends
try: import holoviews as hv
except Exception: pass

try: import matplotlib.pyplot as plt
except Exception: pass

# - Configure exports
__all__ = ['TSSpikes']


class TSSpikes(TimeSeries):
    def __init__(self,
                 vtTimeTrace: np.ndarray,
                 vnChannels: np.ndarray = None,
                 vfSamples: np.ndarray = None,
                 strInterpKind = 'linear',
                 bPeriodic: bool = False,
                 strName = None,
                 ):

        # - Only 1D samples and channels are supported
        assert np.ndim(vfSamples) <= 1, \
            '`vfSamples` must be 1-dimensional'

        assert np.ndim(vnChannels) <= 1, \
            '`vnChannels` must be 1-dimensional'

        # - Check size of inputs
        assert (np.size(vnChannels) == 1) or (np.size(vtTimeTrace) == np.size(vnChannels)), \
            '`vnChannels` must match the size of `vtTimeTrace`'

        # - Provide default inputs
        if vfSamples is None:
            vfSamples = full_nan(np.size(vtTimeTrace))

        if vnChannels is None:
            vnChannels = np.zeros(np.size(vtTimeTrace))

        # - Initialise superclass
        super().__init__(vtTimeTrace, vfSamples,
                         strInterpKind = strInterpKind,
                         bPeriodic = bPeriodic,
                         strName = strName)

        # - Store channels
        self.__vnChannels = np.array(vnChannels).flatten()

    def __create_interpolator(self):
        # - No interpolation is performed for a spiking time series
        pass


    def find(self, vtTimeBounds: np.ndarray = None):
        """
        find - Return events that fall within a range

        :param vtTimeBounds: array-like [tStart tFinish]

        :return:    (vtMatchingTimes, vnMatchingChannels, vfMatchingSamples)
                vtMatchingTimes:    np.ndarray Event times falling within time bounds (exclusive end time)
                vnMatchingChannels: np.ndarray Channels corresponding to matching times
                vfMatchingSamples:  np.ndarray Samples corresponding to matching times
        """

        if vtTimeBounds is not None:
            # - Permit unsorted bounds
            vtTimeBounds = np.sort(vtTimeBounds)

            # - Find matching times
            vbMatchingTimes = np.logical_and(self.vtTimeTrace >= vtTimeBounds[0],
                                             self.vtTimeTrace < vtTimeBounds[-1])

            # - Return matching samples
            return self.vtTimeTrace[vbMatchingTimes], \
                   self.__vnChannels[vbMatchingTimes], \
                   self.mfSamples[vbMatchingTimes]

        else:
            # - Return all samples
            return self.vtTimeTrace, self.__vnChannels, self.mfSamples


    def plot(self, vtTimes: np.ndarray = None, **kwargs):
        """
        plot - Visualise a time series on plot

        :param vtTimes: Optional. Time base on which to plot. Default: time base of time series
        :param kwargs:  Optional arguments to pass to plotting function

        :return: Plot object. Either holoviews Layout, or matplotlib plot
        """

        # - Get samples
        vtTimes, vnChannels, _ = self.find(vtTimes)

        bUseHoloviews, bUseMatplotlib = GetPlottingBackend()
        if bUseHoloviews:
            return hv.Scatter((vtTimes, vnChannels)) \
                .redim(x = 'Time', y = 'Channel') \
                .relabel(self.strName)

        elif bUseMatplotlib:
            return plt.plot(vtTimes, self(vtTimes), **kwargs)

        else:
            warn('No plotting back-end detected.')


    def __compatibleShape(self, other) -> np.ndarray:
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
            raise ValueError('Input data must have shape ' + str(self.vtTimeTrace.shape))

# - Convenience method to return a nan array
def full_nan(vnShape: Union[tuple, int]):
    a = np.empty(vnShape)
    a.fill(np.nan)
    return a
