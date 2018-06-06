
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
__all__ = ['TSEvent']


class TSEvent(TimeSeries):
    def __init__(self,
                 vtTimeTrace: np.ndarray,
                 vnChannels: np.ndarray = None,
                 vfSamples: np.ndarray = None,
                 strInterpKind = 'linear',
                 bPeriodic: bool = False,
                 strName = None,
                 ):

        # - Only 1D samples and channels are supported
        assert (np.ndim(vfSamples) <= 1) or (np.sum(vfSamples.shape > 1) == 1), \
            '`vfSamples` must be 1-dimensional'

        assert (np.ndim(vnChannels) <= 1) or (np.sum(vnChannels.shape > 1) == 1), \
            '`vnChannels` must be 1-dimensional'

        # - Check size of inputs
        assert (np.size(vnChannels) == 1) or (np.size(vtTimeTrace) == np.size(vnChannels)), \
            '`vnChannels` must match the size of `vtTimeTrace`'

        # - Provide default inputs
        if vfSamples is None:
            vfSamples = full_nan((np.size(vtTimeTrace), 1))

        if vnChannels is None:
            vnChannels = np.zeros(np.size(vtTimeTrace))

        # - Initialise superclass
        super().__init__(vtTimeTrace, vfSamples,
                         strInterpKind = strInterpKind,
                         bPeriodic = bPeriodic,
                         strName = strName)

        # - Store channels
        self._vnChannels = np.array(vnChannels).flatten()


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
            # - Permit unsorted bounds
            vtTimeBounds = np.sort(vtTimeBounds)

            # - Find matching times
            vbMatchingTimes = np.logical_and(self.vtTimeTrace >= vtTimeBounds[0],
                                             self.vtTimeTrace < vtTimeBounds[-1])

            # - Return matching samples
            return self.vtTimeTrace[vbMatchingTimes], \
                   self.vnChannels[vbMatchingTimes], \
                   (self.mfSamples[vbMatchingTimes]).flatten()

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
        assert np.min(vnSelectChannels) >= 0 and np.max(vnSelectChannels) <= np.max(self.vnChannels), \
            '`vnSelectChannels` must be between 0 and {}'.format(np.max(self.vnChannels))

        # - Find samples to return
        vbIncludeSamples = np.any(np.concatenate([np.atleast_2d(self._vnChannels == i)
                                                  for i in np.array(vnSelectChannels).flatten()]), axis = 0)

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
            return hv.Scatter((vtTimes, vnChannels)) \
                .redim(x = 'Time', y = 'Channel') \
                .relabel(self.strName)

        elif self._bUseMatplotlib:
            return plt.plot(vtTimes, self(vtTimes), **kwargs)

        else:
            warn('No plotting back-end detected.')


    def resample(self, vtTimes: np.ndarray):
        return self.clip(vtTimes)

    def resample_within(self, tStart: float=None, tStop: float=None, tDt: float=None):
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

    def merge(self, tsOther):
        """
        merge - Merge another TimeSeries into this one
        :param tsOther: TimeSeries to merge into this one
        :return: self with new samples included
        """

        # - Check tsOther
        assert isinstance(tsOther, TSEvent), \
            '`tsOther` must be a `TSEvent` object.'

        # - Merge samples
        vtNewTimeBase = np.concatenate((self.vtTimeTrace, tsOther.vtTimeTrace))
        vnNewChannels = np.concatenate((self.vnChannels, tsOther.vnChannels))
        vfNewSamples = np.concatenate((self.mfSamples, tsOther.mfSamples))

        # - Sort on time
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
            raise ValueError('Input data must have shape ' + str(self.vtTimeTrace.shape))

    @property
    def vnChannels(self):
        return self._vnChannels

    @vnChannels.setter
    def vnChannels(self, vnNewChannels):
        # - Check size of new data
        assert np.size(vnNewChannels) == np.size(self._vnChannels), \
            '`vnNewChannels` must be the same size as `vnChannels`.'

        # - Assign vnChannels
        self._vnChannels = vnNewChannels


# - Convenience method to return a nan array
def full_nan(vnShape: Union[tuple, int]):
    a = np.empty(vnShape)
    a.fill(np.nan)
    return a
