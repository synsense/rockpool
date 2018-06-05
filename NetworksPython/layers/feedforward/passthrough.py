import numpy as np

from TimeSeries import TimeSeries
from layers.layer import Layer
from layers import noisy, isMultiple

class PassThrough(Layer):
    """ Neuron states directly correspond to input, but can be delayed. """

    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = 1,
                 fNoiseStd: float = 0,
                 tDelay: float = 0,
                 strName: str = None):
        super().__init__(mfW=mfW, tDt=tDt, fNoiseStd=fNoiseStd, strName=strName)
        self._tDelay = (0 if tDelay is None else tDelay)
        self.reset_all()

        # Buffer already reset by super().__init__ which calls self.reset_all()
        # self.reset_buffer()

    def reset_buffer(self):
        if self.tDelay != 0:
            # - Make sure that self.tDelay is a multiple of self.tDt
            if not isMultiple(self.tDelay, self.tDt):
                raise ValueError('tDelay must be a multiple of tDt')

            vtBuffer = np.arange(0, self.tDelay+self._tDt, self._tDt)
            self.tsBuffer = TimeSeries(vtBuffer,
                                       np.zeros((len(vtBuffer), self.nSize)))
        else:
            self.tsBuffer = None

    def evolve(self, tsInput: np.ndarray, tDuration: float = None):
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TimeSeries TxM or Tx1 input to this layer
        :param tDuration:   float Duration of evolution, in seconds

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Discretize input time series
        vtTimeIn, mfInput, tTrueDuration = self._prepare_input(tsInput, tDuration)

        # - Apply input weights and add noise
        mfInProcessed = noisy(mfInput@self.mfW, self.fNoiseStd)

        if self.tsBuffer is not None:
            # - Combined time trace for buffer and processed input
            vtTimeComb = self._gen_time_trace(self.t, tTrueDuration+self.tDelay)
            # - Array for buffered and new data
            mfSamplesComb = np.zeros((vtTimeComb.size, self.nDimIn))
            nStepsIn = vtTimeIn.size
            # - Buffered data: last point of buffer data corresponds to self.t,
            #   which is also part of current input
            mfSamplesComb[ :-nStepsIn] = self.tsBuffer.mfSamples[:-1]
            # - Processed input data (weights and noise)
            mfSamplesComb[-nStepsIn: ] = mfInProcessed

            # - Output data
            mfSamplesOut = mfSamplesComb[ :nStepsIn]

            # - Update buffer with new data
            self.tsBuffer.mfSamples = mfSamplesComb[nStepsIn-1:]

        else:
            # - Undelayed processed input
            mfSamplesOut = mfInProcessed

        self._t += tTrueDuration

        return TimeSeries(vtTimeIn, mfSamplesOut)

    def __repr__(self):
        return 'PassThrough layer object `{}`.\nnSize: {}, nDimIn: {}, tDelay: {}'.format(
            self.strName, self.nSize, self.nDimIn, self.tDelay)

    def print_buffer(self, **kwargs):
        if self.tsBuffer is not None:
            self.tsBuffer.print(**kwargs)
        else:
            print('This layer does not use a delay.')

    @property
    def mfBuffer(self):
        if self.tsBuffer is not None:
            return self.tsBuffer.mfSamples
        else:
            print('This layer does not use a delay.')

    def reset_state(self):
        super().reset_state()
        self.reset_buffer()

    def reset_all(self):
        super().reset_all()
        self.reset_buffer()

    @property
    def tDelay(self):
        return self._tDelay

    # @tDelay.setter
    # def tDelay(self, tNewDelay):
        # Some method to extend self.tsBuffer
