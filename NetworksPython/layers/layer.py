import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

from TimeSeries import TimeSeries

# Implements the Layer class

class Layer(ABC):
    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = 1,
                 fNoiseStd: float = 0,
                 sName: str = None):
        """
        Layer class - Implement an abstract layer of neurons (no implementation)

        :param mfW:
        :param tDt:
        :param fNoiseStd:
        :param sName:
        """

        # - Ensure weights are at least 2D
        mfW = np.atleast_2d(mfW)

        # - Assign properties
        self._mfW = mfW
        self._nDimIn, self._nSize = mfW.shape

        if sName is None:
            self.sName = '(unnamed)'
        else:
            self.sName = sName

        self._tDt = tDt
        self.fNoiseStd = fNoiseStd
        self.t = None
        self.vState = None

        # - Reset state
        self.reset_all()


    ### --- Common methods

    def _check_input_dims(self, tsInput: TimeSeries) -> TimeSeries:
        """
        Verify if dimension of input matches layer instance. If input
        dimension == 1, scale it up to self._nDimIn by repeating signal.
            tsInput : input time series
            return : tsInput, possibly with dimensions repeated
        """
        # - Replicate `tsInput` if necessary
        if tsInput.nNumTraces == 1:
            tsInput = deepcopy(tsInput)
            tsInput.mfSamples = np.repeat(tsInput.mfSamples.reshape((-1, 1)),
                                          self._nDimIn, axis = 1)

        # - Check dimensionality of input
        assert tsInput.nNumTraces == self._nDimIn, \
            'Input dimensionality {} does not match layer input size {}.'.format(tsInput.nNumTraces, self._nDimIn)

        # - Return possibly corrected input
        return tsInput

    def _gen_time_trace(self, tStart: float, tDuration: float):
        """
        Generate a time trace starting at tStart, of length tDuration with 
        time step length self._tDt. Make sure it does not go beyond 
        tStart+tDuration.
        """
        # - Generate a periodic trace
        tStop = tStart + tDuration
        vtTimeTrace = np.arange(0, tDuration+self._tDt, self._tDt) + tStart

        # - Make sure that vtTimeTrace doesn't go beyond tStop
        return vtTimeTrace[vtTimeTrace <= tStop]

    ### --- String representations

    def __str__(self):
        return '{} object: "{}"'.format(self.__class__.__name__, self.sName)

    def __repr__(self):
        return self.__str__()


    ### --- State evolution methods

    @abstractmethod
    def evolve(self,
               tsInput: TimeSeries = None,
               tDuration: float = None):
        pass

    def reset_state(self):
        self.vState = np.zeros(self.nSize)

    def reset_all(self):
        self.t = 0
        self.reset_state()


    #### --- Properties

    @property
    def nSize(self) -> int:
        return self._nSize

    @property
    def nDimIn(self) -> int:
        return self._nDimIn

    @property
    def tDt(self) -> float:
        return self._tDt

    @tDt.setter
    def tDt(self, fNewDt: float):
        self._tDt = fNewDt

    @property
    def mfW(self) -> np.ndarray:
        return self._mfW

    @mfW.setter
    def mfW(self, mfNewW: np.ndarray):
        # - Check dimensionality of new weights
        assert mfNewW.size == self.nDimIn * self.nSize, \
            '`mfNewW` must be of shape {}'.format((self.nDimIn, self.nSize))

        # - Save weights with appropriate size
        self._mfW = np.reshape(mfNewW, (self.nDimIn, self.nSize))