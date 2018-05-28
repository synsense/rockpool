import numpy as np
from abc import ABC, abstractmethod

from ..TimeSeries import TimeSeries

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
        self.sName = sName
        self._tDt = tDt
        self.fNoiseStd = fNoiseStd
        self.t = None
        self.vState = None

        # - Reset state
        self.reset_all()


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

    @abstractmethod
    def reset_state(self):
        self.vState = np.zeros(self.nSize)

    @abstractmethod
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