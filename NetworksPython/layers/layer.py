import numpy as np

# Implements the Layer class

class Layer():
    def __init__(self,
                 mfW: np.ndarray = None,
                 tDt: float = 1,
                 sName: str = None):
        """
        Layer class - Implement an abstract layer of neurons (no implementation)

        :param mfW:
        :param tDt:
        :param sName:
        """

        # - Ensure weights are at least 2D
        mfW = np.atleast_2d(mfW)

        # - Assign properties
        self._mfW = mfW
        self._nDimIn, self._nSize = mfW.shape
        self.vState = np.zeros(self.nSize)
        self.sName = sName
        self.t = 0
        self._tDt = tDt


    ### --- String representations

    def __str__(self):
        return '{} object: "{}"'.format(self.__class__.__name__, self.sName)

    def __repr__(self):
        return self.__str__()


    #### --- Properties

    @property
    def nSize(self):
        return self._nSize

    @property
    def nDimIn(self):
        return self._nDimIn

    @property
    def tDt(self):
        return self._tDt

    @tDt.setter
    def tDt(self, fNewDt):
        self._tDt = fNewDt

    @property
    def mfW(self):
        return self._mfW

    @mfW.setter
    def mfW(self, mfNewW: np.ndarray):
        # - Check dimensionality of new weights
        assert mfNewW.size == self.nDimIn * self.nSize, \
            '`mfNewW` must be of shape {}'.format((self.nDimIn, self.nSize))

        # - Save weights with appropriate size
        self._mfW = np.reshape(mfNewW, (self.nDimIn, self.nSize))