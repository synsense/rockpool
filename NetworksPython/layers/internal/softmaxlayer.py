##
# softmaxlayer.py - Implement a softmax layer using spiking inputs
##

import numpy as np
from ...timeseries import TSEvent, TSContinuous
from .iaf_cl import FFCLIAF
from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


def softmax(x: np.ndarray) -> float:
    """
    softmax - Compute softmax values for each of scores in x
    :param x:   ndarray Vector of values over which to compute softmax
    :return:    float   SoftMax of the input values
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class SoftMaxLayer(FFCLIAF):
    """
    SoftMaxLayer: SoftMaxLayer with spiking inputs and outputs. Constant leak.
    """

    def __init__(
        self,
        mfW: np.ndarray = None,
        fVth: float = 1e10,         # Just some absurdly large number that will never be reachable
        tDt: float = 1,
        strName: str = "unnamed",
    ):
        """
        SoftMaxLayer - Implements a softmax on the inputs

        :param mfW:     np.ndarray  Weight matrix
        :param fVth:    float       Spiking threshold
        :param tDt:     float       Time step
        :param strName: str         Name of this layer.
        """

        # Call parent constructor
        FFCLIAF.__init__(self, mfW, tDt=tDt, strName=strName)
        self.fVth = fVth
        self.__nIdMonitor__ = None  # Monitor all neurons

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:                TSEvent  output spike series

        """

        # - Use `evolve()` from the base class
        _evOut = FFCLIAF.evolve(
            self, tsInput=tsInput, tDuration=tDuration, nNumTimeSteps=nNumTimeSteps
        )
        assert len(_evOut.times) == 0

        # - Analyse states
        mfStateHistoryLog = self._mfStateTimeSeries[10:]

        # - Convert state data to TimeSeries format
        mfStateTimeSeries = np.zeros((nNumTimeSteps, self.nSize))
        for t in range(tDuration):
            mfDataTimeStep = mfStateHistoryLog[(mfStateHistoryLog[:, 0] == t)]
            mfStateTimeSeries[t] = mfDataTimeStep[:, 2]

        # - Compute softmax over the input states
        mfSoftMax = softmax(mfStateTimeSeries)
        tsOut = TSContinuous(
            times=np.arange(tDuration),
            samples=mfSoftMax,
            name="SoftMaxOutput",
        )
        return tsOut

    @property
    def cOutput(self):
        return TSContinuous
