import numpy as np
from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer
from typing import Optional, Union, Tuple, List
from AudiotoryProcessing.preprocessing import *

class Filter(Layer):
    """ Filter - Class: define a filtering layer with continuous time series output
    """
## - Constructor
    def __init__(
        self,
        mfW: np.ndarray,
        filterName: str,
        fs: float,
        vfBias: Union[float, np.ndarray] = 0,
        tDt: float = 0.1 ,
        strName: str = "unnamed",
    ):
        """
        FFIAFBrian - Construct a spiking feedforward layer with IAF neurons, with a Brian2 back-end
                     Inputs are continuous currents; outputs are spiking events

        :param mfW:             np.array MxN weight matrix.
        :param filterName:      str with the filtering method
        :param fs:              float sampling frequency of input signal
        :param vfBias:          np.array Nx1 bias vector
        :param tDt:             float Time-step. Default: 0.1 ms
        :param fNoiseStd:       float Noise std. dev. per second. Default: 0
        :param strName:         str Name for the layer. Default: 'unnamed'

        """

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            mfW=np.asarray(mfW),
            tDt=np.asarray(tDt),
            strName=strName,
        )

        self.fs = fs
        self.nNumTraces = mfW.shape[1]
        self.filtFunct = function_Filterbank(filterName)
        self.vfBias = vfBias
        self.mfW = mfW
        self._nTimeStep = 0

    def reset_all(self):
        self.mfW = np.copy(self.mfW)
        vfBias = np.copy(self.vfBias)
        self._nTimeStep = 0

    def evolve(
        self,
        tsInput: Optional[TSContinuous] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False
    ) ->TSContinuous:

        # - Prepare time base
        vtTimeBase, mfInputStep, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        filtOuput = self.filtFunct(mfInputStep.T[0], self.fs, self.nNumTraces, downSampleFs=self.fs, bSamelength=False, order=2)

        self._nTimeStep += mfInputStep.shape[0] - 1

        return TSContinuous(
            vtTimeBase,
            filtOuput,
            strName="filteredInput",
        )





