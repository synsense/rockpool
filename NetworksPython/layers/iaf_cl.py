###
# iaf_cl.py - Abstract base class for layers consisting of
#             I&F-neurons with constant leak. Clock based.
###

import numpy as np
from typing import Optional, Union, List, Tuple
from tqdm import tqdm
from abc import abstractmethod
from .cnnweights import CNNWeight
from ..timeseries import TSEvent
from . import Layer

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class CLIAF(Layer):
    """
    CLIAF - Abstract layer class of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        mfWIn: Union[np.ndarray, CNNWeight],
        vfVBias: Union[ArrayLike, float] = 0,
        vfVThresh: Union[ArrayLike, float] = 8,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVSubtract: Union[ArrayLike, float, None] = 8,
        tDt: float = 1,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        strName: str = "unnamed",
    ):
        """
        FFCLIAF - Feedforward layer of integrate and fire neurons with constant leak

        :param mfWIn:       array-like  Input weight matrix
        :param vfVBias:     array-like  Constant bias to be added to state at each time step
        :param vfVThresh:   array-like  Spiking threshold
        :param vfVReset:    array-like  Reset potential after spike (also see param bSubtract)
        :param vfVSubtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.
        :vnIdMonitor:       array-like  IDs of neurons to be recorded
        :param strName:     str  Name of this layer.
        """

        # Call parent constructor
        super().__init__(mfW=mfWIn, tDt=tDt, strName=strName)

        # - Set neuron parameters
        self.mfWIn = mfWIn
        self.vfVBias = vfVBias
        self.vfVThresh = vfVThresh
        self.vfVSubtract = vfVSubtract
        self.vfVReset = vfVReset

        # - IDs of neurons to be recorded
        self.vnIdMonitor = vnIdMonitor

    def addToRecord(
        self,
        aStateTimeSeries: list,
        tCurrentTime: float,
        vnIdOut: Union[ArrayLike, bool] = True,
        vState: Optional[np.ndarray] = None,
        bDebug: bool = False,
    ):
        """
        addToRecord: Convenience function to record current state of the layer
                     or individual neuron

        :param aStateTimeSeries: list  A simple python list object to which the
                                       state needs to be appended
        :param tCurrentTime:     float Current simulation time
        :param vnIdOut:          np.ndarray   Neuron IDs to record the state of,
                                              if True all the neuron's states
                                              will be added to the record.
                                              Default = True
        :param vState:           np.ndarray If not None, record this as state,
                                            otherwise self.vState
        :param bDebug:           bool Print debug info
        """

        vState = self.vState if vState is None else vState

        if vnIdOut is True:
            vnIdOut = np.arange(self.nSize)
        elif vnIdOut is False:
            # - Do nothing
            return

        # Update record of state changes
        for nIdOutIter in np.asarray(vnIdOut):
            aStateTimeSeries.append([tCurrentTime, nIdOutIter, vState[nIdOutIter]])
            if bDebug:
                print([tCurrentTime, nIdOutIter, vState[nIdOutIter, 0]])

    def _prepare_input(
        self, tsInput: Optional[TSEvent] = None, tDuration: Optional[float] = None
    ) -> (np.ndarray, float):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:     TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:   float Duration of the desired evolution, in seconds

        :return: (vtEventTimes, vnEventChannels, tDuration, tFinal)
            mfSpikeRaster:    ndarray Boolean raster containing spike info
            tDuration:        float Actual duration for evolution
        """

        # - Determine default duration
        if tDuration is None:
            assert (
                tsInput is not None
            ), "One of `tsInput` or `tDuration` must be supplied"

            if tsInput.bPeriodic:
                # - Use duration of periodic TimeSeries, if possible
                tDuration = tsInput.tDuration

            else:
                # - Evolve until the end of the input TImeSeries
                tDuration = tsInput.tStop - self.t
                assert (
                    tDuration > 0
                ), "Cannot determine an appropriate evolution duration. `tsInput` finishes before the current " "evolution time."

        # - Discretize tDuration wrt self.tDt
        nSamples = tDuration // self.tDt
        tDuration = nSamples * self.tDt

        # - Extract spike timings and channels
        if tsInput is not None:
            # Extract spike data from the input variable
            __, __, mfSpikeRaster, __ = tsInput.raster(
                tDt=self.tDt,
                tStart=self.t,
                tStop=self.t + tDuration,
                vnSelectChannels=np.arange(self.nDimIn),
            )

        else:
            mfSpikeRaster = np.zeros((nSamples, nDimIn), bool)

        return mfSpikeRaster, tDuration

    def reset_time(self):
        # - Set internal clock to 0
        self._t = 0

    def reset_state(self):
        # - Reset neuron state to 0
        self._vState = self.vfVReset

    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

    @property
    def cInput(self):
        return TSEvent

    @property
    def mfW(self):
        return self._mfWIn

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWIn = mfNewW

    @property
    def mfWIn(self):
        return self._mfWIn

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        if isinstance(mfNewW, CNNWeight):
            assert mfNewW.shape == (self.nDimIn, self.nSize)
            self._mfWIn = mfNewW
        else:
            assert (
                np.size(mfNewW) == self.nDimIn * self.nSize
            ), "`mfWIn` must have [{}] elements.".format(
                self.nDimIn * self.nSize
            )
            self._mfWIn = np.array(mfNewW).reshape(self.nDimIn, self.nSize)

    @property
    def vState(self):
        return self._vState

    @vState.setter
    def vState(self, vNewState):
        self._vState = self._expand_to_net_size(vNewState, "vState", bAllowNone=False)

    @property
    def vfVThresh(self):
        return self._vfVThresh

    @vfVThresh.setter
    def vfVThresh(self, vfNewThresh):
        self._vfVThresh = self._expand_to_net_size(
            vfNewThresh, "vfVThresh", bAllowNone=False
        )

    @property
    def vfVReset(self):
        return self._vfVReset

    @property
    def vfVReset(self):
        return self._vfVReset

    @vfVReset.setter
    def vfVReset(self, vfNewReset):
        self._vfVReset = self._expand_to_net_size(
            vfNewReset, "vfVReset", bAllowNone=False
        )

    @property
    def vfVSubtract(self):
        return self._vfVSubtract

    @vfVSubtract.setter
    def vfVSubtract(self, vfVNew):
        if vfVNew is None:
            self._vfVSubtract = None
        else:
            self._vfVSubtract = self._expand_to_net_size(vfVNew, "vfVSubtract")

    @property
    def vfVBias(self):
        return self._vfVBias

    @vfVBias.setter
    def vfVBias(self, vfNewBias):

        self._vfVBias = self._expand_to_net_size(vfNewBias, "vfVBias", bAllowNone=False)

    @Layer.tDt.setter
    def tDt(self, tNewDt):
        assert tNewDt > 0, "tDt must be greater than 0."
        self._tDt = tNewDt

    @property
    def vnIdMonitor(self):
        return self._vnIdMonitor

    @vnIdMonitor.setter
    def vnIdMonitor(self, vnNewIDs):
        if vnNewIDs is True:
            self._vnIdMonitor = np.arange(self.nSize)
        elif vnNewIDs is None or vnNewIDs is False or np.size(vnNewIDs) == 0:
            self._vnIdMonitor = np.array([])
        else:
            self._vnIdMonitor = self._expand_to_net_size(vnNewIDs, "vnIdMonitor")