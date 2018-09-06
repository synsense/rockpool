import numpy as np
import warnings
from collections import UserList
from functools import reduce
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from ...timeseries import TSEvent
from .spiking_conv2d_torch import CNNWeightTorch
from .iaf_cl import FFCLIAF

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class FFCLIAFTorch(FFCLIAF):
    """
    FFCLIAFTorch - Feedforward layer of integrate and fire neurons with constant leak
    Implemented using pytorch for speed and is meant to be used for convolutions
    """

    def __init__(
        self,
        mfW: CNNWeightTorch,
        vfVBias: Union[ArrayLike, float] = 0,
        vfVThresh: Union[ArrayLike, float] = 8,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVSubtract: Union[ArrayLike, float, None] = 8,
        tDt: float = 1,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        strName: str = "unnamed",
    ):
        """
        FFCLIAFTorch - Feedforward layer of integrate and fire neurons with constant leak

        :param mfW:         array-like  Input weight matrix
        :param vfVBias:     array-like  Constant bias to be added to state at each time step
        :param vfVThresh:   array-like  Spiking threshold
        :param vfVReset:    array-like  Reset potential after spike (also see param bSubtract)
        :param vfVSubtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.
        :vnIdMonitor:       array-like  IDs of neurons to be recorded
        :param strName:     str  Name of this layer.
        """

        # Call parent constructor
        FFCLIAF.__init__(
            self,
            mfW=mfW,
            vfVBias=vfVBias,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVSubtract=vfVSubtract,
            tDt=tDt,
            vnIdMonitor=vnIdMonitor,
            strName=strName,
        )

        self.reset_state()

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
        :return:            TSEvent  output spike series

        """

        # - Generate input in rasterized form, get actual evolution duration
        mfInptSpikeRaster, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        # Hold the sate of network at any time step when updated
        aStateTimeSeries = []
        ltSpikeTimes = []
        liSpikeIDs = []

        # Local variables
        vState = self.vState
        vfVThresh = self.vfVThresh
        mfWIn = self.mfWIn
        vfVBias = self.vfVBias
        tDt = self.tDt
        nSizeIn = self.nSizeIn
        nSize = self.nSize
        vfVSubtract = self.vfVSubtract
        vfVReset = self.vfVReset

        # - Check type of mfWIn
        assert isinstance(mfWIn, CNNWeightTorch)
        # - Indices of neurons to be monitored
        vnIdMonitor = None if self.vnIdMonitor.size == 0 else self.vnIdMonitor
        # - Count number of spikes for each neuron in each time step
        vnNumSpikes = np.zeros(nSize, int)
        # - Time before first time step
        tCurrentTime = self.t

        if vnIdMonitor is not None:
            # Record initial state of the network
            self._add_to_record(aStateTimeSeries, tCurrentTime)

        # Iterate over all time steps
        for iCurrentTimeStep in tqdm(range(mfInptSpikeRaster.shape[0])):

            # - Spikes from input synapses
            vbInptSpikeRaster = mfInptSpikeRaster[iCurrentTimeStep]

            # Update neuron states
            vfUpdate = mfWIn[vbInptSpikeRaster]

            # State update (write this way to avoid that type casting fails)
            vState = vState + vfUpdate + vfVBias

            # - Update current time
            tCurrentTime += tDt

            if vnIdMonitor is not None:
                # - Record state before reset
                self._add_to_record(
                    aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor, vState=vState
                )

            # - Reset spike counter
            vnNumSpikes[:] = 0

            # - Check threshold crossings for spikes
            vbRecSpikeRaster = vState >= vfVThresh

            # - Reset or subtract from membrane state after spikes
            if vfVSubtract is not None:
                while vbRecSpikeRaster.any():
                    # - Subtract from states
                    vState[vbRecSpikeRaster] -= vfVSubtract[vbRecSpikeRaster]
                    # - Add to spike counter
                    vnNumSpikes[vbRecSpikeRaster] += 1
                    # - Neurons that are still above threshold will emit another spike
                    vbRecSpikeRaster = vState >= vfVThresh
            else:
                # - Add to spike counter
                vnNumSpikes = vbRecSpikeRaster.astype(int)
                # - Reset neuron states
                vState[vbRecSpikeRaster] = vfVReset[vbRecSpikeRaster]

            # - Record spikes
            ltSpikeTimes += [tCurrentTime] * np.sum(vnNumSpikes)
            liSpikeIDs += list(np.repeat(np.arange(nSize), vnNumSpikes))

            if vnIdMonitor is not None:
                # - Record state after reset
                self._add_to_record(
                    aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor, vState=vState
                )

        # - Update state
        self._vState = vState

        # Update time
        self._nTimeStep += nNumTimeSteps

        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            vtTimeTrace=ltSpikeTimes, vnChannels=liSpikeIDs, nNumChannels=self.nSize
        )

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        return tseOut


class TorchSpikingConv2dLayer(nn.Module):
    def __init__(self, kernel, strides, padding, img_data_format="channels_last"):
        """
        """
        super(TorchSpikingConv2dLayer, self).__init__()
