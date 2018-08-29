###
# iaf_cl.py - Class implementing a recurrent layer consisting of
#                 I&F-neurons with constant leak. Clock based.
###

from typing import Optional, Union, List, Tuple

import numpy as np
from tqdm import tqdm
from ..cnnweights import CNNWeight
from ...timeseries import TSEvent
from .. import CLIAF


# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9
# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class RecCLIAF(CLIAF):
    """
    RecCLIAF - Recurrent layer of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        mfWIn: Union[np.ndarray, CNNWeight],
        mfWRec: np.ndarray,
        vfVBias: Union[ArrayLike, float] = 0,
        vfVThresh: Union[ArrayLike, float] = 8,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVSubtract: Union[ArrayLike, float, None] = 8,
        tDt: float = 1,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        strName: str = "unnamed",
    ):
        """
        RecCLIAF - Recurrent layer of integrate and fire neurons with constant leak

        :param mfWIn:       array-like  nSizeInxN input weight matrix.
        :param mfWRec:      array-like  Weight matrix
        :param vfVBias:     array-like  Constant bias to be added to state at each time step
        :param vfVThresh:   array-like  Spiking threshold
        :param vfVReset:    array-like  Reset potential after spike (also see param bSubtract)
        :param vfVSubtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.
        :vnIdMonitor:       array-like  IDs of neurons to be recorded
        :param strName:     str  Name of this layer.
        """

        # Call parent constructor
        super().__init__(
            mfWIn=mfWIn,
            vfVBias=vfVBias,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVSubtract=vfVSubtract,
            tDt=tDt,
            vnIdMonitor=vnIdMonitor,
            strName=strName,
        )

        # - Set recurrent weights
        self.mfWRec = mfWRec

        self.reset_state()

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> (TSEvent, np.ndarray):
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
        mfWRec = self.mfWRec
        mfWIn = self.mfWIn
        vfVBias = self.vfVBias
        tDt = self.tDt
        nSizeIn = self.nSizeIn
        nSize = self.nSize
        vfVSubtract = self.vfVSubtract
        vfVReset = self.vfVReset

        # - Check type of mfWIn
        bCNNWeights = isinstance(mfWIn, CNNWeight)
        # - Number of spike sources (input neurons and layer neurons)
        nSpikeSources = self.nSizeIn + self.nSize
        # - Count number of spikes for each neuron in each time step
        vnNumRecSpikes = self._vnNumRecSpikes
        # - Indices of neurons to be monitored
        vnIdMonitor = None if self.vnIdMonitor.size == 0 else self.vnIdMonitor
        # - Time before first time step
        tCurrentTime = self.t

        if vnIdMonitor is not None:
            # Record initial state of the network
            self._add_to_record(aStateTimeSeries, self.t)

        # Iterate over all time steps
        for iCurrentTimeStep in tqdm(range(mfInptSpikeRaster.shape[0])):

            # - Spikes from input synapses
            vbInptSpikeRaster = mfInptSpikeRaster[iCurrentTimeStep]

            # Update neuron states
            if bCNNWeights:
                vfUpdate = mfWIn.reverse_dot(vbInptSpikeRaster) + (
                    vnNumRecSpikes @ mfWRec
                )
            else:
                vfUpdate = (vbInptSpikeRaster @ mfWIn) + (vnNumRecSpikes @ mfWRec)

            # State update (write this way to avoid that type casting fails)
            vState = vState + vfUpdate + vfVBias

            # - Update current time
            tCurrentTime += tDt

            if vnIdMonitor is not None:
                # - Record state before reset
                self._add_to_record(
                    aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor, vState=vState
                )

            # - Reset recurrent spike counter
            vnNumRecSpikes[:] = 0

            # - Check threshold crossings for spikes
            vbRecSpikeRaster = vState >= vfVThresh

            # - Reset or subtract from membrane state after spikes
            if vfVSubtract is not None:
                while vbRecSpikeRaster.any():
                    # - Subtract from states
                    vState[vbRecSpikeRaster] -= vfVSubtract[vbRecSpikeRaster]
                    # - Add to spike counter
                    vnNumRecSpikes[vbRecSpikeRaster] += 1
                    # - Neurons that are still above threshold will emit another spike
                    vbRecSpikeRaster = vState >= vfVThresh
            else:
                # - Add to spike counter
                vnNumRecSpikes = vbRecSpikeRaster.astype(int)
                # - Reset neuron states
                vState[vbRecSpikeRaster] = vfVReset[vbRecSpikeRaster]

            # - Record spikes
            ltSpikeTimes += [tCurrentTime] * np.sum(vnNumRecSpikes)
            liSpikeIDs += list(np.repeat(np.arange(nSize), vnNumRecSpikes))

            if vnIdMonitor is not None:
                # - Record state after reset
                self._add_to_record(
                    aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor, vState=vState
                )

        # - Update state
        self._vState = vState

        # - Store IDs of neurons that would spike in next time step
        self._vnNumRecSpikes = vnNumRecSpikes

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

    def reset_state(self):
        # - Delete spikes that would arrive in recurrent synapses during next time step
        self._vnNumRecSpikes = np.zeros(self.nSize, int)
        # - Reset neuron state to 0
        self._vState = self.vfVReset

    ### --- Properties

    @property
    def mfW(self):
        return self.mfWRec

    # - mfW as synonym for mfWRec
    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWRec = mfNewW

    @property
    def mfWRec(self):
        return self._mfWRec

    @mfWRec.setter
    def mfWRec(self, mfNewW):
        self._mfWRec = self._expand_to_weight_size(mfNewW, "mfWRec", bAllowNone=False)
