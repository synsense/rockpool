###
# iaf_cl_extd.py - Extended version of RecCLIAF
###

from typing import Optional, Union, List, Tuple

import numpy as np
from tqdm import tqdm
from collections import deque
from ..cnnweights import CNNWeight
from ...timeseries import TSEvent
from .. import CLIAF


# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9
# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class RecCLIAFExtd(CLIAF):
    """
    RecCLIAFExtd - Extended version of RecCLIAF
    """

    def __init__(
        self,
        mfWIn: Union[np.ndarray, CNNWeight],
        mfWRec: np.ndarray,
        vfVBias: Union[ArrayLike, float] = 0,
        vfVThresh: Union[ArrayLike, float] = 8,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVSubtract: Union[ArrayLike, float, None] = 8,
        vtRefractoryTime: Union[ArrayLike, float] = 0,
        tDt: float = 1e-4,
        tSpikeDelay: Optional[float] = None,
        tTauBias: float = 1e-4,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        strName: str = "unnamed",
    ):
        """
        RecCLIAFExtd - Extended version of RecCLIAF

        :param mfWIn:       array-like  nSizeInxN input weight matrix.
        :param mfWRec:      array-like  Weight matrix
        :param vfVBias:     array-like  Constant bias to be added to state at each time step
        :param vfVThresh:   array-like  Spiking threshold
        :param vfVReset:    array-like  Reset potential after spike (also see param bSubtract)
        :param vfVSubtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.
        :param vtRefractoryTime: array-like Nx1 vector of refractory times.
        :param tDt:         float       time step size
        :param tSpikeDelay: float       Time after which a spike within the
                                        layer arrives at the recurrent
                                        synapses of the receiving neurons
                                        within the network. Rounded down to multiple of tDt.
                                        Must be at least tDt.
        :param tTauBias:    float       Perioud for applying bias. Must be at least tDt.
                                        Is rounded down to multiple of tDt.
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
        self.tSpikeDelay = tSpikeDelay
        self.tTauBias = tTauBias
        self.vtRefractoryTime = vtRefractoryTime

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
        :param bVerbose:        bool     Show progress bar during evolution
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
        tTauBias = self.tTauBias
        

        # - Check type of mfWIn
        bCNNWeights = isinstance(mfWIn, CNNWeight)

        # - Deque of arrays with number of delayed spikes for each neuron for each time step
        dqvnNumRecSpikes = self._dqvnNumRecSpikes
        # - Array for storing new recurrent spikes
        vnNumRecSpikes = np.zeros(self.nSize)

        # - For each neuron store number time steps until refractoriness ends
        vnTSRefractoryEnds = np.zeros(self.nSize)
        vnNumTSperRefractory = self._vnNumTSperRefractory

        # - Indices of neurons to be monitored
        vnIdMonitor = None if self.vnIdMonitor.size == 0 else self.vnIdMonitor
        # - Time before first time step
        tCurrentTime = self.t

        # - Boolean array indicating evolution time steps where bias is applied
        vbBias = np.zeros(nNumTimeSteps)
        # - Determine where bias is applied: Index i corresponds to bias taking effect at
        #   nTimeStep = self._nTimeStep+1+i, want them when nTimeStep%_nNumTSperBias == 0
        vbBias[-(self._nTimeStep+1) % self._nNumTSperBias :: self._nNumTSperBias] = 1

        if vnIdMonitor is not None:
            # Record initial state of the network
            self._add_to_record(aStateTimeSeries, self.t)

        if bVerbose:
            rangeIterator = tqdm(range(nNumTimeSteps))
        else:
            rangeIterator = range(nNumTimeSteps)

        # Iterate over all time steps
        for iCurrentTimeStep in rangeIterator:

            # - Spikes from input synapses
            vbInptSpikeRaster = mfInptSpikeRaster[iCurrentTimeStep]

            # Update neuron states
            if bCNNWeights:
                vfUpdate = (
                    mfWIn[vbInptSpikeRaster]  # Input spikes
                    + (dqvnNumRecSpikes.popleft() @ mfWRec)  # Recurrent spikes
                    + (vbBias[iCurrentTimeStep] * vfVBias)  # Bias
                )
            else:
                vfUpdate = (
                    (vbInptSpikeRaster @ mfWIn)  # Input spikes
                    + (dqvnNumRecSpikes.popleft() @ mfWRec)  # Recurrent spikes
                    + (vbBias[iCurrentTimeStep] * vfVBias)  # Bias
                )

            # - Only neurons that are not refractory can receive inputs and be updated
            vbRefractory = vnTSRefractoryEnds > 0
            vfUpdate[vbRefractory] = 0

            # State update (write this way to avoid that type casting fails)
            vState = vState + vfUpdate

            # - Update current time
            tCurrentTime += tDt

            if vnIdMonitor is not None:
                # - Record state before reset
                self._add_to_record(
                    aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor, vState=vState
                )

            # - Check threshold crossings for spikes
            vbSpiking = vState >= vfVThresh

            # - Reset or subtract from membrane state after spikes
            if vfVSubtract is not None:  # - Subtract from potential
                if (vnNumTSperRefractory == 0).all():  # - No refractoriness - neurons can emit multiple spikes per time step
                    # - Reset recurrent spike counter
                    vnNumRecSpikes[:] = 0
                    while vbSpiking.any():
                        # - Add to spike counter
                        vnNumRecSpikes[vbSpiking] += 1
                        # - Subtract from states
                        vState[vbSpiking] -= vfVSubtract[vbSpiking]
                        # - Neurons that are still above threshold will emit another spike
                        vbSpiking = vState >= vfVThresh
                else:  # With refractoriness, at most one spike per time step is possible
                    # - Add to spike counter
                    vnNumRecSpikes = vbSpiking.astype(int)
                    # - Reset neuron states
                    vState[vbSpiking] -= vfVSubtract[vbSpiking]              
            else:  # - Reset potential
                # - Add to spike counter
                vnNumRecSpikes = vbSpiking.astype(int)
                # - Reset neuron states
                vState[vbSpiking] = vfVReset[vbSpiking]

            if (vnNumTSperRefractory > 0).any():
                # - Update refractoryness
                vnTSRefractoryEnds = np.clip(vnTSRefractoryEnds-1, 0, None)
                vnTSRefractoryEnds[vbSpiking] = vnNumTSperRefractory[vbSpiking]

            # - Store recurrent spikes in deque
            dqvnNumRecSpikes.append(vnNumRecSpikes)

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
        # - Delete spikes that would arrive in recurrent synapses during future time steps
        #   by filling up deque with zeros
        nNumTSperDelay = self._dqvnNumRecSpikes.maxlen
        self._dqvnNumRecSpikes += [np.zeros(self.nSize) for _ in range(nNumTSperDelay)]
        # - Reset neuron state to 0
        self._vState = self.vfVReset
        # - Reset 

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

    @property
    def tTauBias(self):
        return self._nNumTSperBias * self._tDt

    @tTauBias.setter
    def tTauBias(self, tNewBias):
        assert (
            np.isscalar(tNewBias) and tNewBias >= self.tDt
        ), "Layer `{}`: tTauBias must be a scalar greater than tDt ({})".format(self.strName, self.tDt)
        # - tNewBias is rounded to multiple of tDt and at least tDt
        self._nNumTSperBias = int(np.floor(tNewBias / self.tDt))

    @property
    def tSpikeDelay(self):
        return self._dqvnNumRecSpikes.maxlen * self._tDt

    @tSpikeDelay.setter
    def tSpikeDelay(self, tNewDelay):
        if tNewDelay is None:
            nNumTSperDelay = 1
        else:
            assert (
                np.isscalar(tNewDelay) and tNewDelay >= self.tDt
            ), "Layer `{}`: tSpikeDelay must be a scalar greater than tDt ({})".format(self.strName, self.tDt)
            # - tNewDelay is rounded to multiple of tDt and at least tDt
            nNumTSperDelay = int(np.floor(tNewDelay / self.tDt))

        ## -- Create a deque for holding delayed spikes
        # - Copy spikes from previous deque
        if hasattr(self, "_dqvnNumRecSpikes"):
            lPrevSpikes = list(self._dqvnNumRecSpikes)
            nDifference = nNumTSperDelay - len(lPrevSpikes)
            # - If new delay is less, some spikes will be lost
            self._dqvnNumRecSpikes = deque(lPrevSpikes, maxlen=nNumTSperDelay)
            if nDifference >= 0:    
                self._dqvnNumRecSpikes = deque(
                    [np.zeros(self.nSize) for _ in range(nDifference)] + lPrevSpikes,
                    maxlen=nNumTSperDelay
                )
            else:
                self._dqvnNumRecSpikes = deque(lPrevSpikes, maxlen=nNumTSperDelay)
                print(
                    "Layer `{}`: Last {} spikes in buffer have been lost due to reduction of tSpikeDelay.".format(
                        self.strName, np.sum(np.array(lPrevSpikes[:-nDifference]))
                    )
                )
        else:
            self._dqvnNumRecSpikes = deque(
                [np.zeros(self.nSize) for _ in range(nNumTSperDelay)],
                maxlen=nNumTSperDelay
            )


    @property
    def vtRefractoryTime(self):
        return (
            None if self._vnNumTSperRefractory is None
            else self._vnNumTSperRefractory * self.tDt
        )
        
    @vtRefractoryTime.setter
    def vtRefractoryTime(self, vtNewTime):
        if vtNewTime is None:
            self._vnNumTSperRefractory = None
        else:
            vtRefractoryTime = self._expand_to_net_size(vtNewTime, "vtRefractoryTime")           
            # - vtRefractoryTime is rounded to multiple of tDt and at least tDt
            self._vnNumTSperRefractory = (np.floor(vtRefractoryTime / self.tDt)).astype(int)