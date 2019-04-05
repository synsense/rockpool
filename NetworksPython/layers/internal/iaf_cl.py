###
# iaf_cl.py - Classes implementing feedforward and recurrent
#             layers consisting of I&F-neurons with constant
#             leak. Clock based.
###

import numpy as np
from typing import Optional, Union, List, Tuple
from tqdm import tqdm
from collections import deque
from ...weights import CNNWeight, CNNWeightTorch
from ...timeseries import TSEvent, TSContinuous
from .. import Layer

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9

__all__ = ["FFCLIAF", "RecCLIAF"]


class CLIAF(Layer):
    """
    CLIAF - Abstract layer class of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        mfWIn: Union[np.ndarray, CNNWeight, CNNWeightTorch],
        vfVBias: Union[ArrayLike, float] = 0,
        vfVThresh: Union[ArrayLike, float] = 8,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVSubtract: Union[ArrayLike, float, None] = 8,
        tDt: float = 1,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        strName: str = "unnamed",
    ):
        """
        CLIAF - Feedforward layer of integrate and fire neurons with constant leak

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

    def _add_to_record(
        self,
        aStateTimeSeries: list,
        tCurrentTime: float,
        vnIdOut: Union[ArrayLike, bool] = True,
        vState: Optional[np.ndarray] = None,
        bDebug: bool = False,
    ):
        """
        _add_to_record: Convenience function to record current state of the layer
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
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:      TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:    float Duration of the desired evolution, in seconds
        :param nNumTimeSteps int Number of evolution time steps

        :return:
            mfSpikeRaster:    ndarray Boolean raster containing spike info
            nNumTimeSteps:    int Number of evlution time steps
        """
        print("Preparing input for processing")
        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Layer {}: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied".format(
                    self.strName
                )

                if tsInput.bPeriodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.tDuration

                else:
                    # - Evolve until the end of the input TImeSeries
                    tDuration = tsInput.tStop - self.t
                    assert tDuration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.strName
                        )
                        + "`tsInput` finishes before the current "
                        "evolution time."
                    )
            # - Discretize tDuration wrt self.tDt
            nNumTimeSteps = int((tDuration + fTolAbs) // self.tDt)
        else:
            assert isinstance(
                nNumTimeSteps, int
            ), "Layer `{}`: nNumTimeSteps must be of type int.".format(self.strName)

        # - Extract spike timings and channels
        if tsInput is not None:
            # Extract spike data from the input variable
            __, __, mfSpikeRaster = tsInput.raster(
                tDt=self.tDt,
                tStart=self.t,
                tStop=(self._nTimeStep + nNumTimeSteps) * self._tDt,
                vnSelectChannels=np.arange(self.nSizeIn),
            )
            # - Make sure size is correct
            mfSpikeRaster = mfSpikeRaster[:nNumTimeSteps, :]

        else:
            mfSpikeRaster = np.zeros((nNumTimeSteps, self.nSizeIn), bool)

        print("Done preparing input!")
        return mfSpikeRaster, nNumTimeSteps

    def reset_time(self):
        # - Set internal clock to 0
        self._nTimeStep = 0

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
    def mfWIn(self):
        return self._mfWIn

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        if isinstance(mfNewW, CNNWeight) or isinstance(mfNewW, CNNWeightTorch):
            assert mfNewW.shape == (self.nSizeIn, self.nSize)
            self._mfWIn = mfNewW
        else:
            assert (
                np.size(mfNewW) == self.nSizeIn * self.nSize
            ), "`mfWIn` must have [{}] elements.".format(self.nSizeIn * self.nSize)
            self._mfWIn = np.array(mfNewW).reshape(self.nSizeIn, self.nSize)

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
            self._vnIdMonitor = np.array(vnNewIDs)


class FFCLIAF(CLIAF):
    """
    FFCLIAF - Feedforward layer of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        mfW: Union[np.ndarray, CNNWeight],
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
        super().__init__(
            mfWIn=mfW,
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
        vState = self.vState.astype(np.float32)
        vfVThresh = self.vfVThresh
        mfWIn = self.mfWIn
        vfVBias = self.vfVBias
        tDt = self.tDt
        nSize = self.nSize
        vfVSubtract = self.vfVSubtract
        vfVReset = self.vfVReset

        # - Check type of mfWIn
        bCNNWeights = isinstance(mfWIn, CNNWeight) or isinstance(mfWIn, CNNWeightTorch)
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
            if bCNNWeights:
                # vfUpdate = mfWIn.reverse_dot(vbInptSpikeRaster) # This is too slow, only if network activity is super sparse
                vfUpdate = mfWIn[vbInptSpikeRaster]
            else:
                vfUpdate = vbInptSpikeRaster @ mfWIn

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
            np.set_printoptions(precision=4, suppress=True)

        # - Update state
        self._vState = vState

        # - Start and stop times for output time series
        tStart = self._nTimeStep * self.tDt
        tStop = (self._nTimeStep + nNumTimeSteps) * self.tDt

        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            vtTimeTrace=np.clip(
                ltSpikeTimes, tStart, tStop
            ),  # Clip due to possible numerical errors,
            vnChannels=liSpikeIDs,
            nNumChannels=self.nSize,
            tStart=tStart,
            tStop=tStop,
        )

        # Update time
        self._nTimeStep += nNumTimeSteps

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        return tseOut

    # - mfW as synonym for mfWIn
    @property
    def mfW(self):
        return self._mfWIn

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWIn = mfNewW


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
        vtRefractoryTime: Union[ArrayLike, float] = 0,
        tDt: float = 1e-4,
        tSpikeDelay: Optional[float] = None,
        tTauBias: Optional[float] = None,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        dtypeState: Union[type, str] = float,
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

        :param vtRefractoryTime: array-like Nx1 vector of refractory times.
        :param tDt:         float       time step size
        :param tSpikeDelay: float       Time after which a spike within the
                                        layer arrives at the recurrent
                                        synapses of the receiving neurons
                                        within the network. Rounded down to multiple of tDt.
                                        Must be at least tDt.
        :param tTauBias:    float       Period for applying bias. Must be at least tDt.
                                        Is rounded down to multiple of tDt.
                                        If None, will be set to tDt

        :vnIdMonitor:       array-like  IDs of neurons to be recorded

        :param dtypeState:  type data type for the membrane potential

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
        self.tTauBias = tDt if tTauBias is None else tTauBias
        self.vtRefractoryTime = vtRefractoryTime
        self.dtypeState = dtypeState

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

        # Lists for recording spikes
        lnTSSpikes = []
        liSpikeIDs = []

        # Local variables
        vState = self.vState
        vfVThresh = self.vfVThresh
        mfWRec = self.mfWRec
        mfWIn = self.mfWIn
        vfVBias = self.vfVBias
        nSize = self.nSize
        vfVSubtract = self.vfVSubtract
        vfVReset = self.vfVReset

        # - Check type of mfWIn
        bCNNWeights = isinstance(mfWIn, CNNWeight)

        # - Deque of arrays with number of delayed spikes for each neuron for each time step
        dqvnNumRecSpikes = self._dqvnNumRecSpikes
        # - Array for storing new recurrent spikes
        vnNumRecSpikes = np.zeros(self.nSize, int)

        # - For each neuron store number time steps until refractoriness ends
        vnTSUntilRefrEnds = self._vnTSUntilRefrEnds
        vnNumTSperRefractory = self._vnNumTSperRefractory

        # - Indices of neurons to be monitored
        vnIdMonitor = None if self.vnIdMonitor.size == 0 else self.vnIdMonitor

        # - Boolean array indicating evolution time steps where bias is applied
        vbBias = np.zeros(nNumTimeSteps)
        # - Determine where bias is applied: Index i corresponds to bias taking effect at
        #   nTimeStep = self._nTimeStep+1+i, want them when nTimeStep%_nNumTSperBias == 0
        vbBias[-(self._nTimeStep + 1) % self._nNumTSperBias :: self._nNumTSperBias] = 1

        # - State type dependent variables
        dtypeState = self.dtypeState
        nStateMin = self._nStateMin
        nStateMax = self._nStateMax

        if vnIdMonitor is not None:
            # States are recorded after update and after spike-triggered reset, i.e. twice per timestep
            mfRecord = np.zeros((2 * nNumTimeSteps + 1, vnIdMonitor.size))
            # Record initial state of the network
            mfRecord[0, :] = vState[vnIdMonitor]

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
            vbRefractory = vnTSUntilRefrEnds > 0
            vfUpdate[vbRefractory] = 0

            # State update (write this way to avoid that type casting fails)
            vState = np.clip(vState + vfUpdate, nStateMin, nStateMax).astype(dtypeState)

            if vnIdMonitor is not None:
                # - Record state before reset
                mfRecord[2 * iCurrentTimeStep + 1] = vState[vnIdMonitor]

            # - Check threshold crossings for spikes
            vbSpiking = vState >= vfVThresh

            # - Reset or subtract from membrane state after spikes
            if vfVSubtract is not None:  # - Subtract from potential
                if (
                    vnNumTSperRefractory == 0
                ).all():  # - No refractoriness - neurons can emit multiple spikes per time step
                    # - Reset recurrent spike counter
                    vnNumRecSpikes[:] = 0
                    while vbSpiking.any():
                        # - Add to spike counter
                        vnNumRecSpikes[vbSpiking] += 1
                        # - Subtract from states
                        vState[vbSpiking] = np.clip(
                            vState[vbSpiking] - vfVSubtract[vbSpiking],
                            nStateMin,
                            nStateMax,
                        ).astype(dtypeState)
                        # - Neurons that are still above threshold will emit another spike
                        vbSpiking = vState >= vfVThresh
                else:  # With refractoriness, at most one spike per time step is possible
                    # - Add to spike counter
                    vnNumRecSpikes = vbSpiking.astype(int)
                    # - Reset neuron states
                    vState[vbSpiking] = np.clip(
                        vState[vbSpiking] - vfVSubtract[vbSpiking], nStateMin, nStateMax
                    ).astype(dtypeState)
            else:  # - Reset potential
                # - Add to spike counter
                vnNumRecSpikes = vbSpiking.astype(int)
                # - Reset neuron states
                vState[vbSpiking] = np.clip(
                    vfVReset[vbSpiking], nStateMin, nStateMax
                ).astype(dtypeState)

            if (vnNumTSperRefractory > 0).any():
                # - Update refractoryness
                vnTSUntilRefrEnds = np.clip(vnTSUntilRefrEnds - 1, 0, None)
                vnTSUntilRefrEnds[vbSpiking] = vnNumTSperRefractory[vbSpiking]

            # - Store recurrent spikes in deque
            dqvnNumRecSpikes.append(vnNumRecSpikes)

            # - Record spikes
            lnTSSpikes += [iCurrentTimeStep] * np.sum(vnNumRecSpikes)
            liSpikeIDs += list(np.repeat(np.arange(nSize), vnNumRecSpikes))

            if vnIdMonitor is not None:
                # - Record state after reset
                mfRecord[2 * iCurrentTimeStep + 2] = vState[vnIdMonitor]

        # - Store IDs of neurons that would spike in furute time steps
        self._dqvnNumRecSpikes = dqvnNumRecSpikes

        # - Store refractoriness of neurons
        self._vnTSUntilRefrEnds = vnTSUntilRefrEnds

        # - Start and stop times for output time series
        tStart = self._nTimeStep * self.tDt
        tStop = (self._nTimeStep + nNumTimeSteps) * self.tDt

        # Generate output sime series
        vtSpikeTimes = (np.array(lnTSSpikes) + 1 + self._nTimeStep) * self.tDt
        tseOut = TSEvent(
            # Clip due to possible numerical errors,
            vtTimeTrace=np.clip(vtSpikeTimes, tStart, tStop),
            vnChannels=liSpikeIDs,
            nNumChannels=self.nSize,
            tStart=tStart,
            tStop=tStop,
        )

        if vnIdMonitor is not None:
            # - Store recorded data in timeseries
            vtRecordTimes = np.repeat(
                (self._nTimeStep + np.arange(nNumTimeSteps + 1)) * self.tDt, 2
            )[1:]
            self.tscRecorded = TSContinuous(vtRecordTimes, mfRecord)

        # Update time
        self._nTimeStep += nNumTimeSteps

        # - Update state
        self._vState = vState

        return tseOut

    def reset_state(self):
        # - Delete spikes that would arrive in recurrent synapses during future time steps
        #   by filling up deque with zeros
        nNumTSperDelay = self._dqvnNumRecSpikes.maxlen
        self._dqvnNumRecSpikes += [np.zeros(self.nSize) for _ in range(nNumTSperDelay)]
        # - Reset refractoriness
        self._vnTSUntilRefrEnds = np.zeros(self.nSize, int)
        # - Reset neuron state to self.vfVReset
        self.vState = np.clip(self.vfVReset, self._nStateMin, self._nStateMax).astype(
            self.dtypeState
        )

    def randomize_state(self):
        # - Set state to random values between reset value and theshold
        self.vState = np.clip(
            (np.amin(self.vfVThresh) - np.amin(self.vfVReset))
            * np.random.rand(self.nSize)
            - np.amin(self.vfVReset),
            self._nStateMin,
            self._nStateMax,
        ).astype(self.dtypeState)

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
        ), "Layer `{}`: tTauBias must be a scalar greater than tDt ({})".format(
            self.strName, self.tDt
        )
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
            ), "Layer `{}`: tSpikeDelay must be a scalar greater than tDt ({})".format(
                self.strName, self.tDt
            )
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
                    maxlen=nNumTSperDelay,
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
                maxlen=nNumTSperDelay,
            )

    @property
    def vtRefractoryTime(self):
        return (
            None
            if self._vnNumTSperRefractory is None
            else self._vnNumTSperRefractory * self.tDt
        )

    @vtRefractoryTime.setter
    def vtRefractoryTime(self, vtNewTime):
        if vtNewTime is None:
            self._vnNumTSperRefractory = None
        else:
            vtRefractoryTime = self._expand_to_net_size(vtNewTime, "vtRefractoryTime")
            # - vtRefractoryTime is rounded to multiple of tDt and at least tDt
            self._vnNumTSperRefractory = (np.floor(vtRefractoryTime / self.tDt)).astype(
                int
            )

    @property
    def vState(self):
        return self._vState

    @vState.setter
    def vState(self, vNewState):
        self._vState = np.clip(
            self._expand_to_net_size(vNewState, "vState"),
            self._nStateMin,
            self._nStateMax,
        ).astype(self.dtypeState)

    @property
    def dtypeState(self):
        return self._dtypeState

    @dtypeState.setter
    def dtypeState(self, dtypeNew):
        if np.issubdtype(dtypeNew, np.integer):
            # - Set limits for integer type states
            self._nStateMin = np.iinfo(dtypeNew).min
            self._nStateMax = np.iinfo(dtypeNew).max
        elif np.issubdtype(dtypeNew, np.floating):
            self._nStateMin = np.finfo(dtypeNew).min
            self._nStateMax = np.finfo(dtypeNew).max
        else:
            raise ValueError(
                "Layer `{}`: dtypeState must be integer or float data type.".format(
                    self.strName
                )
            )
        self._dtypeState = dtypeNew
        # - Convert vState to dtype
        if hasattr(self, "_vState"):
            self.vState = self.vState
