
###
# iaf_cl.py - Classes implementing feedforward and recurrent
#             layers consisting of I&F-neurons with constant
#             leak. Clock based.
###

import numpy as np
from typing import Optional, Union, List, Tuple
from tqdm import tqdm
from ...weights.internal.cnnweights import CNNWeight
from ...weights.internal.spiking_conv2d_torch import CNNWeightTorch
from ...timeseries import TSEvent
from abc import abstractmethod
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

        # - End time of evolution
        tFinal = self.t + nNumTimeSteps * self.tDt

        # - Extract spike timings and channels
        if tsInput is not None:
            # Extract spike data from the input variable
            __, __, mfSpikeRaster, __ = tsInput.raster(
                tDt=self.tDt,
                tStart=self.t,
                tStop=(self._nTimeStep + nNumTimeSteps) * self._tDt,
                # vnSelectChannels=np.arange(self.nSizeIn), ## This causes problems when tsInput has no events in some channels
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
            ), "`mfWIn` must have [{}] elements.".format(
                self.nSizeIn * self.nSize
            )
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
        nSizeIn = self.nSizeIn
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

        print(sum(mfInptSpikeRaster))

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

        # Update time
        self._nTimeStep += nNumTimeSteps

        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            vtTimeTrace=ltSpikeTimes,
            vnChannels=liSpikeIDs,
            nNumChannels=self.nSize,
            tStart=self.t,
            tStop=(self._nTimeStep + nNumTimeSteps) * tDt,
        )

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
        mfWRec = self.mfWRec
        mfWIn = self.mfWIn
        vfVBias = self.vfVBias
        tDt = self.tDt
        nSizeIn = self.nSizeIn
        nSize = self.nSize
        vfVSubtract = self.vfVSubtract
        vfVReset = self.vfVReset

        # - Check type of mfWIn
        bCNNWeights = isinstance(mfWIn, CNNWeight) or isinstance(mfWIn, CNNWeightTorch)
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
            vtTimeTrace=ltSpikeTimes,
            vnChannels=liSpikeIDs,
            nNumChannels=self.nSize,
            tStart=self.t,
            tStop=(self._nTimeStep + nNumTimeSteps) * tDt,
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
