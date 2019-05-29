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
tol_abs = 1e-9

__all__ = ["FFCLIAF", "RecCLIAF"]


class CLIAF(Layer):
    """
    CLIAF - Abstract layer class of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        weights_in: Union[np.ndarray, CNNWeight, CNNWeightTorch],
        vfVBias: Union[ArrayLike, float] = 0,
        vfVThresh: Union[ArrayLike, float] = 8,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVSubtract: Union[ArrayLike, float, None] = 8,
        dt: float = 1,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        name: str = "unnamed",
    ):
        """
        CLIAF - Feedforward layer of integrate and fire neurons with constant leak

        :param weights_in:       array-like  Input weight matrix
        :param vfVBias:     array-like  Constant bias to be added to state at each time step
        :param vfVThresh:   array-like  Spiking threshold
        :param vfVReset:    array-like  Reset potential after spike (also see param bSubtract)
        :param vfVSubtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.
        :vnIdMonitor:       array-like  IDs of neurons to be recorded
        :param name:     str  Name of this layer.
        """

        # Call parent constructor
        super().__init__(weights=weights_in, dt=dt, name=name)

        # - Set neuron parameters
        self.weights_in = weights_in
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
        state: Optional[np.ndarray] = None,
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
        :param state:           np.ndarray If not None, record this as state,
                                            otherwise self.state
        :param bDebug:           bool Print debug info
        """

        state = self.state if state is None else state

        if vnIdOut is True:
            vnIdOut = np.arange(self.size)
        elif vnIdOut is False:
            # - Do nothing
            return

        # Update record of state changes
        for nIdOutIter in np.asarray(vnIdOut):
            aStateTimeSeries.append([tCurrentTime, nIdOutIter, state[nIdOutIter]])
            if bDebug:
                print([tCurrentTime, nIdOutIter, state[nIdOutIter, 0]])

    def _prepare_input(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, int):
        """
        _prepare_input - Sample input, set up time base

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            mfSpikeRaster:    ndarray Boolean raster containing spike info
            num_timesteps:    int Number of evlution time steps
        """
        print("Preparing input for processing")
        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer {}: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Layer {}: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + "`ts_input` finishes before the current "
                        "evolution time."
                    )
            # - Discretize duration wrt self.dt
            num_timesteps = int((duration + tol_abs) // self.dt)
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)

        # - Extract spike timings and channels
        if ts_input is not None:
            # Extract spike data from the input variable
            mfSpikeRaster = ts_input.raster(
                dt=self.dt,
                t_start=self.t,
                t_stop=(self._timestep + num_timesteps) * self._dt,
                channels=np.arange(self.size_in),
            )
            # - Make sure size is correct
            mfSpikeRaster = mfSpikeRaster[:num_timesteps, :]

        else:
            mfSpikeRaster = np.zeros((num_timesteps, self.size_in), bool)

        print("Done preparing input!")
        return mfSpikeRaster, num_timesteps

    def reset_time(self):
        # - Set internal clock to 0
        self._timestep = 0

    def reset_state(self):
        # - Reset neuron state to 0
        self._state = self.vfVReset

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @property
    def input_type(self):
        return TSEvent

    @property
    def weights_in(self):
        return self._weights_in

    @weights_in.setter
    def weights_in(self, mfNewW):
        if isinstance(mfNewW, CNNWeight) or isinstance(mfNewW, CNNWeightTorch):
            assert mfNewW.shape == (self.size_in, self.size)
            self._weights_in = mfNewW
        else:
            assert (
                np.size(mfNewW) == self.size_in * self.size
            ), "`weights_in` must have [{}] elements.".format(self.size_in * self.size)
            self._weights_in = np.array(mfNewW).reshape(self.size_in, self.size)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, vNewState):
        self._state = self._expand_to_net_size(vNewState, "state", bAllowNone=False)

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

    @Layer.dt.setter
    def dt(self, tNewDt):
        assert tNewDt > 0, "dt must be greater than 0."
        self._dt = tNewDt

    @property
    def vnIdMonitor(self):
        return self._vnIdMonitor

    @vnIdMonitor.setter
    def vnIdMonitor(self, vnNewIDs):
        if vnNewIDs is True:
            self._vnIdMonitor = np.arange(self.size)
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
        weights: Union[np.ndarray, CNNWeight],
        vfVBias: Union[ArrayLike, float] = 0,
        vfVThresh: Union[ArrayLike, float] = 8,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVSubtract: Union[ArrayLike, float, None] = 8,
        dt: float = 1,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        name: str = "unnamed",
    ):
        """
        FFCLIAF - Feedforward layer of integrate and fire neurons with constant leak

        :param weights:         array-like  Input weight matrix
        :param vfVBias:     array-like  Constant bias to be added to state at each time step
        :param vfVThresh:   array-like  Spiking threshold
        :param vfVReset:    array-like  Reset potential after spike (also see param bSubtract)
        :param vfVSubtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.
        :vnIdMonitor:       array-like  IDs of neurons to be recorded
        :param name:     str  Name of this layer.
        """

        # Call parent constructor
        super().__init__(
            weights_in=weights,
            vfVBias=vfVBias,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVSubtract=vfVSubtract,
            dt=dt,
            vnIdMonitor=vnIdMonitor,
            name=name,
        )

        self.reset_state()

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Generate input in rasterized form, get actual evolution duration
        mfInptSpikeRaster, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # Hold the sate of network at any time step when updated
        aStateTimeSeries = []
        ltSpikeTimes = []
        liSpikeIDs = []

        # Local variables
        state = self.state.astype(np.float32)
        vfVThresh = self.vfVThresh
        weights_in = self.weights_in
        vfVBias = self.vfVBias
        dt = self.dt
        size = self.size
        vfVSubtract = self.vfVSubtract
        vfVReset = self.vfVReset

        # - Check type of weights_in
        bCNNWeights = isinstance(weights_in, CNNWeight) or isinstance(weights_in, CNNWeightTorch)
        # - Indices of neurons to be monitored
        vnIdMonitor = None if self.vnIdMonitor.size == 0 else self.vnIdMonitor
        # - Count number of spikes for each neuron in each time step
        vnNumSpikes = np.zeros(size, int)
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
                # vfUpdate = weights_in.reverse_dot(vbInptSpikeRaster) # This is too slow, only if network activity is super sparse
                vfUpdate = weights_in[vbInptSpikeRaster]
            else:
                vfUpdate = vbInptSpikeRaster @ weights_in

            # State update (write this way to avoid that type casting fails)
            state = state + vfUpdate + vfVBias

            # - Update current time
            tCurrentTime += dt

            if vnIdMonitor is not None:
                # - Record state before reset
                self._add_to_record(
                    aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor, state=state
                )

            # - Reset spike counter
            vnNumSpikes[:] = 0

            # - Check threshold crossings for spikes
            vbRecSpikeRaster = state >= vfVThresh

            # - Reset or subtract from membrane state after spikes
            if vfVSubtract is not None:
                while vbRecSpikeRaster.any():
                    # - Subtract from states
                    state[vbRecSpikeRaster] -= vfVSubtract[vbRecSpikeRaster]
                    # - Add to spike counter
                    vnNumSpikes[vbRecSpikeRaster] += 1
                    # - Neurons that are still above threshold will emit another spike
                    vbRecSpikeRaster = state >= vfVThresh
            else:
                # - Add to spike counter
                vnNumSpikes = vbRecSpikeRaster.astype(int)
                # - Reset neuron states
                state[vbRecSpikeRaster] = vfVReset[vbRecSpikeRaster]

            # - Record spikes
            ltSpikeTimes += [tCurrentTime] * np.sum(vnNumSpikes)
            liSpikeIDs += list(np.repeat(np.arange(size), vnNumSpikes))

            if vnIdMonitor is not None:
                # - Record state after reset
                self._add_to_record(
                    aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor, state=state
                )
            np.set_printoptions(precision=4, suppress=True)

        # - Update state
        self._state = state

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            times=np.clip(
                ltSpikeTimes, t_start, t_stop
            ),  # Clip due to possible numerical errors,
            channels=liSpikeIDs,
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

        # Update time
        self._timestep += num_timesteps

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        return tseOut

    # - weights as synonym for weights_in
    @property
    def weights(self):
        return self._weights_in

    @weights.setter
    def weights(self, mfNewW):
        self.weights_in = mfNewW


class RecCLIAF(CLIAF):
    """
    RecCLIAF - Recurrent layer of integrate and fire neurons with constant leak
    """

    def __init__(
        self,
        weights_in: Union[np.ndarray, CNNWeight],
        weights_rec: np.ndarray,
        vfVBias: Union[ArrayLike, float] = 0,
        vfVThresh: Union[ArrayLike, float] = 8,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVSubtract: Union[ArrayLike, float, None] = 8,
        vtRefractoryTime: Union[ArrayLike, float] = 0,
        dt: float = 1e-4,
        tSpikeDelay: Optional[float] = None,
        tTauBias: Optional[float] = None,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        dtypeState: Union[type, str] = float,
        name: str = "unnamed",
    ):
        """
        RecCLIAF - Recurrent layer of integrate and fire neurons with constant leak

        :param weights_in:       array-like  nSizeInxN input weight matrix.
        :param weights_rec:      array-like  Weight matrix

        :param vfVBias:     array-like  Constant bias to be added to state at each time step
        :param vfVThresh:   array-like  Spiking threshold
        :param vfVReset:    array-like  Reset potential after spike (also see param bSubtract)
        :param vfVSubtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.

        :param vtRefractoryTime: array-like Nx1 vector of refractory times.
        :param dt:         float       time step size
        :param tSpikeDelay: float       Time after which a spike within the
                                        layer arrives at the recurrent
                                        synapses of the receiving neurons
                                        within the network. Rounded down to multiple of dt.
                                        Must be at least dt.
        :param tTauBias:    float       Period for applying bias. Must be at least dt.
                                        Is rounded down to multiple of dt.
                                        If None, will be set to dt

        :vnIdMonitor:       array-like  IDs of neurons to be recorded

        :param dtypeState:  type data type for the membrane potential

        :param name:     str  Name of this layer.
        """

        # Call parent constructor
        super().__init__(
            weights_in=weights_in,
            vfVBias=vfVBias,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVSubtract=vfVSubtract,
            dt=dt,
            vnIdMonitor=vnIdMonitor,
            name=name,
        )

        # - Set recurrent weights
        self.weights_rec = weights_rec
        self.tSpikeDelay = tSpikeDelay
        self.tTauBias = dt if tTauBias is None else tTauBias
        self.vtRefractoryTime = vtRefractoryTime
        self.dtypeState = dtypeState

        self.reset_state()

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Show progress bar during evolution
        :return:            TSEvent  output spike series

        """

        # - Generate input in rasterized form, get actual evolution duration
        mfInptSpikeRaster, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # Lists for recording spikes
        lnTSSpikes = []
        liSpikeIDs = []

        # Local variables
        state = self.state
        vfVThresh = self.vfVThresh
        weights_rec = self.weights_rec
        weights_in = self.weights_in
        vfVBias = self.vfVBias
        size = self.size
        vfVSubtract = self.vfVSubtract
        vfVReset = self.vfVReset

        # - Check type of weights_in
        bCNNWeights = isinstance(weights_in, CNNWeight)

        # - Deque of arrays with number of delayed spikes for each neuron for each time step
        dqvnNumRecSpikes = self._dqvnNumRecSpikes
        # - Array for storing new recurrent spikes
        vnNumRecSpikes = np.zeros(self.size, int)

        # - For each neuron store number time steps until refractoriness ends
        vnTSUntilRefrEnds = self._vnTSUntilRefrEnds
        vnNumTSperRefractory = self._vnNumTSperRefractory

        # - Indices of neurons to be monitored
        vnIdMonitor = None if self.vnIdMonitor.size == 0 else self.vnIdMonitor

        # - Boolean array indicating evolution time steps where bias is applied
        vbBias = np.zeros(num_timesteps)
        # - Determine where bias is applied: Index i corresponds to bias taking effect at
        #   nTimeStep = self._timestep+1+i, want them when nTimeStep%_nNumTSperBias == 0
        vbBias[-(self._timestep + 1) % self._nNumTSperBias :: self._nNumTSperBias] = 1

        # - State type dependent variables
        dtypeState = self.dtypeState
        nStateMin = self._nStateMin
        nStateMax = self._nStateMax

        if vnIdMonitor is not None:
            # States are recorded after update and after spike-triggered reset, i.e. twice per _timestep
            mfRecord = np.zeros((2 * num_timesteps + 1, vnIdMonitor.size))
            # Record initial state of the network
            mfRecord[0, :] = state[vnIdMonitor]

        if verbose:
            rangeIterator = tqdm(range(num_timesteps))
        else:
            rangeIterator = range(num_timesteps)

        # Iterate over all time steps
        for iCurrentTimeStep in rangeIterator:

            # - Spikes from input synapses
            vbInptSpikeRaster = mfInptSpikeRaster[iCurrentTimeStep]

            # Update neuron states
            if bCNNWeights:
                vfUpdate = (
                    weights_in[vbInptSpikeRaster]  # Input spikes
                    + (dqvnNumRecSpikes.popleft() @ weights_rec)  # Recurrent spikes
                    + (vbBias[iCurrentTimeStep] * vfVBias)  # Bias
                )
            else:
                vfUpdate = (
                    (vbInptSpikeRaster @ weights_in)  # Input spikes
                    + (dqvnNumRecSpikes.popleft() @ weights_rec)  # Recurrent spikes
                    + (vbBias[iCurrentTimeStep] * vfVBias)  # Bias
                )

            # - Only neurons that are not refractory can receive inputs and be updated
            vbRefractory = vnTSUntilRefrEnds > 0
            vfUpdate[vbRefractory] = 0

            # State update (write this way to avoid that type casting fails)
            state = np.clip(state + vfUpdate, nStateMin, nStateMax).astype(dtypeState)

            if vnIdMonitor is not None:
                # - Record state before reset
                mfRecord[2 * iCurrentTimeStep + 1] = state[vnIdMonitor]

            # - Check threshold crossings for spikes
            vbSpiking = state >= vfVThresh

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
                        state[vbSpiking] = np.clip(
                            state[vbSpiking] - vfVSubtract[vbSpiking],
                            nStateMin,
                            nStateMax,
                        ).astype(dtypeState)
                        # - Neurons that are still above threshold will emit another spike
                        vbSpiking = state >= vfVThresh
                else:  # With refractoriness, at most one spike per time step is possible
                    # - Add to spike counter
                    vnNumRecSpikes = vbSpiking.astype(int)
                    # - Reset neuron states
                    state[vbSpiking] = np.clip(
                        state[vbSpiking] - vfVSubtract[vbSpiking], nStateMin, nStateMax
                    ).astype(dtypeState)
            else:  # - Reset potential
                # - Add to spike counter
                vnNumRecSpikes = vbSpiking.astype(int)
                # - Reset neuron states
                state[vbSpiking] = np.clip(
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
            liSpikeIDs += list(np.repeat(np.arange(size), vnNumRecSpikes))

            if vnIdMonitor is not None:
                # - Record state after reset
                mfRecord[2 * iCurrentTimeStep + 2] = state[vnIdMonitor]

        # - Store IDs of neurons that would spike in furute time steps
        self._dqvnNumRecSpikes = dqvnNumRecSpikes

        # - Store refractoriness of neurons
        self._vnTSUntilRefrEnds = vnTSUntilRefrEnds

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # Generate output sime series
        vtSpikeTimes = (np.array(lnTSSpikes) + 1 + self._timestep) * self.dt
        tseOut = TSEvent(
            # Clip due to possible numerical errors,
            times=np.clip(vtSpikeTimes, t_start, t_stop),
            channels=liSpikeIDs,
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

        if vnIdMonitor is not None:
            # - Store recorded data in timeseries
            vtRecordTimes = np.repeat(
                (self._timestep + np.arange(num_timesteps + 1)) * self.dt, 2
            )[1:]
            self.tscRecorded = TSContinuous(vtRecordTimes, mfRecord)

        # Update time
        self._timestep += num_timesteps

        # - Update state
        self._state = state

        return tseOut

    def reset_state(self):
        # - Delete spikes that would arrive in recurrent synapses during future time steps
        #   by filling up deque with zeros
        nNumTSperDelay = self._dqvnNumRecSpikes.maxlen
        self._dqvnNumRecSpikes += [np.zeros(self.size) for _ in range(nNumTSperDelay)]
        # - Reset refractoriness
        self._vnTSUntilRefrEnds = np.zeros(self.size, int)
        # - Reset neuron state to self.vfVReset
        self.state = np.clip(self.vfVReset, self._nStateMin, self._nStateMax).astype(
            self.dtypeState
        )

    def randomize_state(self):
        # - Set state to random values between reset value and theshold
        self.state = np.clip(
            (np.amin(self.vfVThresh) - np.amin(self.vfVReset))
            * np.random.rand(self.size)
            - np.amin(self.vfVReset),
            self._nStateMin,
            self._nStateMax,
        ).astype(self.dtypeState)

    ### --- Properties

    @property
    def weights(self):
        return self.weights_rec

    # - weights as synonym for weights_rec
    @weights.setter
    def weights(self, mfNewW):
        self.weights_rec = mfNewW

    @property
    def weights_rec(self):
        return self._weights_rec

    @weights_rec.setter
    def weights_rec(self, mfNewW):
        self._weights_rec = self._expand_to_weight_size(mfNewW, "weights_rec", bAllowNone=False)

    @property
    def tTauBias(self):
        return self._nNumTSperBias * self._dt

    @tTauBias.setter
    def tTauBias(self, tNewBias):
        assert (
            np.isscalar(tNewBias) and tNewBias >= self.dt
        ), "Layer `{}`: tTauBias must be a scalar greater than dt ({})".format(
            self.name, self.dt
        )
        # - tNewBias is rounded to multiple of dt and at least dt
        self._nNumTSperBias = int(np.floor(tNewBias / self.dt))

    @property
    def tSpikeDelay(self):
        return self._dqvnNumRecSpikes.maxlen * self._dt

    @tSpikeDelay.setter
    def tSpikeDelay(self, tNewDelay):
        if tNewDelay is None:
            nNumTSperDelay = 1
        else:
            assert (
                np.isscalar(tNewDelay) and tNewDelay >= self.dt
            ), "Layer `{}`: tSpikeDelay must be a scalar greater than dt ({})".format(
                self.name, self.dt
            )
            # - tNewDelay is rounded to multiple of dt and at least dt
            nNumTSperDelay = int(np.floor(tNewDelay / self.dt))

        ## -- Create a deque for holding delayed spikes
        # - Copy spikes from previous deque
        if hasattr(self, "_dqvnNumRecSpikes"):
            lPrevSpikes = list(self._dqvnNumRecSpikes)
            nDifference = nNumTSperDelay - len(lPrevSpikes)
            # - If new delay is less, some spikes will be lost
            self._dqvnNumRecSpikes = deque(lPrevSpikes, maxlen=nNumTSperDelay)
            if nDifference >= 0:
                self._dqvnNumRecSpikes = deque(
                    [np.zeros(self.size) for _ in range(nDifference)] + lPrevSpikes,
                    maxlen=nNumTSperDelay,
                )
            else:
                self._dqvnNumRecSpikes = deque(lPrevSpikes, maxlen=nNumTSperDelay)
                print(
                    "Layer `{}`: Last {} spikes in buffer have been lost due to reduction of tSpikeDelay.".format(
                        self.name, np.sum(np.array(lPrevSpikes[:-nDifference]))
                    )
                )
        else:
            self._dqvnNumRecSpikes = deque(
                [np.zeros(self.size) for _ in range(nNumTSperDelay)],
                maxlen=nNumTSperDelay,
            )

    @property
    def vtRefractoryTime(self):
        return (
            None
            if self._vnNumTSperRefractory is None
            else self._vnNumTSperRefractory * self.dt
        )

    @vtRefractoryTime.setter
    def vtRefractoryTime(self, vtNewTime):
        if vtNewTime is None:
            self._vnNumTSperRefractory = None
        else:
            vtRefractoryTime = self._expand_to_net_size(vtNewTime, "vtRefractoryTime")
            # - vtRefractoryTime is rounded to multiple of dt and at least dt
            self._vnNumTSperRefractory = (np.floor(vtRefractoryTime / self.dt)).astype(
                int
            )

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, vNewState):
        self._state = np.clip(
            self._expand_to_net_size(vNewState, "state"),
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
                    self.name
                )
            )
        self._dtypeState = dtypeNew
        # - Convert state to dtype
        if hasattr(self, "_state"):
            self.state = self.state
