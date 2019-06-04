###
# iaf_digital.py - Class implementing a recurrent layer consisting of
#                    digital neurons with constant leak and fixed-size
#                    integer as state. Event based.
###

# - Imports
# import sys
# strNetworkPath = sys.path[0] + '../../..'
# sys.path.insert(1, strNetworkPath)

from typing import Union, Optional, List, Tuple
import numpy as np
import heapq

from ...timeseries import TSEvent, TSContinuous

from ..layer import Layer

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["RecDIAF"]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-10
# - Minimum refractory time
tMinRefractory = 1e-9


# - RecDIAF - Class: define a spiking recurrent layer based on digital IAF neurons


class RecDIAF(Layer):
    """ RecDIAF - Class: define a spiking recurrent layer based on digital IAF neurons
    """

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        dt: float = 1e-4,
        tSpikeDelay: float = 1e-8,
        tTauLeak: float = 1e-3,
        vtRefractoryTime: Union[ArrayLike, float] = tMinRefractory,
        vfVThresh: Union[ArrayLike, float] = 100,
        vfVReset: Union[ArrayLike, float] = 0,
        vfVRest: Union[ArrayLike, float, None] = None,
        vfCleak: Union[ArrayLike, float] = 1,
        vfVSubtract: Union[ArrayLike, float, None] = None,
        dtypeState: Union[type, str] = "int8",
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        name: str = "unnamed",
    ):
        """
        RecDIAF - Construct a spiking recurrent layer with digital IAF neurons

        :param weights_in:           np.array nSizeInxN input weight matrix.
        :param weights_rec:          np.array NxN weight matrix

        :param dt:             float Length of single time step

        :param tSpikeDelay:     float Time after which a spike within the
                                      layer arrives at the recurrent
                                      synapses of the receiving neurons
                                      within the network.
        :param tTauLeak:        float Period for applying leak
        :param vtRefractoryTime:np.array Nx1 vector of refractory times.

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds.
        :param vfVReset:        np.array Nx1 vector of neuron reset potentials.
        :param vfVRest:         np.array Nx1 vector of neuron resting potentials.
                                Leak will change sign for neurons with state below this.
                                If None, leak will not change sign
        :param vfCleak:         np.array Nx1 vector of leak values.

        :param vfVSubtract:     np.array If not None, subtract provided values
                                         from neuron state after spike.
                                         Otherwise will reset.

        :param dtypeState:      type data type for the membrane potential
        :param vnIdMonitor:     array-like  IDs of neurons to be recorded

        :param name:         str Name for the layer. Default: 'unnamed'
        """

        # - Call super constructor
        super().__init__(weights=weights_in, dt=dt, name=name)

        # - Input weights must be provided
        assert (
            weights_rec is not None
        ), "Layer {}: Recurrent weights weights_rec must be provided.".format(self.name)

        # - Channel for leak
        self._nLeakChannel = self.size_in + self.size

        # - One large weight matrix to process input and recurrent connections
        #   as well as leak and multiple spikes if state after subtraction is
        #   still above threshold.
        self._mfWTotal = np.zeros((self._nLeakChannel + 2, self.size))

        # - Set minimum refractory time
        self._tMinRefractory = tMinRefractory

        # - Set neuron parameters
        self.weights = weights_rec
        self.weights_in = weights_in
        self.vfVSubtract = vfVSubtract
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vfCleak = vfCleak
        self.tSpikeDelay = tSpikeDelay
        self.tTauLeak = tTauLeak
        self.vtRefractoryTime = vtRefractoryTime
        self.dtypeState = dtypeState
        # - Record states of these neurons
        self.vnIdMonitor = vnIdMonitor

        self.reset_state()

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.state = np.clip(self.vfVReset, self._nStateMin, self._nStateMax).astype(
            self.dtypeState
        )
        # - Initialize heap and for events that are to be processed in future evolution
        self._heapRemainingSpikes = []

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        # - Adapt spike times in heap
        self._heapRemainingSpikes = [
            (tTime - self.t, iID) for tTime, iID in self._heapRemainingSpikes
        ]
        heapq.heapify(self._heapRemainingSpikes)
        self._timestep = 0

    ### --- State evolution
    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> TSEvent:
        """
        evolve - Evolve the state of this layer

        :param ts_input:         TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Prepare input and infer real duration of evolution
        vtEventTimes, vnEventChannels, num_timesteps, tFinal = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        ## -- Consider leak as periodic input spike with fixed weight

        # - Leak timings
        # First leak is at multiple of self.tTauLeak
        tFirstLeak = np.ceil(self.t / self.tTauLeak) * self.tTauLeak
        # Maximum possible number of leak steps in evolution period
        nMaxNumLeaks = np.ceil((tFinal - self.t) / self.tTauLeak) + 1
        vtLeak = np.arange(nMaxNumLeaks) * self.tTauLeak + tFirstLeak
        # - Do not apply leak at t=self.t, assume it has already been applied previously
        vtLeak = vtLeak[
            np.logical_and(vtLeak <= tFinal + tol_abs, vtLeak > self.t + tol_abs)
        ]

        # - Include leaks in event trace, assign channel self.LeakChannel to leak
        vnEventChannels = np.r_[
            vnEventChannels, np.ones_like(vtLeak) * self._nLeakChannel
        ]
        vtEventTimes = np.r_[vtEventTimes, vtLeak]

        # - Push spike timings and IDs to a heap, ordered by spike time
        # - Include spikes from previous evolution that might fall into this time interval
        heapSpikes = self._heapRemainingSpikes + list(
            zip(vtEventTimes, vnEventChannels.astype(int))
        )
        heapq.heapify(heapSpikes)

        # - Store layer spike times and IDs in lists
        ltSpikeTimes = []
        liSpikeIDs = []

        # - Times when neurons are able to spike again
        vtRefractoryEnds = np.zeros(self.size)

        # print("prepared")

        # import time
        # t0 = time.time()

        tTime = self.t
        i = 0
        # - Iterate over spike times. Stop when tFinal is exceeded.

        # - Copy instance variables to local variables
        state = self.state
        mfWTotal = self._mfWTotal
        nStateMin = self._nStateMin
        nStateMax = self._nStateMax
        nLeakChannel = self._nLeakChannel
        dtypeState = self.dtypeState
        vfVRest = self.vfVRest
        vfVThresh = self.vfVThresh
        vfVReset = self.vfVReset
        vtRefr = self.vtRefractoryTime
        tDelay = self.tSpikeDelay
        size_in = self.size_in
        vfVSubtract = self.vfVSubtract
        vnIdMonitor = None if self._vnIdMonitor.size == 0 else self._vnIdMonitor
        name = self.name

        if vnIdMonitor is not None:
            # - Lists for storing states, times and the channel from the heap
            lvStates = [state[vnIdMonitor].copy()]
            ltTimes = [tTime]
            lnChannels = [np.nan]

        while tTime <= tFinal:
            try:
                # - Iterate over spikes in temporal order
                tTime, nChannel = heapq.heappop(heapSpikes)
                # print(i, tTime, nChannel, "                       ", end="\r")
                if verbose:
                    print(
                        "Layer `{}`: Time passed: {:10.4f} of {} s.  Channel: {:4d}.  On heap: {:5d} events".format(
                            name, tTime, duration, nChannel, len(heapSpikes)
                        ),
                        end="\r",
                    )
            except IndexError:
                # - Stop if there are no spikes left
                break
            else:
                # print("update: ", self._mfWTotal[nChannel])

                if vnIdMonitor is not None:
                    # - Record state before updates
                    ltTimes.append(tTime)
                    lvStates.append(state[vnIdMonitor].copy())
                    lnChannels.append(nChannel)

                # - Only neurons that are not refractory can receive inputs
                vbNotRefractory = vtRefractoryEnds <= tTime
                # - Resting potential: Sign of leat so that it drives neuron states to vfVRest
                if vfVRest is not None and nChannel == nLeakChannel:
                    vbStateBelowRest = state[vbNotRefractory] < vfVRest[vbNotRefractory]
                    # Flip sign of leak for corresponding neurons
                    vnSign = -2 * vbStateBelowRest + 1
                    # Make sure leak is 0 when resting potential is reached
                    vnSign[state[vbNotRefractory] == vfVRest[vbNotRefractory]] = 0
                else:
                    vnSign = 1
                # - State updates after incoming spike
                state[vbNotRefractory] = np.clip(
                    state[vbNotRefractory]
                    + mfWTotal[nChannel, vbNotRefractory] * vnSign,
                    nStateMin,
                    nStateMax,
                ).astype(dtypeState)

                # - Neurons above threshold that are not refractory will spike
                vbSpiking = np.logical_and(state >= vfVThresh, vbNotRefractory)

                if vnIdMonitor is not None:
                    # - Record state after update but before subtraction/resetting
                    ltTimes.append(tTime)
                    lvStates.append(state[vnIdMonitor].copy())
                    lnChannels.append(np.nan)

                if vfVSubtract is not None:
                    # - Subtract from states of spiking neurons
                    state[vbSpiking] = np.clip(
                        state[vbSpiking] - vfVSubtract[vbSpiking], nStateMin, nStateMax
                    ).astype(dtypeState)
                    # - Check if among the neurons that are spiking there are still states above threshold
                    vbStillAboveThresh = (state >= vfVThresh) & vbSpiking
                    # - Add the time(s) when they stop being refractory to the heap
                    #   on the last channel, where weights are 0, so that no neuron
                    #   states are updated but neurons that are still above threshold
                    #   can spike immediately after they stop being refractory
                    vtStopRefr = vtRefr[vbStillAboveThresh] + tTime + tol_abs
                    # - Could use np.unique to only add each time once, but is very slow
                    # vtStopRefr = np.unique(vtRefr[vbStillAboveThresh]) + tTime + tol_abs
                    for tStopRefr in vtStopRefr:
                        heapq.heappush(heapSpikes, (tStopRefr, -1))
                else:
                    # - Set states to reset potential
                    state[vbSpiking] = vfVReset[vbSpiking].astype(dtypeState)

                if vnIdMonitor is not None:
                    # - Record state after subtraction/resetting
                    ltTimes.append(tTime)
                    lvStates.append(state[vnIdMonitor].copy())
                    lnChannels.append(np.nan)

                # - Determine times when refractory period will end for neurons that have just fired
                vtRefractoryEnds[vbSpiking] = tTime + vtRefr[vbSpiking]

                # - IDs of spiking neurons
                viSpikeIDs = np.where(vbSpiking)[0]
                # print("spiking: ", viSpikeIDs)
                # - Append spike events to lists
                ltSpikeTimes += [tTime] * np.sum(vbSpiking)
                liSpikeIDs += list(viSpikeIDs)

                # - Append new spikes to heap
                for nID in viSpikeIDs:
                    # - Delay spikes by self.tSpikeDelay. Set IDs off by self.size_in in order
                    #   to distinguish them from spikes coming from the input
                    heapq.heappush(heapSpikes, (tTime + tDelay, nID + size_in))
                # print("heap: ", heapq.nsmallest(5, heapSpikes))
            i += 1

        # - Update state variable
        self._state = state

        # print("finished loop")
        # print(time.time() - t0)

        # - Store remaining spikes (happening after tFinal) for next call of evolution
        self._heapRemainingSpikes = heapSpikes

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # - Update time
        self._timestep += num_timesteps

        if vnIdMonitor is not None:
            # - Store evolution of states in lists
            mStates = np.hstack((lvStates, np.reshape(lnChannels, (-1, 1))))
            self.tsRecorded = TSContinuous(ltTimes, mStates)

            # - Output time series
        return TSEvent(
            np.clip(ltSpikeTimes, t_start, t_stop),
            liSpikeIDs,
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

    def _prepare_input(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray, float, float):
        """
        _prepare_input - Sample input, set up time base

        :param ts_input:      TimeSeries TxM or Tx1 Input signals for this layer
        :param duration:    float Duration of the desired evolution, in seconds
        :param num_timesteps int Number of evolution time steps

        :return:
            vtEventTimes:     ndarray Event times
            vnEventChannels:  ndarray Event channels
            num_timesteps:    int Number of evlution time steps
            tFinal:           float End time of evolution
        """

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
            num_timesteps = (duration + tol_abs) // self.dt
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)

        # - End time of evolution
        tFinal = self.t + num_timesteps * self.dt

        # - Extract spike timings and channels
        if ts_input is not None:
            vtEventTimes, vnEventChannels = ts_input(
                t_start=self.t, t_stop=(self._timestep + num_timesteps) * self.dt
            )
            if np.size(vnEventChannels) > 0:
                # - Make sure channels are within range
                assert (
                    np.amax(vnEventChannels) < self.size_in
                ), "Layer {}: Only channels between 0 and {} are allowed".format(
                    self.name, self.size_in - 1
                )
        else:
            vtEventTimes, vnEventChannels = [], []

        return vtEventTimes, vnEventChannels, num_timesteps, tFinal

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
    def output_type(self):
        return TSEvent

    @property
    def input_type(self):
        return TSEvent

    @property
    def weights(self):
        return self.weights_rec

    @weights.setter
    def weights(self, mfNewW):
        self.weights_rec = mfNewW

    @property
    def weights_rec(self):
        return self._mfWTotal[self.size_in : self._nLeakChannel, :]

    @weights_rec.setter
    def weights_rec(self, mfNewW):

        self._mfWTotal[
            self.size_in : self._nLeakChannel, :
        ] = self._expand_to_weight_size(mfNewW, "weights_rec")

    @property
    def weights_in(self):
        return self._mfWTotal[: self.size_in, :]

    @weights_in.setter
    def weights_in(self, mfNewW):
        assert (
            np.size(mfNewW) == self.size_in * self.size
        ), "`mfNewW` must have [{}] elements.".format(self.size_in * self.size)

        self._mfWTotal[: self.size_in, :] = np.array(mfNewW)

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
    def vfVThresh(self):
        return self._vfVThresh

    @vfVThresh.setter
    def vfVThresh(self, vfNewThresh):
        self._vfVThresh = self._expand_to_net_size(vfNewThresh, "vfVThresh")

    @property
    def vfVReset(self):
        return self._vfVReset

    @vfVReset.setter
    def vfVReset(self, vfNewReset):
        self._vfVReset = self._expand_to_net_size(vfNewReset, "vfVReset")

    @property
    def vfVRest(self):
        return self._vfVRest

    @vfVRest.setter
    def vfVRest(self, vfNewRest):
        if vfNewRest is None:
            self._vfVRest = None
        else:
            self._vfVRest = self._expand_to_net_size(vfNewRest, "vfVRest")

    @property
    def vfCleak(self):
        return -self._mfWTotal[self._nLeakChannel, :]

    @vfCleak.setter
    def vfCleak(self, vfNewLeak):
        self._mfWTotal[self._nLeakChannel, :] = self._expand_to_net_size(
            -vfNewLeak, "vfCleak"
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
    def vtRefractoryTime(self):
        return self._vtRefractoryTime

    @vtRefractoryTime.setter
    def vtRefractoryTime(self, vtNewTime):

        self._vtRefractoryTime = np.clip(
            self._expand_to_net_size(vtNewTime, "vtRefractoryTime"),
            max(0, self._tMinRefractory),
            None,
        )

        if (np.array(vtNewTime) < self._tMinRefractory).any():
            print(
                "Refractory times must be at least {}.".format(self._tMinRefractory)
                + " Lower values have been clipped. The minimum value can be"
                + " set by changing _tMinRefractory."
            )

    @Layer.dt.setter
    def dt(self, _):
        raise ValueError("The `dt` property cannot be set for this layer")

    @property
    def tTauLeak(self):
        return self._tTauLeak

    @tTauLeak.setter
    def tTauLeak(self, tNewTauLeak):
        assert (
            np.isscalar(tNewTauLeak) and tNewTauLeak > 0
        ), "`tNewTauLeak` must be a scalar greater than 0."

        self._tTauLeak = tNewTauLeak

    @property
    def tSpikeDelay(self):
        return self._tSpikeDelay

    @tSpikeDelay.setter
    def tSpikeDelay(self, tNewSpikeDelay):
        assert (
            np.isscalar(tNewSpikeDelay) and tNewSpikeDelay > 0
        ), "`tNewSpikeDelay` must be a scalar greater than 0."

        self._tSpikeDelay = tNewSpikeDelay

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
