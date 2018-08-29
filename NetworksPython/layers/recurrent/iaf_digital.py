###
# digital_brian.py - Class implementing a recurrent layer consisting of
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

from ...timeseries import TSEvent

from ..layer import Layer

from .timedarray_shift import TimedArray as TAShift

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["RecDIAF"]


# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-10
# - Minimum refractory time
tMinRefractory = 1e-9
# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - RecDIAFBrian - Class: define a spiking recurrent layer based on digital IAF neurons


class RecDIAF(Layer):
    """ RecDIAFBrian - Class: define a spiking recurrent layer based on digital IAF neurons
    """

    ## - Constructor
    def __init__(
        self,
        mfWIn: np.ndarray,
        mfWRec: np.ndarray,
        tDt: float = 1e-4,
        tSpikeDelay: float = 1e-8,
        tTauLeak: float = 1e-3,
        vtRefractoryTime: Union[ArrayLike, float] = tMinRefractory,
        vfVThresh: Union[ArrayLike, float] = 100,
        vfVReset: Union[ArrayLike, float] = 0,
        vfCleak: Union[ArrayLike, float] = 1,
        vfVSubtract: Union[ArrayLike, float, None] = None,
        strDtypeState: str = "int8",
        strName: str = "unnamed",
    ):
        """
        RecDIAFBrian - Construct a spiking recurrent layer with digital IAF neurons

        :param mfWIn:           np.array nSizeInxN input weight matrix.
        :param mfWRec:          np.array NxN weight matrix

        :param tDt:             float Length of single time step

        :param tSpikeDelay:     float Time after which a spike within the
                                      layer arrives at the recurrent
                                      synapses of the receiving neurons
                                      within the network.
        :param tTauLeak:        float Period for applying leak
        :param vtRefractoryTime:np.array Nx1 vector of refractory times.

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds.
        :param vfVReset:        np.array Nx1 vector of neuron reset potentials.
        :param vfCleak:         np.array Nx1 vector of leak values.

        :param vfVSubtract:     np.array If not None, subtract provided values
                                         from neuron state after spike.
                                         Otherwise will reset.

        :param strDtypeState:   str data type for the membrane potential

        :param strName:         str Name for the layer. Default: 'unnamed'
        """

        # - Call super constructor
        super().__init__(mfW=mfWIn, tDt=tDt, strName=strName)

        # - Input weights must be provided
        assert (
            mfWRec is not None
        ), "Layer {}: Recurrent weights mfWRec must be provided.".format(
            self.strName
        )

        # - Channel for leak
        self._nLeakChannel = self.nSizeIn + self.nSize

        # - One large weight matrix to process input and recurrent connections
        #   as well as leak and multiple spikes if state after subtraction is
        #   still above threshold.
        self._mfWTotal = np.zeros((self._nLeakChannel + 2, self.nSize))

        # - Set minimum refractory time
        self._tMinRefractory = tMinRefractory

        # - Set neuron parameters
        self.mfW = mfWRec
        self.mfWIn = mfWIn
        self.vfVSubtract = vfVSubtract
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfCleak = vfCleak
        self.tSpikeDelay = tSpikeDelay
        self.tTauLeak = tTauLeak
        self.vtRefractoryTime = vtRefractoryTime
        self.strDtypeState = strDtypeState

        self.reset_state()

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self.vState = self.vfVReset
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
        self._nTimeStep = 0

    ### --- State evolution

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: Optional[bool] = False,
    ) -> TSEvent:
        """
        evolve - Evolve the state of this layer

        :param tsInput:         TSEvent  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Prepare input and infer real duration of evolution
        vtEventTimes, vnEventChannels, nNumTimeSteps, tFinal = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        ## -- Consider leak as periodic input spike with fixed weight

        # - Leak timings
        # First leak is at multiple of self.tTauLeak
        tFirstLeak = np.ceil(self.t / self.tTauLeak) * self.tTauLeak
        # Maximum possible number of leak steps in evolution period
        nMaxNumLeaks = np.ceil((tFinal-self.t) / self.tTauLeak) + 1
        vtLeak = np.arange(nMaxNumLeaks) * self.tTauLeak + tFirstLeak
        # - Do not apply leak at t=self.t, assume it has already been applied previously
        vtLeak = vtLeak[
            np.logical_and(vtLeak <= tFinal + fTolAbs, vtLeak > self.t + fTolAbs)
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
        vtRefractoryEnds = np.zeros(self.nSize)

        # print("prepared")

        # import time
        # t0 = time.time()

        tTime = self.t
        i = 0
        # - Iterate over spike times. Stop when tFinal is exceeded.

        # - Copy instance variables to local variables
        vState = self.vState
        mfWTotal = self._mfWTotal
        nStateMin = self._nStateMin
        nStateMax = self._nStateMax
        strDtypeState = self.strDtypeState
        vfVThresh = self.vfVThresh
        vfVReset = self.vfVReset
        vtRefr = self.vtRefractoryTime
        tDelay = self.tSpikeDelay
        nSizeIn = self.nSizeIn
        vfVSubtract = self.vfVSubtract

        while tTime <= tFinal:
            try:
                # - Iterate over spikes in temporal order
                tTime, nChannel = heapq.heappop(heapSpikes)
                print(i, tTime, nChannel, end="\r")

            except IndexError:
                # - Stop if there are no spikes left
                break
            else:
                # print("update: ", self._mfWTotal[nChannel])

                # - Only neurons that are not refractory can receive inputs
                vbNotRefractory = vtRefractoryEnds <= tTime
                # - State updates after incoming spike
                vState[vbNotRefractory] = np.clip(
                    vState[vbNotRefractory] + mfWTotal[nChannel, vbNotRefractory],
                    nStateMin,
                    nStateMax,
                ).astype(strDtypeState)

                # - Neurons above threshold and not refractory
                vbSpiking = np.logical_and(vState >= vfVThresh, vbNotRefractory)

                if vfVSubtract is not None:
                    # - Subtract from states of spiking neurons
                    vState[vbSpiking] = np.clip(
                        vState[vbSpiking] - vfVSubtract[vbSpiking], nStateMin, nStateMax
                    ).astype(strDtypeState)
                    # - Check if there are still states above threshold
                    for iStillAboveThresh in np.where(vState >= vfVThresh)[0]:
                        # - Add the time when they stop being refractory to the heap
                        #   on the last channel, where weights are 0, so that no states
                        #   are updated, only respective states are subtracted
                        heapq.heappush(
                            heapSpikes,
                            (tTime + vtRefr[iStillAboveThresh] + fTolAbs, -1),
                        )

                else:
                    # - Set states to reset potential
                    vState[vbSpiking] = vfVReset[vbSpiking].astype(strDtypeState)

                # print("new state: ", self._vState)

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
                    # - Delay spikes by self.tSpikeDelay. Set IDs off by self.nSizeIn in order
                    #   to distinguish them from spikes coming from the input
                    heapq.heappush(heapSpikes, (tTime + tDelay, nID + nSizeIn))
                # print("heap: ", heapq.nsmallest(5, heapSpikes))
            i += 1
        # - Update state variable
        self._vState = vState

        # print("finished loop")
        # print(time.time() - t0)

        # - Store remaining spikes (happening after tFinal) for next call of evolution
        self._heapRemainingSpikes = heapSpikes

        # - Update time
        self._nTimeStep += nNumTimeSteps

        # - Output time series
        return TSEvent(ltSpikeTimes, liSpikeIDs)

    def _prepare_input(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray, float, float):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:      TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:    float Duration of the desired evolution, in seconds
        :param nNumTimeSteps int Number of evolution time steps

        :return:
            vtEventTimes:     ndarray Event times
            vnEventChannels:  ndarray Event channels
            nNumTimeSteps:    int Number of evlution time steps
            tFinal:           float End time of evolution
        """

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
            nNumTimeSteps = (tDuration + fTolAbs) // self.tDt
        else:
            assert isinstance(
                nNumTimeSteps, int
            ), "Layer `{}`: nNumTimeSteps must be of type int.".format(self.strName)

        # - End time of evolution
        tFinal = self.t + nNumTimeSteps * self.tDt

        # - Extract spike timings and channels
        if tsInput is not None:
            vtEventTimes, vnEventChannels, __ = tsInput.find(
                [self.t, (self._nTimeStep + nNumTimeSteps) * self.tDt]
            )
            if np.size(vnEventChannels) > 0:
                # - Make sure channels are within range
                assert (
                    np.amax(vnEventChannels) < self.nSizeIn
                ), "Layer {}: Only channels between 0 and {} are allowed".format(
                    self.strName, self.nSizeIn - 1
                )
        else:
            vtEventTimes, vnEventChannels = [], []

        return vtEventTimes, vnEventChannels, nNumTimeSteps, tFinal

    def randomize_state(self):
        self.vState = np.random.randint(
            self._nStateMin, self._nStateMax + 1, size=self.nSize
        )

    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

    @property
    def cInput(self):
        return TSEvent

    @property
    def mfW(self):
        return self.mfWRec

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWRec = mfNewW

    @property
    def mfWRec(self):
        return self._mfWTotal[self.nSizeIn : self._nLeakChannel, :]

    @mfWRec.setter
    def mfWRec(self, mfNewW):

        self._mfWTotal[
            self.nSizeIn : self._nLeakChannel, :
        ] = self._expand_to_weight_size(mfNewW, "mfWRec")

    @property
    def mfWIn(self):
        return self._mfWTotal[: self.nSizeIn, :]

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        assert (
            np.size(mfNewW) == self.nSizeIn * self.nSize
        ), "`mfNewW` must have [{}] elements.".format(
            self.nSizeIn * self.nSize
        )

        self._mfWTotal[: self.nSizeIn, :] = np.array(mfNewW)

    @property
    def vState(self):
        return self._vState

    @vState.setter
    def vState(self, vNewState):
        self._vState = np.clip(
            self._expand_to_net_size(vNewState, "vState"),
            self._nStateMin,
            self._nStateMax,
        ).astype(self.strDtypeState)

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

    @Layer.tDt.setter
    def tDt(self, _):
        raise ValueError("The `tDt` property cannot be set for this layer")

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
    def strDtypeState(self):
        return self._strDtypeState

    @strDtypeState.setter
    def strDtypeState(self, strNewDtype):
        self._nStateMin = np.iinfo(strNewDtype).min
        self._nStateMax = np.iinfo(strNewDtype).max
        self._strDtypeState = strNewDtype
