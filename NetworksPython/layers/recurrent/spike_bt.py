###
# spike_bt - Implement a back-tick precise spike time recurrent layer, with fast and slow synapses
###



from ..layer import Layer
from TimeSeries import *
import numpy as np
from typing import Union
# from warnings import warn
# from scipy.interpolate import interp1d
import copy

from numba import njit, float64

# - Try to import holoviews
try:
    import holoviews as hv

except Exception:
     pass


# - Configure exports
__all__ = ['RecFSSpikeEulerBT']

### --- Functions implementing membrane and synapse dynamics

# @jit(float64(float64, float64[:], float64,
#              float64[:], float64[:], float64[:], float64[:],
#              float64[:], float64[:], float64[:],
#              float64[:], float64[:], float64[:], float64))
def Neuron_dotV(t, V, dt,
                I_s_S, I_s_F, I_O, I_ext, I_bias,
                V_rest, V_reset, V_thresh,
                tau_V, tau_S, tau_F, tau_O):
    return (V_rest - V + I_s_S + I_s_F + I_ext + I_bias) / tau_V

# @jit(float64(float64, float64[:], float64,
#              float64[:],
#              float64[:]))
def Syn_dotI(t, I, dt,
             I_spike,
             tau_Syn):
    return -I / tau_Syn + I_spike / dt


@njit
def _backstep(vCurrent, vLast, tStep, tDesiredStep):
    return (vCurrent - vLast) / tStep * tDesiredStep + vLast

### --- RecFSSpikeEulerBT class implementation

class RecFSSpikeEulerBT(Layer):
    def __init__(self,
                 mfW_f: np.ndarray = None,
                 mfW_s: np.ndarray = None,
                 vfBias: np.ndarray = 0.,
                 fNoiseStd: float = 0.,

                 vtTauN: Union[np.ndarray, float] = 20e-3,
                 vtTauSynR_f: Union[np.ndarray, float] = 1e-3,
                 vtTauSynR_s: Union[np.ndarray, float] = 100e-3,
                 tTauSynO: float = 100e-3,

                 vfVThresh: Union[np.ndarray, float] = -55e-3,
                 vfVReset: Union[np.ndarray, float] = -65e-3,
                 vfVRest: Union[np.ndarray, float] = -65e-3,

                 tRefractoryTime: float = 0e-3,

                 tDt: float = None,
                 strName: str = None,
                 ):
        """
        DeneveReservoir - Implement a spiking reservoir with tight E/I balance
            This class does NOT use a Brian2 back-end. See the class code for possibilities
            to modify neuron and synapse dynamics. Currently uses leaky IAF neurons and exponential
            current synapses. Note that network parameters are tightly constrained for the reservoir
            to work as desired. See the documentation and source publications for details.

        :param mfW_f:           ndarray [NxN] Recurrent weight matrix (fast synapses)
        :param mfW_s:           ndarray [NxN] Recurrent weight matrix (slow synapses)
        :param vfBias:          ndarray [Nx1] Bias currents for each neuron
        :param vtTauN:          ndarray [Nx1] Neuron time constants
        :param vtTauSynR_f:     ndarray [Nx1] Post-synaptic neuron fast synapse TCs
        :param vtTauSynR_s:     ndarray [Nx1] Post-synaptic neuron slow synapse TCs
        :param tTauSynO:        float         Output slow TC
        :param vfVThresh:       ndarray [Nx1] Neuron firing thresholds
        :param vfVReset:        ndarray [Nx1] Neuron reset potentials
        :param vfVRest:         ndarray [Nx1] Neuron rest potentials
        :param tRefractoryTime: float         Post-spike refractory period

        :param tDt:             float         Nominal time step (Euler solver)
        :param strName:           str           Name of this layer
        """
        # - Initialise object and set properties
        super().__init__(mfW = mfW_f,
                         fNoiseStd = fNoiseStd,
                         strName = strName)

        self.mfW_s = mfW_s
        self.vfBias = np.asarray(vfBias).astype('float')
        self.vtTauN = np.asarray(vtTauN).astype('float')
        self.vtTauSynR_f = np.asarray(vtTauSynR_f).astype('float')
        self.vtTauSynR_s = np.asarray(vtTauSynR_s).astype('float')
        self.tTauSynO = np.asarray(tTauSynO).astype('float')
        self.vfVThresh = np.asarray(vfVThresh).astype('float')
        self.vfVReset = np.asarray(vfVReset).astype('float')
        self.vfVRest = np.asarray(vfVRest).astype('float')
        self.tRefractoryTime = np.asarray(tRefractoryTime).astype('float')

        # - Set a reasonable tDt
        if tDt is None:
            self.tDt = self._tMinTau / 10
        else:
            self.tDt = np.asarray(tDt).astype('float')

        # - Initialise network state
        self.reset_all()

    def reset_state(self):
        """
        reset_state() - Reset the internal state of the network
        """
        self.vState = self.vfVRest.copy()
        self.I_s_S = np.zeros(self.nSize)
        self.I_s_F = np.zeros(self.nSize)
        self.I_s_O = np.zeros(self.nSize)

    @property
    def _tMinTau(self):
        """
        ._tMinTau - Smallest time constant of the layer
        """
        return min(np.min(self.vtTauSynR_s), np.min(self.vtTauSynR_f), np.min(self.tTauSynO))

    def evolve(self,
               tsInput: TimeSeries = None,
               tDuration: float = None,
               tMinDelta: float = None,
               ) -> TimeSeries:
        """
        evolve() - Simulate the spiking reservoir, using a precise-time spike detector
            This method implements an Euler integrator, coupled with precise spike time detection using a linear
            interpolation between integration steps. Time is then reset to the spike time, and integration proceeds.
            For this reason, the time steps returned by the integrator are not homogenous. A minimum time step can be set;
            by default this is 1/10 of the nominal time step.

        :param tsInput:     TimeSeries input for a given time t [TxN]
        :param tDuration:   float Duration of simulation in seconds. Default: 100ms
        :param tMinDelta:   float Minimum time step taken. Default: 1/10 nominal TC

        :return: TimeSeries containing the output currents of the reservoir
        """

        # - Work out reasonable default for nominal time step (1/10 fastest time constant)
        if tMinDelta is None:
            tMinDelta = self.tDt / 10

        # - Check time step values
        assert tMinDelta < self.tDt, \
            '`tMinDelta` must be shorter than `tDt`'

        # - Get discretised input and nominal time trace
        vtInputTimeTrace, mfStaticInput, tDuration = self._prepare_input(tsInput, tDuration)
        tFinalTime = vtInputTimeTrace[-1]

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(np.size(vtInputTimeTrace), self.nSize) * self.fNoiseStd
        mfStaticInput += mfNoiseStep

        # - Allocate state storage variables
        nMaxTimeStep = int(tDuration // self.tDt)
        nSpikePointer = 0
        vtTimes = full_nan(nMaxTimeStep)
        mfV = full_nan((self.nSize, nMaxTimeStep))
        mfS = full_nan((self.nSize, nMaxTimeStep))
        mfF = full_nan((self.nSize, nMaxTimeStep))
        mfDotV = full_nan((self.nSize, nMaxTimeStep))
        mfO = full_nan((self.nSize, nMaxTimeStep))

        # - Allocate storage for spike times
        nMaxSpikePointer = nMaxTimeStep * self.nSize
        vtSpikeTimes = full_nan(nMaxSpikePointer)
        vnSpikeIndices = full_nan(nMaxSpikePointer)

        # - Refractory time variable
        vtRefractory = np.zeros(self.nSize)

        # - Initialise step and "previous step" variables
        tTime = self._t
        tStart = self._t
        nStep = 0
        tLast = 0.
        VLast = self._vState.copy()
        I_s_S_Last = self.I_s_S.copy()
        I_s_F_Last = self.I_s_F.copy()
        I_s_O_Last = self.I_s_O.copy()

        vfZeros = np.zeros(self.nSize)
        # tSpike = np.nan
        # nFirstSpikeId = 0

        # - Euler integrator loop
        while tTime < tFinalTime:
            # - Enforce refractory period by clamping membrane potential to reset
            self._vState[vtRefractory > 0] = self.vfVReset[vtRefractory > 0]

            ## - Back-tick spike detector

            # - Locate spiking neurons
            # vnSpikeIDs = np.argwhere(self.vState > self.vfVThresh).flatten()
            vnSpikeIDs = argwhere(self._vState > self.vfVThresh)

            # - Were there any spikes?
            if np.size(vnSpikeIDs) > 0:
                # - Predict the precise spike times using linear interpolation
                vtSpikeDeltas = (self.vfVThresh[vnSpikeIDs] - VLast[vnSpikeIDs]) * self.tDt / \
                                (self.vState[vnSpikeIDs] - VLast[vnSpikeIDs])

                # - Was there more than one neuron above threshold?
                if np.size(vnSpikeIDs) > 1:
                    # - Find the earliest spike
                    tSpikeDelta, nFirstSpikeId = min_argmin(vtSpikeDeltas)
                    nFirstSpikeId = vnSpikeIDs[nFirstSpikeId]
                else:
                    tSpikeDelta = vtSpikeDeltas[0]
                    nFirstSpikeId = vnSpikeIDs[0]

                # - Find time of actual spike
                tShortestStep = tLast + tMinDelta
                tSpike = np.clip(tLast + tSpikeDelta, tShortestStep, tTime)
                tSpikeDelta = tSpike - tLast

                # - Back-step time to spike
                tTime = tSpike
                vtRefractory = vtRefractory + self._tDt - tSpikeDelta

                # - Back-step all membrane and synaptic potentials to time of spike (linear interpolation)
                self._vState = _backstep(self._vState, VLast, self._tDt, tSpikeDelta)
                self.I_s_S = _backstep(self.I_s_S, I_s_S_Last, self._tDt, tSpikeDelta)
                self.I_s_F = _backstep(self.I_s_F, I_s_F_Last, self._tDt, tSpikeDelta)
                self.I_s_O = _backstep(self.I_s_O, I_s_O_Last, self._tDt, tSpikeDelta)

                # - Apply reset to spiking neuron
                self._vState[nFirstSpikeId] = self.vfVReset[nFirstSpikeId]

                # - Begin refractory period for spiking neuron
                vtRefractory[nFirstSpikeId] = self.tRefractoryTime

                # - Set spike currents
                I_spike_slow = self.mfW_s[:, nFirstSpikeId]
                I_spike_fast = self.mfW[:, nFirstSpikeId]
                I_spike_output = vfZeros
                I_spike_output[nFirstSpikeId] = 1

                # - Extend spike record, if necessary
                if nSpikePointer >= nMaxSpikePointer:
                    nExtend = int(nMaxSpikePointer // 2)
                    vtSpikeTimes = np.append(vtSpikeTimes, full_nan(nExtend))
                    vnSpikeIndices = np.append(vnSpikeIndices, full_nan(nExtend))
                    nMaxSpikePointer += nExtend

                # - Record spiking neuron
                vtSpikeTimes[nSpikePointer] = tTime
                vnSpikeIndices[nSpikePointer] = nFirstSpikeId
                nSpikePointer += 1

            else:
                # - Clear spike currents
                I_spike_slow = 0.
                I_spike_fast = 0.
                I_spike_output = 0.

            ### End of back-tick spike detector

            # - Save synapse and neuron states for previous time step
            VLast[:] = self.vState
            I_s_S_Last[:] = self.I_s_S
            I_s_F_Last[:] = self.I_s_F
            I_s_O_Last[:] = self.I_s_O

            # - Update synapse and neuron states (Euler step)
            dotI_s_S = Syn_dotI(tTime, self.I_s_S, self.tDt, I_spike_slow, self.vtTauSynR_s)
            self.I_s_S += dotI_s_S * self.tDt

            dotI_s_F = Syn_dotI(tTime, self.I_s_F, self.tDt, I_spike_fast, self.vtTauSynR_f)
            self.I_s_F += dotI_s_F * self.tDt

            dotx_hat = Syn_dotI(tTime, self.I_s_O, self.tDt, I_spike_output, self.tTauSynO)
            self.I_s_O += dotx_hat * self.tDt

            nIntTime = int((tTime-tStart) // self.tDt)
            I_ext = mfStaticInput[nIntTime, :]
            dotV = Neuron_dotV(tTime, self._vState, self._tDt,
                               self.I_s_S, self.I_s_F, [], I_ext, self.vfBias,
                               self.vfVRest, self.vfVReset, self.vfVThresh,
                               self.vtTauN, self.vtTauSynR_s, self.vtTauSynR_f, self.tTauSynO)
            self._vState += dotV * self._tDt

            # - Extend state storage variables, if needed
            if nStep >= nMaxTimeStep:
                nExtend = nMaxTimeStep
                vtTimes = np.append(vtTimes, full_nan(nExtend))
                mfV = np.append(mfV, full_nan((self.nSize, nExtend)), axis = 1)
                mfS = np.append(mfS, full_nan((self.nSize, nExtend)), axis = 1)
                mfF = np.append(mfF, full_nan((self.nSize, nExtend)), axis = 1)
                mfO = np.append(mfO, full_nan((self.nSize, nExtend)), axis = 1)
                mfDotV = np.append(mfDotV, full_nan((self.nSize, nExtend)), axis = 1)
                nMaxTimeStep += nExtend

            # - Store the network states for this time step
            vtTimes[nStep] = tTime
            mfV[:, nStep] = self._vState
            mfS[:, nStep] = self.I_s_S
            mfF[:, nStep] = self.I_s_F
            mfO[:, nStep] = self.I_s_O
            mfDotV[:, nStep] = dotV

            # - Next nominal time step
            tLast = copy.copy(tTime)
            tTime += self._tDt
            nStep += 1
            vtRefractory -= self.tDt
        ### End of Euler integration loop

        ## - Back-step to exact final time
        self.vState = _backstep(self.vState, VLast, self._tDt, tTime - tFinalTime)
        self.I_s_S = _backstep(self.I_s_S, I_s_S_Last, self._tDt, tTime - tFinalTime)
        self.I_s_F = _backstep(self.I_s_F, I_s_F_Last, self._tDt, tTime - tFinalTime)
        self.I_s_O = _backstep(self.I_s_O, I_s_O_Last, self._tDt, tTime - tFinalTime)

        ## - Store the network states for final time step
        vtTimes[nStep-1] = tFinalTime
        mfV[:, nStep-1] = self.vState
        mfS[:, nStep-1] = self.I_s_S
        mfF[:, nStep-1] = self.I_s_F
        mfO[:, nStep-1] = self.I_s_O

        ## - Trim state storage variables
        vtTimes = vtTimes[:nStep]
        mfV = mfV[:, :nStep]
        mfS = mfS[:, :nStep]
        mfF = mfF[:, :nStep]
        mfO = mfO[:, :nStep]
        mfDotV = mfDotV[:, :nStep]
        vtSpikeTimes = vtSpikeTimes[:nSpikePointer]
        vnSpikeIndices = vnSpikeIndices[:nSpikePointer]

        ## - Construct return time series
        dResp = {'vt': vtTimes,
                 'mfX': mfV,
                 'mfO': mfO,
                 'mfA': mfS,
                 'mfF': mfF,
                 'mfFast': mfF,
                 'mfDotV': mfDotV,
                 'mfStaticInput': mfStaticInput}

        bUseHV, _ = GetPlottingBackend()
        if bUseHV:
            dSpikes = {'vtTimes':  vtSpikeTimes,
                       'vnNeuron': vnSpikeIndices}

            dResp['spReservoir'] = hv.Points(dSpikes, kdims = ['vtTimes', 'vnNeuron'],
                                             label = 'Reservoir spikes').redim.range(vtTimes = (0, tDuration),
                                                                                     vnNeuron = (0, self.nSize))
        else:
            dResp['spReservoir'] = dict(vtTimes = vtSpikeTimes,
                                        vnNeuron = vnSpikeIndices)

        # - Convert some elements to time series
        dResp['tsX'] = TimeSeries(dResp['vt'], dResp['mfX'].T, strName = 'Membrane potential')
        dResp['tsA'] = TimeSeries(dResp['vt'], dResp['mfA'].T, strName = 'Slow synaptic state')
        dResp['tsO'] = TimeSeries(dResp['vt'], dResp['mfO'].T, strName = 'Output')

        # - Store "last evolution" state
        self._dLastEvolve = dResp
        self._t = tFinalTime

        # - Return output TimeSeries
        return dResp['tsO']

    @property
    def vfTauSynR_f(self):
        return self.__vfTauSynR_f

    @vfTauSynR_f.setter
    def vfTauSynR_f(self, vfTauSynR_f):
        self.__vfTauSynR_f = self._expand_to_net_size(vfTauSynR_f, 'vfTauSynR_f')

    @property
    def vfTauSynR_s(self):
        return self.__vfTauSynR_s

    @vfTauSynR_s.setter
    def vfTauSynR_s(self, vfTauSynR_s):
        self.__vfTauSynR_s = self._expand_to_net_size(vfTauSynR_s, 'vfTauSynR_s')

    @property
    def vfVThresh(self):
        return self.__vfVThresh

    @vfVThresh.setter
    def vfVThresh(self, vfVThresh):
        self.__vfVThresh = self._expand_to_net_size(vfVThresh, 'vfVThresh')

    @property
    def vfVRest(self):
        return self.__vfVRest

    @vfVRest.setter
    def vfVRest(self, vfVRest):
        self.__vfVRest = self._expand_to_net_size(vfVRest, 'vfVRest')

    @property
    def vfVReset(self):
        return self.__vfVReset

    @vfVReset.setter
    def vfVReset(self, vfVReset):
        self.__vfVReset = self._expand_to_net_size(vfVReset, 'vfVReset')

    @Layer.tDt.setter
    def tDt(self, tNewDt):
        assert tNewDt <= self._tMinTau / 10, \
            '`tNewDt` must be shorter than 1/10 of the shortest time constant, for numerical stability.'

        # - Call super-class setter
        super(RecFSSpikeEulerBT, RecFSSpikeEulerBT).tDt.__set__(self, tNewDt)


###### Convenience functions

# - Convenience method to return a nan array
def full_nan(vnShape: Union[tuple, int]):
    a = np.empty(vnShape)
    a.fill(np.nan)
    return a

# - Python-only min and argmin function
# @njit
def min_argmin(vfData):
    n = 0
    nMinLoc = -1
    fMinVal = float('inf')
    for x in vfData:
        if x < fMinVal:
            nMinLoc = n
            fMinVal = x
        n += 1

    return fMinVal, nMinLoc

# @njit
def argwhere(vbData):
    vnLocs = []
    n = 0
    for val in vbData:
        if val: vnLocs.append(n)
        n += 1

    return vnLocs

def RepToNetSize(oData, nSize):
    if np.size(oData) == 1:
        return np.repeat(oData, nSize)
    else:
        return oData.flatten()