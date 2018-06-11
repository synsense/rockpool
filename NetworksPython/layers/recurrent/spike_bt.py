###
# spike_bt - Implement a back-tick precise spike time recurrent layer, with fast and slow synapses
###

### --- Imports

from ..layer import Layer
from TimeSeries import *
import numpy as np
from typing import Union, Callable
import copy

from numba import njit

# - Try to import holoviews
try: import holoviews as hv
except Exception: pass

# - Configure exports
__all__ = ['RecFSSpikeEulerBT']


### --- Functions implementing membrane and synapse dynamics

@njit
def Neuron_dotV(t, V, dt,
                I_s_S, I_s_F, I_ext, I_bias,
                V_rest, V_reset, V_thresh,
                tau_V, tau_S, tau_F):
    return (V_rest - V + I_s_S + I_s_F + I_ext + I_bias) / tau_V

@njit
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

                 vfVThresh: Union[np.ndarray, float] = -55e-3,
                 vfVReset: Union[np.ndarray, float] = -65e-3,
                 vfVRest: Union[np.ndarray, float] = -65e-3,

                 tRefractoryTime: float = -np.finfo(float).eps,

                 fhSpikeCallback: Callable = None,

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
        :param fNoiseStd:       float Noise Std. Dev.

        :param vtTauN:          ndarray [Nx1] Neuron time constants
        :param vtTauSynR_f:     ndarray [Nx1] Post-synaptic neuron fast synapse TCs
        :param vtTauSynR_s:     ndarray [Nx1] Post-synaptic neuron slow synapse TCs

        :param vfVThresh:       ndarray [Nx1] Neuron firing thresholds
        :param vfVReset:        ndarray [Nx1] Neuron reset potentials
        :param vfVRest:         ndarray [Nx1] Neuron rest potentials

        :param tRefractoryTime: float         Post-spike refractory period

        :param fhSpikeCallback  Callable(lyrSpikeBT, tTime, nSpikeInd). Spike-based learning callback function. Default: None.

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
        self.vfVThresh = np.asarray(vfVThresh).astype('float')
        self.vfVReset = np.asarray(vfVReset).astype('float')
        self.vfVRest = np.asarray(vfVRest).astype('float')
        self.tRefractoryTime = float(tRefractoryTime)
        self.fhSpikeCallback = fhSpikeCallback

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

    @property
    def _tMinTau(self):
        """
        ._tMinTau - Smallest time constant of the layer
        """
        return min(np.min(self.vtTauSynR_s), np.min(self.vtTauSynR_f))

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

        :param tsInput:         TimeSeries input for a given time t [TxN]
        :param tDuration:       float Duration of simulation in seconds. Default: 100ms
        :param tMinDelta:       float Minimum time step taken. Default: 1/10 nominal TC

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

        vfZeros = np.zeros(self.nSize)
        # tSpike = np.nan
        # nFirstSpikeId = 0

        # - Euler integrator loop
        while tTime < tFinalTime:

            ### --- Numba-compiled inner function for speed
            # @njit
            def _evolve_backstep(tTime, mfW, mfW_s,
                                 vState, I_s_S, I_s_F, tDt,
                                 VLast, I_s_S_Last, I_s_F_Last,
                                 vfVReset, vfVRest, vfVThresh, vfBias,
                                 vtTauN, vtTauSynR_s, vtTauSynR_f,
                                 tRefractoryTime, vtRefractory,
                                 vfZeros):
                # - Enforce refractory period by clamping membrane potential to reset
                vState[vtRefractory > 0] = vfVReset[vtRefractory > 0]

                ## - Back-tick spike detector

                # - Locate spiking neurons
                vbSpikeIDs = vState > vfVThresh
                vnSpikeIDs = argwhere(vbSpikeIDs)
                nNumSpikes = np.sum(vbSpikeIDs)

                # - Were there any spikes?
                if nNumSpikes > 0:
                    # - Predict the precise spike times using linear interpolation
                    vtSpikeDeltas = (vfVThresh[vbSpikeIDs] - VLast[vbSpikeIDs]) * tDt / \
                                    (vState[vbSpikeIDs] - VLast[vbSpikeIDs])

                    # - Was there more than one neuron above threshold?
                    if nNumSpikes > 1:
                        # - Find the earliest spike
                        tSpikeDelta, nFirstSpikeId = min_argmin(vtSpikeDeltas)
                        nFirstSpikeId = vnSpikeIDs[nFirstSpikeId]
                    else:
                        tSpikeDelta = vtSpikeDeltas[0]
                        nFirstSpikeId = vnSpikeIDs[0]

                    # - Find time of actual spike
                    tShortestStep = tLast + tMinDelta
                    tSpike = clip_scalar(tLast + tSpikeDelta, tShortestStep, tTime)
                    tSpikeDelta = tSpike - tLast

                    # - Back-step time to spike
                    tTime = tSpike
                    vtRefractory = vtRefractory + tDt - tSpikeDelta

                    # - Back-step all membrane and synaptic potentials to time of spike (linear interpolation)
                    vState = _backstep(vState, VLast, tDt, tSpikeDelta)
                    I_s_S = _backstep(I_s_S, I_s_S_Last, tDt, tSpikeDelta)
                    I_s_F = _backstep(I_s_F, I_s_F_Last, tDt, tSpikeDelta)

                    # - Apply reset to spiking neuron
                    vState[nFirstSpikeId] = vfVReset[nFirstSpikeId]

                    # - Begin refractory period for spiking neuron
                    vtRefractory[nFirstSpikeId] = tRefractoryTime

                    # - Set spike currents
                    I_spike_slow = mfW_s[:, nFirstSpikeId]
                    I_spike_fast = mfW[:, nFirstSpikeId]

                else:
                    # - Clear spike currents
                    nFirstSpikeId = -1
                    I_spike_slow = vfZeros
                    I_spike_fast = vfZeros

                ### End of back-tick spike detector

                # - Save synapse and neuron states for previous time step
                VLast[:] = vState
                I_s_S_Last[:] = I_s_S
                I_s_F_Last[:] = I_s_F

                # - Update synapse and neuron states (Euler step)
                dotI_s_S = Syn_dotI(tTime, I_s_S, tDt, I_spike_slow, vtTauSynR_s)
                I_s_S += dotI_s_S * tDt

                dotI_s_F = Syn_dotI(tTime, I_s_F, tDt, I_spike_fast, vtTauSynR_f)
                I_s_F += dotI_s_F * tDt

                nIntTime = int((tTime-tStart) // tDt)
                I_ext = mfStaticInput[nIntTime, :]
                dotV = Neuron_dotV(tTime, vState, tDt,
                                   I_s_S, I_s_F, I_ext, vfBias,
                                   vfVRest, vfVReset, vfVThresh,
                                   vtTauN, vtTauSynR_s, vtTauSynR_f)
                vState += dotV * tDt

                return (tTime, nFirstSpikeId, dotV, vState, I_s_S, I_s_F, tDt,
                        VLast, I_s_S_Last, I_s_F_Last,
                        vtRefractory)

            ### --- END of compiled inner function

            # - Call compiled inner function
            (tTime, nFirstSpikeId, dotV,
             self._vState, self.I_s_S, self.I_s_F, self._tDt,
             VLast, I_s_S_Last, I_s_F_Last,
             vtRefractory) = _evolve_backstep(tTime, self._mfW, self.mfW_s,
                                              self._vState, self.I_s_S, self.I_s_F, self._tDt,
                                              VLast, I_s_S_Last, I_s_F_Last,
                                              self.vfVReset, self.vfVRest, self.vfVThresh, self.vfBias,
                                              self.vtTauN, self.vtTauSynR_s, self.vtTauSynR_f,
                                              self.tRefractoryTime, vtRefractory,
                                              vfZeros)

            # - Call spike-based learning callback
            if nFirstSpikeId > -1 and self.fhSpikeCallback is not None:
                self.fhSpikeCallback(self, tTime, nFirstSpikeId)

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

            # - Extend state storage variables, if needed
            if nStep >= nMaxTimeStep:
                nExtend = nMaxTimeStep
                vtTimes = np.append(vtTimes, full_nan(nExtend))
                mfV = np.append(mfV, full_nan((self.nSize, nExtend)), axis = 1)
                mfS = np.append(mfS, full_nan((self.nSize, nExtend)), axis = 1)
                mfF = np.append(mfF, full_nan((self.nSize, nExtend)), axis = 1)
                mfDotV = np.append(mfDotV, full_nan((self.nSize, nExtend)), axis = 1)
                nMaxTimeStep += nExtend

            # - Store the network states for this time step
            vtTimes[nStep] = tTime
            mfV[:, nStep] = self._vState
            mfS[:, nStep] = self.I_s_S
            mfF[:, nStep] = self.I_s_F
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

        ## - Store the network states for final time step
        vtTimes[nStep-1] = tFinalTime
        mfV[:, nStep-1] = self.vState
        mfS[:, nStep-1] = self.I_s_S
        mfF[:, nStep-1] = self.I_s_F

        ## - Trim state storage variables
        vtTimes = vtTimes[:nStep]
        mfV = mfV[:, :nStep]
        mfS = mfS[:, :nStep]
        mfF = mfF[:, :nStep]
        mfDotV = mfDotV[:, :nStep]
        vtSpikeTimes = vtSpikeTimes[:nSpikePointer]
        vnSpikeIndices = vnSpikeIndices[:nSpikePointer]

        ## - Construct return time series
        dResp = {'vt': vtTimes,
                 'mfX': mfV,
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

        # - Store "last evolution" state
        self._dLastEvolve = dResp
        self._t = tFinalTime

        # - Return output TimeSeries
        return TSEvent(vtSpikeTimes, vnSpikeIndices)

    @property
    def cOutput(self):
        return TSEvent

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


### --- Compiled concenience functions

@njit
def min_argmin(vfData: np.ndarray):
    """
    min_argmin - Accelerated function to find minimum and location of minimum

    :param vfData:  np.ndarray of data

    :return:        fMinVal, nMinLoc
    """
    n = 0
    nMinLoc = -1
    fMinVal = np.inf
    for x in vfData:
        if x < fMinVal:
            nMinLoc = n
            fMinVal = x
        n += 1

    return fMinVal, nMinLoc


@njit
def argwhere(vbData: np.ndarray):
    """
    argwhere - Accelerated argwhere function

    :param vbData:  np.ndarray Boolean array

    :return:        list vnLocations where vbData = True
    """
    vnLocs = []
    n = 0
    for val in vbData:
        if val: vnLocs.append(n)
        n += 1

    return vnLocs


@njit
def clip_vector(v: np.ndarray,
         fMin: float,
         fMax: float):
    """
    clip_vector - Accelerated vector clip function

    :param v:
    :param fMin:
    :param fMax:

    :return: Clipped vector
    """
    v[v < fMin] = fMin
    v[v > fMax] = fMax
    return v


@njit
def clip_scalar(fVal: float,
                fMin: float,
                fMax: float):
    """
    clip_scalar - Accelerated scalar clip function

    :param fVal:
    :param fMin:
    :param fMax:

    :return: Clipped value
    """
    if fVal < fMin: return fMin
    elif fVal > fMax: return fMax
    else: return fVal


def RepToNetSize(oData, nSize):
    if np.size(oData) == 1:
        return np.repeat(oData, nSize)
    else:
        return oData.flatten()