import numpy as np
import time
from abc import abstractmethod
from numba import njit

import TimeSeries as ts
from layers.layer import Layer

class PassThrough(Layer):
    """ Neuron states directly correspond to input, but can be delayed. """

    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = 1,
                 fNoiseStd: float = 0,
                 tDelay: float = 0,
                 sName: str = None):
        super().__init__(mfW=mfW, tDt=tDt, fNoiseStd=fNoiseStd, sName=sName)
        self._tDelay = tDelay
        self.reset_all()
        
        # Buffer already reset by super().__init__ which calls self.reset_all()
        # self.reset_buffer()

    def reset_buffer(self):
        if self.tDelay != 0:
            vtBuffer = np.arange(0,self.tDelay, self._tDt)
            self.tsBuffer = ts.TimeSeries(vtBuffer,
                                          np.zeros((len(vtBuffer), self.nSize)))
        else:
            self.tsBuffer = None        

    def evolve(self, tsInput: np.ndarray, tDuration: float = None):
        
        vtTime, mfInput = self._prepare_input(tsInput, tDuration)

        if self.tsBuffer is not None:
            nBuffer = len(self.tsBuffer.vtTimeTrace)
            nSteps = len(vtTime)
            if nSteps > nBuffer:
                vtTimeOut = vtTime[:-nBuffer]
                vtTimeBuffer = vtTime[-nBuffer:]
                mSamplesOut = np.vstack((self.tsBuffer.mfSamples,
                                         noisy(mfInput[:-nBuffer]@self.mfW, self.fNoiseStd)))
                self.tsBuffer.mfSamples = noisy(mfInput[-nBuffer:]@self.mfW, self.fNoiseStd)
            else:
                mSamplesOut = self.tsBuffer(vtTime-self.t)
                self.tsBuffer.mfSamples = np.vstack((self.tsBuffer.mfSamples[nSteps:],
                                                     noisy(mfInput@self.mfW, self.fNoiseStd)))
        else:
            mSamplesOut = noisy(mfInput@self.mfW, self.fNoiseStd)
        
        self._t = vtTime[-1] + self.tDt
        
        return ts.TimeSeries(vtTime, mSamplesOut)

    def reset_state(self):
        super().reset_state()
        self.reset_buffer()

    def reset_all(self):
        super().reset_all()
        self.reset_buffer()

    @property
    def tDelay(self):
        return self._tDelay

    # @tDelay.setter
    # def tDelay(self, tNewDelay):
        # Some method to extend self.tsBuffer


class FFRate(Layer):
    """ Feedforward layer consisting of rate-based neurons """

    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = 1,
                 sName: str = None,
                 fNoiseStd: float = 0,
                 fhActivation: Callable[[np.ndarray], np.ndarray] = fhReLU,
                 vtTau: np.ndarray = 10,
                 vfGain: np.ndarray = 1,
                 vfBias: np.ndarray = 0):
        super().__init__(mfW=mfW, tDt=tDt, fNoiseStd=fNoiseStd, sName=sName)
        self.reset_all()
        try:
            self.vtTau, self.vfGain, self.vfBias = map(self.correct_param_shape, (vtTau, vfGain, vfBias))
        except AssertionError:
            raise AssertionError('Numbers of elements in vtTau, vfGain and vfBias'
                                 + ' must be 1 or match layer size.')
        self.vfAlpha = self._tDt/self.vtTau
        self.fhActivation = fhActivation
        
    def correct_param_shape(self, v) -> np.ndarray:
        """
        correct_param_shape - Convert v to 1D-np.ndarray and verify
                              that dimensions match self.nSize
        :param v:   Float or array-like that is to be converted
        :return:    v as 1D-np.ndarray
        """
        v = np.array(v).flatten()
        assert v.shape in ((1,), (self.nSize,), (1,self.nSize), (self.nSize), 1), (
            'Numbers of elements in v must be 1 or match layer size')
        return v

    def evolve(self, tsInput: ts.TimeSeries = None, tDuration: float = None) -> ts.TimeSeries:
        
        vtTime, mfInput = self._prepare_input(tsInput, tDuration)
        
        mSamplesAct = self._evolveEuler(vState=self.vState,     #self.vState is automatically updated
                                        mfInput=mfInput,
                                        mfW=self.mfW,
                                        vfGain=self.vfGain,
                                        vfBias=self.vfBias,
                                        vfAlpha=self.vfAlpha)

        # rtStart = time.time()
        # for i, vIn in enumerate(mSamplesIn):
        #     self.vState = self.potential(vIn)
        #     mSamplesOut[i] = self.vActivation
        #     print_progress(i, len(vtTime), time.time()-rtStart)
        # print('')

        # - Increment internal time representation
        self._t = vtTime[-1] + self.tDt
        
        return ts.TimeSeries(vtTime, mSamplesAct)
    
    @njit
    def potential(self, vInput: np.ndarray) -> np.ndarray:
        return (self._vfAlpha * noisy(vInput*self.vfGain + self.vfBias, self.fNoiseStd)
                + (1-self._vfAlpha)*self.vState)

    @abstractmethod
    def activation(self, *args, **kwargs):
        pass

    ### --- properties

    @property
    def vtTau(self):
        return self._vtTau

    @vtTau.setter
    def vtTau(self, vNewTau):
        vNewTau = self.correct_param_shape(vNewTau)
        if not (vNewTau >= self._tDt).all(): raise ValueError('All vtTau must be at least tDt.')
        self._vtTau = vNewTau
        self._vfAlpha = self._tDt/vNewTau

    @property
    def vfAlpha(self):
        return self._vfAlpha

    @vfAlpha.setter
    def vfAlpha(self, vNewAlpha):
        vNewAlpha = self.correct_param_shape(vNewAlpha)
        if not (vNewAlpha <= 1).all(): raise ValueError('All vfAlpha must be at most 1.')
        self._vfAlpha = vNewAlpha
        self._vtTau = self._tDt/vNewAlpha
    
    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vNewBias):
        self._vfBias = self.correct_param_shape(vNewBias)
    
    @property
    def vfGain(self):
        return self._vfGain

    @vfGain.setter
    def vfGain(self, vNewGain):
        self._vfGain = self.correct_param_shape(vNewGain)

    @property
    def fhActivation(self):
        return self._fhActivation

    @fhActivation.setter
    def fhActivation(self, f):
        self._fhActivation = f
        self._evolveEuler = get_evolution_function(f)

    @Layer.tDt.setter
    def tDt(self, tNewDt):
        if not (self.vtTau >= tNewDt).all(): raise ValueError('All vtTau must be at least tDt.')
        self._tDt = tNewDt
        self._vfAlpha = tNewDt/self._vtTau


def get_evolution_function(fhActivation: Callable[[np.ndarray], np.ndarray]):
    """
    get_evolution_function: Construct a compiled Euler solver for a given activation function

    :param fhActivation: Callable (x) -> f(x)
    :return: Compiled function evolve_Euler_complete(vState, nSize, mfW, mfInputStep, nNumSteps, vfBias, vtTau)
    """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(vState: np.ndarray,
                              mfInput: np.ndarray,
                              mfW: np.ndarray,
                              vfGain: np.ndarray,
                              vfBias: np.ndarray,
                              vfAlpha: np.ndarray,
                              vfNoiseStd) -> np.ndarray:
        
        # - Initialise storage of network output
        nNumSteps = len(mfInput)
        mfWeightedInput = mfInput@mfW
        mfStates = np.zeros_like(mfWeightedInput)

        # - Loop over time steps
        for nStep in range(nNumSteps):
            # - Evolve network state
            vDState = -vState + vfGain * mfWeightedInput[nStep, :]
            vState += vDState * vfAlpha
            # - Store network state
            mfStates[nStep, :] = vState

        return fhActivation(mfStates + vfBias)

    # - Return the compiled function
    return evolve_Euler_complete

@njit
def fhReLU(mfX: np.ndarray, fUpperBound: float = None) -> np.ndarray:
    """
    Activation function for rectified linear units.
    :param mfX:             ndarray with current neuron potentials
    :param fUpperBound:     Upper bound
    :return:                np.clip(mfX, 0, fUpperBound)
    """
    mfCopy = np.copy(mfX)
    mfCopy[np.where(mfX < 0)] = 0
    if fUpperBound is not None:
        mfCopy[np.where(mfX > fUpperBound)] = fUpperBound
    return mfCopy

@njit
def noisy(mX: np.ndarray, fStdDev: float) -> np.ndarray:
    """
    noisy - Add randomly distributed noise to each element of mX
    :param mX:  Array-like with values that noise is added to
    :param fStdDev: Float, the standard deviation of the noise to be added
    :return:        Array-like, mX with noise added
    """
    return fStdDev * np.random.randn(*mX.shape) + mX

def print_progress(iCurr: int, nTotal: int, tPassed: float):
    print('Progress: [{:6.1%}]    in {:6.1f} s. Remaining:   {:6.1f}'.format(
             iCurr/nTotal, tPassed, tPassed*(nTotal-iCurr)/max(0.1, iCurr)),
           end='\r')