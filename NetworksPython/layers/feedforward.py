import numpy as np
import time

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
        if tDuration is None:
            tDuration = tsInput.tStop - self.t
        tsInput = self._check_input_dims(tsInput)
        vtTimeTrace = self._gen_time_trace(self.t, tDuration)
        assert tsInput.contains(vtTimeTrace), ('Desired evolution interval not fully contained in input'
                                                + ' ({:.2f} to {:.2f} vs {:.2f} to {:.2f})'.format(
                                                vtTimeTrace[0], vtTimeTrace[-1], tsInput.tStart, tsInput.tStop))
        if self.tsBuffer is not None:
            nBuffer = len(self.tsBuffer.vtTimeTrace)
            nSamplesTrace = len(vtTimeTrace)
            if nSamplesTrace > nBuffer:
                vtTimeTraceOut = vtTimeTrace[:-nBuffer]
                vtTimeTraceBuffer = vtTimeTrace[-nBuffer:]
                mSamplesOut = np.vstack((self.tsBuffer.mfSamples,
                                         noisy(tsInput(vtTimeTraceOut)@self.mfW, self.fNoiseStd)))
                self.tsBuffer.mfSamples = noisy(tsInput(vtTimeTraceBuffer)@self.mfW, self.fNoiseStd)
            else:
                mSamplesOut = self.tsBuffer(vtTimeTrace-self.t)
                self.tsBuffer.mfSamples = np.vstack((self.tsBuffer.mfSamples[nSamplesTrace:],
                                                     noisy(tsInput(vtTimeTrace)@self.mfW, self.fNoiseStd)))
        else:
            mSamplesOut = noisy(tsInput(vtTimeTrace)@self.mfW, self.fNoiseStd)
        
        self._t += tDuration
        
        return ts.TimeSeries(vtTimeTrace, mSamplesOut)

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

def noisy(oInput: np.ndarray, fStdDev: float):
    """
    noisy - Add randomly distributed noise to each element of oInput
    :param oInput:  Array-like with values that noise is added to
    :param fStdDev: Float, the standard deviation of the noise to be added
    :return:        Array-like, oInput with noise added
    """
    return fStdDev * np.random.randn(*oInput.shape) + oInput


class FFRate(Layer):
    """ Feedforward layer consisting of rate-based neurons """

    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = 1,
                 sName: str = None,
                 fNoiseStd: float = 0,
                 vTau: np.ndarray = 10,
                 vGain: np.ndarray = 1,
                 vBias: np.ndarray = 0):
        super().__init__(mfW=sName, tDt=tDt, fNoiseStd=fNoiseStd, sName=sName)
        self._vTau = vTau
        self._vAlpha = self._tDt/self._vTau
        self.vGain = vGain
        self.vBias = vBias

    def evolve(self, tsInput, tDuration=None):
        if tDuration is None:
            tDuration = tsInput.tStop - self.t
        tsInput = self._check_input_dims(tsInput)
        vtTimeTrace = self._gen_time_trace(self.t, tDuration)
        assert tsInput.contains(vtTimeTrace), ('Desired evolution interval not fully contained in input'
                                                + ' ({:.2f} to {:.2f} vs {:.2f} to {:.2f})'.format(
                                                vtTimeTrace[0], vtTimeTrace[-1], tsInput.tStart, tsInput.tStop))


        mSamplesOut = np.zeros((len(vtTimeTraceOut), self.nSize))
        rtStart = time.time()
        for i, t in enumerate(vtTimeTraceIn):
            mSamplesOut[i] = self.vPotential = self.potential(tsInput(t))
            print_progress(i, len(vtTimeTraceIn), time.time()-rtStart)
        print('')
        self.t += tsInput.tDuration
        return ts.TimeSeries(vtTimeTraceOut, mSamplesOut)
    

    def potential(self, vInput):
        return self._vAlpha*(vInput*self.vGain + self.vBias) + (1-self._vAlpha)*self.vPotential

    @property
    def vState(self):
        return self.fActivation(self.vPotential)

    @property
    def vTau(self):
        return self._vTau

    @vTau.setter
    def vTau(self, vNewTau):
        self._vTau = vNewTau
        self._vAlpha = self._tDt/vNewTau

    @Layer.tDt.setter
    def tDt(self, fNewDt):
        self._tDt = fNewDt
        self._vAlpha = fNewDt/self._vTau

def relu(vPotential, fUpperBound=None):
    """
    Activation function for rectified linear units.
        vPotential : ndarray with current neuron potentials
        fUpperBound : if not None, upper bound for activation
    """
    return np.clip(vPotential, 0, fUpperBound)

dfActivation = {'ReLU' : relu}

def print_progress(iCurr, nTotal, tPassed):
    print('Progress: [{:6.1%}]    in {:6.1f} s. Remaining:   {:6.1f}'.format(
             iCurr/nTotal, tPassed, tPassed*(nTotal-iCurr)/max(0.1, iCurr)),
           end='\r')