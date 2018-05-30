import numpy as np
import time
from abc import abstractmethod

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
        vtTime = self._gen_time_trace(self.t, tDuration)
        assert tsInput.contains(vtTime), ('Desired evolution interval not fully contained in input'
                                                + ' ({:.2f} to {:.2f} vs {:.2f} to {:.2f})'.format(
                                                vtTime[0], vtTime[-1], tsInput.tStart, tsInput.tStop))
        if self.tsBuffer is not None:
            nBuffer = len(self.tsBuffer.vtTimeTrace)
            nSamplesTrace = len(vtTime)
            if nSamplesTrace > nBuffer:
                vtTimeOut = vtTime[:-nBuffer]
                vtTimeBuffer = vtTime[-nBuffer:]
                mSamplesOut = np.vstack((self.tsBuffer.mfSamples,
                                         noisy(tsInput(vtTimeOut)@self.mfW, self.fNoiseStd)))
                self.tsBuffer.mfSamples = noisy(tsInput(vtTimeBuffer)@self.mfW, self.fNoiseStd)
            else:
                mSamplesOut = self.tsBuffer(vtTime-self.t)
                self.tsBuffer.mfSamples = np.vstack((self.tsBuffer.mfSamples[nSamplesTrace:],
                                                     noisy(tsInput(vtTime)@self.mfW, self.fNoiseStd)))
        else:
            mSamplesOut = noisy(tsInput(vtTime)@self.mfW, self.fNoiseStd)
        
        self._t += tDuration
        
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
                 vTau: np.ndarray = 10,
                 vGain: np.ndarray = 1,
                 vBias: np.ndarray = 0):
        super().__init__(mfW=mfW, tDt=tDt, fNoiseStd=fNoiseStd, sName=sName)
        self.reset_all()
        try:
            self.vTau, self.vGain, self.vBias = map(self.correct_param_shape, (vTau, vGain, vBias))
        except AssertionError:
            raise AssertionError('Numbers of elements in vTau, vGain and vBias'
                                 + ' must be 1 or match layer size.')
        self.vAlpha = self._tDt/self.vTau
        
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

    def evolve(self, tsInput: ts.TimeSeries, tDuration: float = None) -> ts.TimeSeries:
        if tDuration is None:
            tDuration = tsInput.tStop - self.t
        tsInput = self._check_input_dims(tsInput)
        vtTime = self._gen_time_trace(self.t, tDuration)
        assert tsInput.contains(vtTime), ('Desired evolution interval not fully contained in input'
                                                + ' ({:.2f} to {:.2f} vs {:.2f} to {:.2f})'.format(
                                                vtTime[0], vtTime[-1], tsInput.tStart, tsInput.tStop))

        mSamplesOut = np.zeros((len(vtTime), self.nSize))
        mSamplesIn = tsInput(vtTime)@self.mfW
        rtStart = time.time()
        for i, vIn in enumerate(mSamplesIn):
            mSamplesOut[i] = self.vState = self.potential(vIn)
            print_progress(i, len(vtTime), time.time()-rtStart)
        print('')
        self._t += tDuration
        
        return ts.TimeSeries(vtTime, mSamplesOut)
    
    def potential(self, vInput: np.ndarray) -> np.ndarray:
        return (self._vAlpha * noisy(vInput*self.vGain + self.vBias, self.fNoiseStd)
                + (1-self._vAlpha)*self.vState)

    @abstractmethod
    def activation(self, *args, **kwargs):
        pass

    ### --- properties

    @property
    def vTau(self):
        return self._vTau

    @vTau.setter
    def vTau(self, vNewTau):
        vNewTau = self.correct_param_shape(vNewTau)
        if not (vNewTau >= self._tDt).all(): raise ValueError('All vTau must be at least tDt.')
        self._vTau = vNewTau
        self._vAlpha = self._tDt/vNewTau

    @property
    def vAlpha(self):
        return self._vAlpha

    @vAlpha.setter
    def vAlpha(self, vNewAlpha):
        vNewAlpha = self.correct_param_shape(vNewAlpha)
        if not (vNewAlpha <= 1).all(): raise ValueError('All vAlpha must be at most 1.')
        self._vAlpha = vNewAlpha
        self._vTau = self._tDt/vNewAlpha
    
    @property
    def vBias(self):
        return self._vBias

    @vBias.setter
    def vBias(self, vNewBias):
        self._vBias = self.correct_param_shape(vNewBias)
    
    @property
    def vGain(self):
        return self._vGain

    @vGain.setter
    def vGain(self, vNewGain):
        self._vGain = self.correct_param_shape(vNewGain)

    @Layer.tDt.setter
    def tDt(self, tNewDt):
        if not (self.vTau >= tNewDt).all(): raise ValueError('All vTau must be at least tDt.')
        self._tDt = tNewDt
        self._vAlpha = tNewDt/self._vTau


class FFReLU(FFRate):
    def __init__(self,
                 mfW: np.ndarray,
                 tDt: float = 1,
                 vTau: np.ndarray = 10,
                 vGain: np.ndarray = 1,
                 vBias: np.ndarray = 0,
                 fNoiseStd: float = 0,
                 fActUpperBound: float = None,
                 sName: str = None):
        super().__init__(mfW=mfW, tDt=tDt, vTau=vTau, vGain=vGain,
                         vBias=vBias, fNoiseStd=fNoiseStd, sName=sName)
        self.fActUpperBound = fActUpperBound

    def activation(self, fUpperBound=None) -> np.ndarray:
        """
        Activation function for rectified linear units.
            vPotential : ndarray with current neuron potentials
            fUpperBound : if not None, upper bound for activation
        """
        return np.clip(self.vState, 0, fUpperBound)

    @property
    def vActivation(self):
        return self.activation(self.fActUpperBound)


def noisy(oInput: np.ndarray, fStdDev: float) -> np.ndarray:
    """
    noisy - Add randomly distributed noise to each element of oInput
    :param oInput:  Array-like with values that noise is added to
    :param fStdDev: Float, the standard deviation of the noise to be added
    :return:        Array-like, oInput with noise added
    """
    return fStdDev * np.random.randn(*oInput.shape) + oInput

def print_progress(iCurr: int, nTotal: int, tPassed: float):
    print('Progress: [{:6.1%}]    in {:6.1f} s. Remaining:   {:6.1f}'.format(
             iCurr/nTotal, tPassed, tPassed*(nTotal-iCurr)/max(0.1, iCurr)),
           end='\r')