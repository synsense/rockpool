import numpy as np
import time

import TimeSeries as ts

class FFLayer():
    def __init__(self, sName, nSize, fDt, **kwargs):
        self.sName = sName
        self.t = 0
        self._fDt = fDt
        self._nSize = nSize

    def evolve(*args, **kwargs):
        pass

    def __str__(self):
        return '{} object: "{}"'.format(self.__class__.__name__, self.sName)

    def __repr__(self):
        return self.__str__()
    
    @property
    def nSize(self):
        return self._nSize

    @property
    def fDt(self):
        return self._fDt

    @fDt.setter
    def fDt(self, fNewDt):
        self._fDt = fNewDt

class PassThrough(FFLayer):
    """ Neuron states directly correspond to input, but can be delayed. """

    def __init__(self, sName, nSize, fDt, **kwargs):
        super().__init__(sName, nSize, fDt)
        self.vState = np.zeros(nSize)
        # Allow for delay, buffer delayed input in time series
        self._tDelay = kwargs.get('tDelay', 0)
        self.set_buffer()

    def set_buffer(self):
        if self.tDelay != 0:
            vtBuffer = np.arange(0,self.tDelay, self._fDt)
            self.tsBuffer = ts.TimeSeries(vtBuffer,
                                          np.zeros((len(vtBuffer), self.nSize)))
        else:
            self.tsBuffer = None        

    def evolve(self, tsInput):
        # Check input dimensions
        if tsInput.nNumTraces == 1:
            tsInput.mfSamples = np.repeat(tsInput.mfSamples.reshape((-1,1)), self.nSize, axis=1)
        assert tsInput.nNumTraces == self.nSize, 'Input and network dimensions do not match.'
        
        vtTimeTraceOut = np.arange(0, tsInput.tDuration+self._fDt, self._fDt) + self.t
        
        if self.tsBuffer is not None:
            mSamplesOut = np.vstack((self.tsBuffer.mfSamples,
                                     tsInput[ : tsInput.tStop-self.tDelay+self._fDt : self._fDt]))
            self.tsBuffer.mfSamples = tsInput[tsInput.tStop-self.tDelay+self._fDt : : self._fDt]
        else:
            mSamplesOut = tsInput[::self._fDt]
        
        self.t += tsInput.tDuration
        
        return ts.TimeSeries(vtTimeTraceOut, mSamplesOut)
        
    @property
    def tDelay(self):
        return self._tDelay

    # @tDelay.setter
    # def tDelay(self, tNewDelay):
        # Some method to extend self.tsBuffer


class FFRate(FFLayer):
    """ Feedforward layer consisting of rate-based neurons """

    def __init__(self, sName, nSize, fDt, **kwargs):
        super().__init__(sName, nSize, fDt)
        self.vPotential = np.zeros(nSize)
        self._vTau = np.array(kwargs.get('vTau', 10*fDt))
        self._vAlpha = self._fDt/self._vTau
        self.vGain = np.array(kwargs.get('vGain', 1))
        self.vBias = np.array(kwargs.get('vBias', 0))
        self.fActivation = dfActivation[kwargs.get('sActivation', 'ReLU')]

    def evolve(self, tsInput):
        if tsInput.nNumTraces == 1:
            tsInput.mfSamples = np.repeat(tsInput.mfSamples.reshape((-1,1)), self.nSize, axis=1)
        assert tsInput.nNumTraces == self.nSize, 'Input and network dimensions do not match.'
        vtTimeTraceIn = np.arange(0, tsInput.tDuration+self._fDt, self._fDt) + tsInput.tStart
        vtTimeTraceOut = vtTimeTraceIn - tsInput.tStart + self.t
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
        self._vAlpha = self._fDt/vNewTau

    @FFLayer.fDt.setter
    def fDt(self, fNewDt):
        self._fDt = fNewDt
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