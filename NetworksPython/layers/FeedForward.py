import numpy as np

import TimeSeries as ts

class FFLayer():
    def __init__(self, sName, **kwargs):
        self.sName = sName
        self.t = 0

    def evolve(*args, **kwargs):
        pass

    def __str__(self):
        return '{} object: "{}"'.format(self.__class__.__name__, self.sName)

    def __repr__(self):
        return self.__str__()


class FFInput(FFLayer):
    """ Neuron states directly correspond to input, but can be delayed. """
    def __init__(self, sName, nSize, fDt, **kwargs):
        super().__init__(sName)
        self.nSize = nSize
        self.vState = np.zeros(nSize)
        self.__fDt = fDt
        # Allow for delay, buffer delayed input in time series
        self.tDelay = kwargs.get('tDelay', 0)
        self.set_buffer()

    def set_buffer(self):
        if self.tDelay != 0:
            vtBuffer = np.arange(0,self.tDelay, self.fDt)
            self.tsBuffer = ts.TimeSeries(vtBuffer,
                                          np.zeros((len(vtBuffer), self.nSize)))
        else:
            self.tsBuffer = None        

    def evolve(self, tsInput):
        if tsInput.nNumTraces == 1:
            tsInput.mfSamples = np.repeat(tsInput.mfSamples.reshape((-1,1)), self.nSize, axis=1)
        assert tsInput.nNumTraces == self.nSize, 'Input and network dimensions do not match.'
        vtTimeTraceOut = np.arange(0, tsInput.tDuration+self.fDt, self.fDt) + self.t
        if self.tsBuffer is not None:
            mOutput = np.vstack((self.tsBuffer.mfSamples,
                                 tsInput[ : tsInput.tStop-self.tDelay+self.fDt : self.fDt]))
            self.tsBuffer.mfSamples = tsInput[tsInput.tStop-self.tDelay+self.fDt : : self.fDt]
        else:
            mOutput = tsInput[::self.fDt]
        self.t += tsInput.tDuration
        print(vtTimeTraceOut.shape)
        print(mOutput.shape)
        return ts.TimeSeries(vtTimeTraceOut, mOutput)
        
    @property
    def fDt(self):
        return self.__fDt

    @fDt.setter
    def fDt(self, fNewDt):
        self.__fDt = fNewDt
        self.set_buffer()