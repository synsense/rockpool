import numpy as np
from ...timeseries import TSEvent, TimeSeries
from .iaf_cl import FFCLIAF


def softmax(x):
    '''
    Comput softmax values for each of scores in x
    :param x: array object
    '''
    return np.exp(x)/np.sum(np.exp(x), axis=0)


class SoftMaxLayer(FFCLIAF):
    '''
    EventCNNLayer: Event driven 2D convolution layer
    '''
    def __init__(self,
                 mfW: np.ndarray = None,
                 fVth: float = 8,
                 tDt: float = 1,
                 strName: str = 'unnamed'):
        """
        EventCNLayer - Implements a 2D convolutional layer of spiking neurons

        :param nfW:        np.ndarray Weight matrix
        :param fVth: float      Spiking threshold
        :param tDt:  float  Time step
        :param strName:    str        Name of this layer.
        """
        # Call parent constructor
        FFCLIAF.__init__(self, mfW, tDt=tDt, strName=strName)
        self.fVth = 1e10  # Just some absurdly large number that will never be reachable
        self.__nIdMonitor__ = None  # Monitor all neurons

    def evolve(self,
               tsInput: TSEvent = None,
               tDuration: float = None) -> (TSEvent, np.ndarray):
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:  TSEvent  Input spike trian
        :param tDuration: float    Simulation/Evolution time
        :return:          TSEvent  output spike series

        """
        _evOut = FFCLIAF.evolve(self, tsInput=tsInput, tDuration=tDuration)
        assert(len(_evOut.vtTimeTrace) == 0)
        # Analyse states
        mfStateHistoryLog = self._mfStateTimeSeries[10:]
        # Convert data to TimeSeries format
        mfStateTimeSeries = np.zeros((int(tDuration/self.tDt), self.nSize))
        for t in range(tDuration):
            mfDataTimeStep = mfStateHistoryLog[(mfStateHistoryLog[:, 0] == t)]
            mfStateTimeSeries[t] = mfDataTimeStep[:, 2]
        mfSoftMax = softmax(mfStateTimeSeries)
        tsOut = TimeSeries(vtTimeTrace=np.arange(tDuration), mfSamples=mfSoftMax, strName='SoftMaxOutput')
        return tsOut
