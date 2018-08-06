import numpy as np
from tqdm import tqdm
from ...timeseries import TSEvent
from .. import Layer


class SpikingLayer(Layer):
    '''
    EventCNNLayer: Event driven 2D convolution layer
    '''
    def __init__(self,
                 mfW: np.ndarray = None,
                 fVth: float = 8,
                 tDt: float = 1,
                 fNoiseStd: float = 0,
                 strName: str = 'unnamed'):
        """
        EventCNLayer - Implements a 2D convolutional layer of spiking neurons

        :param nfW:        np.ndarray Weight matrix
        :param fVth: float      Spiking threshold
        :param tDt:  float  Time step
        :param strName:    str        Name of this layer.
        """
        # Call parent constructor
        Layer.__init__(self, mfW, tDt=tDt,
                       fNoiseStd=fNoiseStd, strName=strName)

        self.fVth = fVth
        self.bias = np.zeros(self.nSize)
        self.reset_state()
        self.__nIdMonitor__ = 0  # Default monitorin of neuron 0

    def evolve(self,
               tsInput: TSEvent = None,
               tDuration: float = None) -> (TSEvent, np.ndarray):
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:  TSEvent  Input spike trian
        :param tDuration: float    Simulation/Evolution time
        :return:          TSEvent  output spike series

        """
        # Extract spike data from the input variable
        _, _, inpSpkRaster, _ = tsInput.raster(tDt=self.tDt)

        # Hold the sate of network at any time step when updated
        aStateTimeSeries = []
        aSpk = []

        # Record initial state of the network
        self.addToRecord(aStateTimeSeries, 0)

        # Local variables
        vState = self.vState
        fVth = self.fVth
        mfW = self.mfW
        bias = self.bias
        tDt = self.tDt

        # Iterate over all time steps
        for tCurrentTimeStep in tqdm(range(int(tDuration/tDt))):
            if tCurrentTimeStep >= len(inpSpkRaster):
                # If the time step goes beyond length of input stream
                # In this case no input is assumed
                vbInputSpkT = np.zeros(mfW.shape[0]).astype(bool)
            else:
                # If the time step is within the provided input stream duration
                vbInputSpkT = inpSpkRaster[tCurrentTimeStep]

            tCurrentTime = tCurrentTimeStep*tDt

            # Add input to neurons
            vW = mfW[vbInputSpkT]

            # If the input current have not been added
            if vW.shape != vState.shape:
                vW = vW.sum(axis=0)

            # State update
            vState[:] += vW + bias  # Membrane update with synaptic input

            self.addToRecord(aStateTimeSeries, tCurrentTime, nIdOut=self.__nIdMonitor__)

            # Check threshold and reset
            mbSpike = vState >= fVth
            if mbSpike.any():
                vbSpike, = np.nonzero(mbSpike)

                # Reset membrane state
                vState[mbSpike] -= fVth

                # Record spikes
                aSpk.append(
                    np.column_stack(([tCurrentTime]*len(vbSpike),
                                     vbSpike)))

                # Record state after reset
                self.addToRecord(aStateTimeSeries, tCurrentTime, nIdOut=self.__nIdMonitor__)

        # Convert arrays to TimeSeries objects
        if len(aSpk) > 0:
            mfSpk = np.row_stack(aSpk)
            evOut = TSEvent(mfSpk[:, 0],
                            mfSpk[:, 1],
                            strName='Output',
                            nNumChannels=self.nSize)
        else:
            evOut = TSEvent(None, nNumChannels=self.nSize)

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        # Update time
        self._t += tDuration
        return evOut

    def addToRecord(self,
                    aStateTimeSeries: list,
                    tCurrentTime: float,
                    nIdOut: int = None,
                    debug: bool = False):
        """
        addToRecord: Convenience function to record current state of the layer
                     or individual neuron

        :param aStateTimeSeries: list  A simple python list object to which the
                                       state needs to be appended
        :param tCurrentTime:     float Current simulation time
        :param nIdOut:           int   Neuron id to record the state of,
                                       if None all the neuron's states
                                       will be added to the record.
                                       Default = None
        """
        # Local variable
        mfState = self.vState

        if nIdOut is None:
            # Update record of state changes
            for nIdOutIter in range(self.nSize):
                aStateTimeSeries.append([tCurrentTime,
                                         nIdOutIter,
                                         mfState[nIdOutIter]])
                if debug:
                    print([tCurrentTime,
                           nIdOutIter,
                           mfState[nIdOutIter, 0]])
        else:
            aStateTimeSeries.append([tCurrentTime,
                                     nIdOut,
                                     mfState[nIdOut]])
            if debug:
                print([tCurrentTime,
                       nIdOutIter,
                       mfState[nIdOutIter, 0]])

        return
