import numpy as np
from ...timeseries import TSEvent
from .. import Layer


class EventCNNLayer(Layer):
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
        self.reset_state()

    def evolve(self,
               tsInput: TSEvent = None,
               tDuration: float = None,
               bVerbose: bool = False,
    ) -> (TSEvent, np.ndarray):
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:  TSEvent  Input spike trian
        :param tDuration:   float    Simulation/Evolution time
        :param bVerbose:    bool Currently no effect, just for conformity
        :return:          TSEvent  output spike series

        """
        # Extract spike data from the input variable
        vSpk = tsInput._vtTimeTrace
        vIdInput = tsInput._vnChannels

        # Hold the sate of network at any time step when updated
        aStateTimeSeries = []
        aSpk = []

        # Record initial state of the network
        self.addToRecord(aStateTimeSeries, 0)

        # Local variables
        vState = self.vState
        fVth = self.fVth

        # Iterate over all input spikes
        for nSpikeIndx in range(len(vSpk)):

            tCurrentTime = vSpk[nSpikeIndx]
            nInputId = vIdInput[nSpikeIndx].astype(int)

            # Add input to neurons
            vW = self._mfW[nInputId]

            # State update
            vState[:] += vW  # Learning state

            # TODO: The above could perhaps be written in a different function
            #       to account for diffrent lookup tables like CNNs

            self.addToRecord(aStateTimeSeries, tCurrentTime)

            # Check threshold and reset
            mbSpike = vState >= fVth
            if mbSpike.any():
                vbSpike, = np.nonzero(mbSpike)

                # Reset membrane state
                vState[mbSpike] = 0

                # Record spikes
                aSpk.append(
                    np.column_stack(([tCurrentTime]*len(vbSpike),
                                     vbSpike)))

                # Record state after reset
                self.addToRecord(aStateTimeSeries, tCurrentTime)

        # Convert arrays to TimeSeries objects
        mfSpk = np.row_stack(aSpk)
        evOut = TSEvent(mfSpk[:, 0],
                        mfSpk[:, 1],
                        strName='Output')

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
                                     nIdOutIter,
                                     mfState[nIdOutIter]])
            if debug:
                print([tCurrentTime,
                       nIdOutIter,
                       mfState[nIdOutIter, 0]])

        return
