import numpy as np
from tqdm import tqdm
from ...timeseries import TSEvent
from .iaf_cl import FFCLIAF


class EventDrivenSpikingLayer(FFCLIAF):
    '''
    EventCNNLayer: Event driven 2D convolution layer
    '''
    def __init__(self, mfW: np.ndarray = None,
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
        FFCLIAF.__init__(self, mfW, fVth=fVth, tDt=tDt, strName=strName)

    def evolve(self,
               tsInput: TSEvent = None,
               tDuration: float = None,
               bVerbose: bool = False,
        ) -> (TSEvent, np.ndarray):
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:  TSEvent  Input spike trian
        :param tDuration: float    Simulation/Evolution time
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
        mfW = self.mfW

        # Iterate over all input spikes
        for nSpikeIndx in tqdm(range(len(vSpk))):

            tCurrentTime = vSpk[nSpikeIndx]
            nInputId = vIdInput[nSpikeIndx].astype(int)

            # Add input to neurons
            vW = mfW[nInputId]

            # State update
            vState[:] += vW  # Membrane state update

            self.addToRecord(aStateTimeSeries, tCurrentTime, nIdOut=self.__nIdMonitor__)

            # Check threshold and reset
            mbSpike = vState >= fVth
            if mbSpike.any():
                vnSpike, = np.nonzero(mbSpike)

                # Reset membrane state
                vState[mbSpike] -= fVth

                # TODO: The above code has a bug
                # If the threshold goes over 2*fVth this spike will not be detected till the next update.

                # Record spikes
                aSpk.append(
                    np.column_stack(([tCurrentTime]*len(vnSpike),
                                     vnSpike)))

                # Record state after reset
                self.addToRecord(aStateTimeSeries, tCurrentTime, nIdOut=self.__nIdMonitor__)

        # Convert arrays to TimeSeries objects
        mfSpk = np.row_stack(aSpk)
        evOut = TSEvent(mfSpk[:, 0],
                        mfSpk[:, 1],
                        strName='Output',
                        nNumChannels=self.nSize)

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        # Update time
        self._t += tDuration

        return evOut
