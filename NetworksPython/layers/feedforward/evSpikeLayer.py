import numpy as np
from tqdm import tqdm
from ...timeseries import TSEvent
from .iaf_cl import FFCLIAF


class EventDrivenSpikingLayer(FFCLIAF):
    '''
    EventCNNLayer: Event driven 2D convolution layer
    '''
    def __init__(self, mfW: np.ndarray = None,
                 vfVThresh: float = 8,
                 tDt: float = 1,
                 strName: str = 'unnamed'):
        """
        EventCNLayer - Implements a 2D convolutional layer of spiking neurons

        :param nfW:        np.ndarray Weight matrix
        :param vfVThresh: float      Spiking threshold
        :param tDt:  float  Time step
        :param strName:    str        Name of this layer.
        """
        # Call parent constructor
        FFCLIAF.__init__(self, mfW, vfVThresh=vfVThresh, tDt=tDt, strName=strName)

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
        self._add_to_record(aStateTimeSeries, 0)

        # Local variables
        vState = self.vState
        vfVThresh = self.vfVThresh
        mfW = self.mfW

        # Iterate over all input spikes
        for nSpikeIndx in tqdm(range(len(vSpk))):

            tCurrentTime = vSpk[nSpikeIndx]
            nInputId = vIdInput[nSpikeIndx].astype(int)

            # Add input to neurons
            vfUpdate = mfW[nInputId]

            # State update (avoiding type cast errors)
            vState = vState + vfUpdate

            self._add_to_record(aStateTimeSeries, tCurrentTime, vnIdOut=self.vnIdMonitor, vState=vState)

            # Check threshold and reset
            vbSpike = (vState >= vfVThresh)
            if vbSpike.any():
                vnSpike, = np.nonzero(vbSpike)

                # Reset membrane state
                vState[vbSpike] -= vfVThresh[vbSpike]

                # TODO: The above code has a bug
                # If the threshold goes over 2*vfVThresh this spike will not be detected till the next update.

                # Record spikes
                aSpk.append(
                    np.column_stack(([tCurrentTime]*len(vnSpike),
                                     vnSpike)))

                # Record state after reset
                self._add_to_record(aStateTimeSeries, tCurrentTime, vnIdOut=self.vnIdMonitor, vState=vState)

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
