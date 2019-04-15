import numpy as np
from tqdm import tqdm
from ...timeseries import TSEvent
from .iaf_cl import FFCLIAF

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class EventDrivenSpikingLayer(FFCLIAF):
    """
    EventCNNLayer: Event driven 2D convolution layer
    """

    def __init__(
        self,
        mfW: np.ndarray = None,
        vfVThresh: float = 8,
        tDt: float = 1,
        strName: str = "unnamed",
    ):
        """
        EventCNLayer - Implements a 2D convolutional layer of spiking neurons

        :param nfW:        np.ndarray Weight matrix
        :param vfVThresh: float      Spiking threshold
        :param tDt:  float  Time step
        :param strName:    str        Name of this layer.
        """
        # Call parent constructor
        FFCLIAF.__init__(self, mfW, vfVThresh=vfVThresh, tDt=tDt, strName=strName)

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        __, nNumTimeSteps = self._prepare_input(tsInput, tDuration, nNumTimeSteps)

        # Extract spike data from the input variable
        # TODO: Handle empty input time series
        vSpk = tsInput.times
        vIdInput = tsInput.channels

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

            self._add_to_record(
                aStateTimeSeries, tCurrentTime, vnIdOut=self.vnIdMonitor, vState=vState
            )

            # Check threshold and reset
            vbSpike = vState >= vfVThresh
            if vbSpike.any():
                vnSpike, = np.nonzero(vbSpike)

                # Reset membrane state
                vState[vbSpike] -= vfVThresh[vbSpike]

                # TODO: The above code has a bug
                # If the threshold goes over 2*vfVThresh this spike will not be detected till the next update.

                # Record spikes
                aSpk.append(np.column_stack(([tCurrentTime] * len(vnSpike), vnSpike)))

                # Record state after reset
                self._add_to_record(
                    aStateTimeSeries,
                    tCurrentTime,
                    vnIdOut=self.vnIdMonitor,
                    vState=vState,
                )

        # Convert arrays to TimeSeries objects

        mfSpk = np.row_stack(aSpk)
        evOut = TSEvent(
            mfSpk[:, 0], mfSpk[:, 1], name="Output", num_channels=self.nSize
        )

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        # Update time
        self._nTimeStep += nNumTimeSteps

        return evOut
