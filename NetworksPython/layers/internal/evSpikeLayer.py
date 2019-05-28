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
        weights: np.ndarray = None,
        vfVThresh: float = 8,
        dt: float = 1,
        name: str = "unnamed",
    ):
        """
        EventCNLayer - Implements a 2D convolutional layer of spiking neurons

        :param nfW:        np.ndarray Weight matrix
        :param vfVThresh: float      Spiking threshold
        :param dt:  float  Time step
        :param name:    str        Name of this layer.
        """
        # Call parent constructor
        FFCLIAF.__init__(self, weights, vfVThresh=vfVThresh, dt=dt, name=name)

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        __, num_timesteps = self._prepare_input(ts_input, duration, num_timesteps)

        # Extract spike data from the input variable
        # TODO: Handle empty input time series
        vSpk = ts_input.times
        vIdInput = ts_input.channels

        # Hold the sate of network at any time step when updated
        aStateTimeSeries = []
        aSpk = []

        # Record initial state of the network
        self._add_to_record(aStateTimeSeries, 0)

        # Local variables
        state = self.state
        vfVThresh = self.vfVThresh
        weights = self.weights

        # Iterate over all input spikes
        for nSpikeIndx in tqdm(range(len(vSpk))):

            tCurrentTime = vSpk[nSpikeIndx]
            nInputId = vIdInput[nSpikeIndx].astype(int)

            # Add input to neurons
            vfUpdate = weights[nInputId]

            # State update (avoiding type cast errors)
            state = state + vfUpdate

            self._add_to_record(
                aStateTimeSeries, tCurrentTime, vnIdOut=self.vnIdMonitor, state=state
            )

            # Check threshold and reset
            vbSpike = state >= vfVThresh
            if vbSpike.any():
                vnSpike, = np.nonzero(vbSpike)

                # Reset membrane state
                state[vbSpike] -= vfVThresh[vbSpike]

                # TODO: The above code has a bug
                # If the threshold goes over 2*vfVThresh this spike will not be detected till the next update.

                # Record spikes
                aSpk.append(np.column_stack(([tCurrentTime] * len(vnSpike), vnSpike)))

                # Record state after reset
                self._add_to_record(
                    aStateTimeSeries,
                    tCurrentTime,
                    vnIdOut=self.vnIdMonitor,
                    state=state,
                )

        # Convert arrays to TimeSeries objects

        mfSpk = np.row_stack(aSpk)
        evOut = TSEvent(
            mfSpk[:, 0], mfSpk[:, 1], name="Output", num_channels=self.size
        )

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        # Update time
        self._timestep += num_timesteps

        return evOut
