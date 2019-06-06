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
        v_thresh: float = 8,
        dt: float = 1,
        name: str = "unnamed",
    ):
        """
        EventCNLayer - Implements a 2D convolutional layer of spiking neurons

        :param nfW:        np.ndarray Weight matrix
        :param v_thresh: float      Spiking threshold
        :param dt:  float  Time step
        :param name:    str        Name of this layer.
        """
        # Call parent constructor
        FFCLIAF.__init__(self, weights, v_thresh=v_thresh, dt=dt, name=name)

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
        input_times = ts_input.times
        input_ids = ts_input.channels

        # Hold the sate of network at any time step when updated
        state_time_series = []
        spikes = []

        # Record initial state of the network
        self._add_to_record(state_time_series, 0)

        # Local variables
        state = self.state
        v_thresh = self.v_thresh
        weights = self.weights

        # Iterate over all input spikes
        for spike_id in tqdm(range(len(input_times))):

            t_now = input_times[spike_id]
            input_id = input_ids[spike_id].astype(int)

            # Add input to neurons
            update = weights[input_id]

            # State update (avoiding type cast errors)
            state = state + update

            self._add_to_record(
                state_time_series, t_now, id_out=self.monitor_id, state=state
            )

            # Check threshold and reset
            has_spiked = state >= v_thresh
            if has_spiked.any():
                num_spikes, = np.nonzero(has_spiked)

                # Reset membrane state
                state[has_spiked] -= v_thresh[has_spiked]

                # TODO: The above code has a bug
                # If the threshold goes over 2*v_thresh this spike will not be detected till the next update.

                # Record spikes
                spikes.append(np.column_stack(([t_now] * len(num_spikes), num_spikes)))

                # Record state after reset
                self._add_to_record(
                    state_time_series,
                    t_now,
                    id_out=self.monitor_id,
                    state=state,
                )

        # Convert arrays to TimeSeries objects

        spikes = np.row_stack(spikes)
        out_events = TSEvent(
            spikes[:, 0], spikes[:, 1], name="Output", num_channels=self.size
        )

        # TODO: Is there a time series object for this too?
        ts_state = np.array(state_time_series)

        # This is only for debugging purposes. Should ideally not be saved
        self._ts_state = ts_state

        # Update time
        self._timestep += num_timesteps

        return out_events
