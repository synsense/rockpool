"""
Pass-through layer that weights events
"""

import numpy as np
from rockpool.timeseries import TSEvent
from rockpool.nn.layers.layer import Layer
from rockpool.nn.modules.timed_module import astimedmodule

from typing import Optional, Union, Tuple, List, Callable
from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

__all__ = ["PassThroughEvents"]


@astimedmodule(
    parameters=["weights"],
    simulation_parameters=["noise_std", "dt"],
    # states=["_timestep"],
)
class PassThroughEvents(Layer):
    """Pass through events by routing to different channels"""

    def __init__(
        self,
        weights: np.ndarray,
        dt: float = 0.001,
        noise_std: Optional[float] = None,
        name: str = "unnamed",
    ):
        """
        Pass through events by routing to different channels

        :param np.ndarray weights:          Positive integer weight matrix for this layer
        :param float dt:                    Time step duration. Only used for determining evolution period and internal clock.
        :param Optional[float] noise_std:   Not actually used
        :param str name:                    Name of this layer. Default: 'unnamed'
        """

        # - Weights should be of integer type
        weights = np.asarray(weights, int)

        if noise_std is not None:
            warn("Layer `{}`: noise_std is not used in this layer.".format(name))

        # - Initialize parent class
        super().__init__(weights=weights, dt=dt, noise_std=noise_std, name=name)

        self.reset_all()

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Function to evolve the states of this layer given an input

        :param Optional[TSEvent] ts_input:        Input spike trian
        :param Optional[float] duration:          Simulation/Evolution time
        :param Optional[int] num_timesteps:       Number of evolution time steps
        :param bool verbose:                      Currently no effect, just for conformity

        :return `.TSEvent`:                       Output spike series

        """

        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Handle empty inputs
        if ts_input is None or ts_input.times.size == 0:
            return TSEvent(
                None, None, num_channels=self.size
            )  # , t_start=self.t, t_stop=t_end)

        num_inp_events = ts_input.times.size
        # - Boolean raster of input events - each row corresponds to one event (not timepoint)
        inp_channel_raster = np.zeros((num_inp_events, self.size_in), bool)
        inp_channel_raster[np.arange(num_inp_events), ts_input.channels] = True
        # - Integer raster of output events with number of occurences
        #   Each row corresponds to one input event (not timepoint)
        out_channel_raster = inp_channel_raster @ self.weights
        ## -- Extract channels from raster
        # - Number of repetitions for each output event in temporal order
        #   (every self.size events occur simultaneously)
        repetitions = out_channel_raster.flatten()
        # - Output channels corresponding to repetitions
        channel_mask = np.tile(np.arange(self.size), num_inp_events)
        # - Output channel train
        out_channels = np.repeat(channel_mask, repetitions)
        # - Output time trace consits of elements from input time trace
        #   repeated by the number of output events they result in
        num_out_events_per_input_event = np.sum(out_channel_raster, axis=1)
        time_trace_out = np.repeat(ts_input.times, num_out_events_per_input_event)

        t_stop = self.t + self.dt * num_timesteps
        # - Ignore events at t_stop or later.
        use_events = np.logical_and(self.t < time_trace_out, time_trace_out < t_stop)

        # - Output time series
        event_out = TSEvent(
            times=time_trace_out[use_events],
            channels=out_channels[use_events],
            num_channels=self.size,
            t_start=self.t,
            t_stop=t_stop,
            name="transformed event raster",
        )

        # - Update clock
        self._timestep += num_timesteps

        return event_out

    def to_dict(self):
        """Convert parameters of `self` to a dict if they are relevant for reconstructing an identical layer"""
        return super().to_dict()

    @property
    def input_type(self):
        """Returns input type class"""
        return TSEvent

    @property
    def output_type(self):
        """Returns output type class"""
        return TSEvent

    @property
    def weights(self):
        """Returns weights"""
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self.weights = np.asarray(new_weights, int)
