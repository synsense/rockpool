import numpy as np
from ...timeseries import TSEvent
from ..layer import Layer
from typing import Optional, Union, Tuple, List, Callable
from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

__all__ = ["PassThroughEvents"]


class PassThroughEvents(Layer):
    def __init__(
        self,
        weights: np.ndarray,
        dt: float = 0.001,
        noise_std: Optional[float] = None,
        name: str = "unnamed",
    ):
        """
        PassThroughEvents class - route events to different channels

        :param weights:         np.ndarray Positive integer weight matrix for this layer
        :param dt:         float Time step duration. Only used for determining evolution period and internal clock.
        :param noise_std:   float Not actually used
        :param name:     str Name of this layer. Default: 'unnamed'
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
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:                TSEvent  output spike series

        """

        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)
        tEnd = self.t + self.dt * num_timesteps

        # - Handle empty inputs
        if ts_input is None or ts_input.times.size == 0:
            return TSEvent(
                None, None, num_channels=self.size
            )  # , t_start=self.t, t_stop=tEnd)

        nNumInputEvents = ts_input.times.size
        # - Boolean raster of input events - each row corresponds to one event (not timepoint)
        mbInputChannelRaster = np.zeros((nNumInputEvents, self.size_in), bool)
        mbInputChannelRaster[np.arange(nNumInputEvents), ts_input.channels] = True
        # - Integer raster of output events with number of occurences
        #   Each row corresponds to one input event (not timepoint)
        mnOutputChannelRaster = mbInputChannelRaster @ self.weights
        ## -- Extract channels from raster
        # - Number of repetitions for each output event in temporal order
        #   (every self.size events occur simultaneously)
        vnRepetitions = mnOutputChannelRaster.flatten()
        # - Output channels corresponding to vnRepetitions
        vnChannelMask = np.tile(np.arange(self.size), nNumInputEvents)
        # - Output channel train
        vnChannelsOut = np.repeat(vnChannelMask, vnRepetitions)
        # - Output time trace consits of elements from input time trace
        #   repeated by the number of output events they result in
        vnNumOutputEventsPerInputEvent = np.sum(mnOutputChannelRaster, axis=1)
        vtTimeTraceOut = np.repeat(ts_input.times, vnNumOutputEventsPerInputEvent)

        # - Output time series
        tseOut = TSEvent(
            times=vtTimeTraceOut,
            channels=vnChannelsOut,
            num_channels=self.size,
            # t_start=self.t,
            # t_stop=tEnd,
            name="transformed event raster",
        )

        # - Update clock
        self._timestep += num_timesteps

        return tseOut

    @property
    def input_type(self):
        return TSEvent

    @property
    def output_type(self):
        return TSEvent

    @property
    def weights(self):
        return self._mfW

    @weights.setter
    def weights(self, new_weights):
        self.weights = np.asarray(new_weights, int)
