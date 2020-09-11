##
# softmaxlayer.py - Implement a softmax layer using spiking inputs
##

import numpy as np
from ...timeseries import TSEvent, TSContinuous
from .iaf_cl import FFCLIAF
from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


def softmax(x: np.ndarray) -> float:
    """
    Compute softmax values for each of scores in x

    :param np.ndarray x:    Vector of values over which to compute softmax
    :return float:          SoftMax of the input values
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class SoftMaxLayer(FFCLIAF):
    """
    *DEPRECATED* A spiking SoftMax layer with spiking inputs and outputs, and constant leak

    This layer implements an approximation of the "soft-max" function used often in deep classification networks.
    """

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        thresh: float = 1e10,  # Just some absurdly large number that will never be reachable
        dt: float = 1.0,
        name: str = "unnamed",
    ):
        """
        Implements a spiking softmax on the inputs

        :param Optional[np.ndarray] weights:    Weight matrix. Default: ``None``
        :param float thresh:          Spiking threshold. Default: ``1e10``
        :param float dt:              Time step. Default: ``1.``
        :param str name:              Name of this layer. Default: ``"unnamed"``
        """

        # Call parent constructor
        FFCLIAF.__init__(self, weights, dt=dt, name=name)
        self.thresh = thresh
        self.__monitor_id__ = None  # Monitor all neurons

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Function to evolve the states of this layer given an input

        :param Optional[TSEvent] ts_input:      Input spike trian
        :param Optional[float] duration:        Simulation/Evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps
        :param bool verbose:                    Currently no effect, just for conformity

        :return TSEvent:                        Output spike series
        """

        # - Use `evolve()` from the base class
        _event_out = FFCLIAF.evolve(
            self, ts_input=ts_input, duration=duration, num_timesteps=num_timesteps
        )
        assert len(_event_out.times) == 0

        # - Analyse states
        state_history_log = self._ts_state[10:]

        # - Convert state data to TimeSeries format
        ts_state = np.zeros((num_timesteps, self.size))
        for t in range(duration):
            data_time_step = state_history_log[(state_history_log[:, 0] == t)]
            ts_state[t] = data_time_step[:, 2]

        # - Compute softmax over the input states
        soft_max = softmax(ts_state)
        ts_out = TSContinuous(
            times=np.arange(duration), samples=soft_max, name="SoftMaxOutput"
        )
        return ts_out

    @property
    def output_type(self):
        """ (`.TSEvent`) Output `.TimeSeries` class of this layer (`.TSEvent`) """
        return TSEvent
