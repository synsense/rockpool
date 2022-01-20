"""
Implement a fast exponential synapse layer using convolution 
"""

# - Rockpool imports
from rockpool.nn.modules import Module
from rockpool.parameters import Parameter, State, SimulationParameter

# - Other imports
import numpy as np
import scipy.signal as sig

from typing import Union

__all__ = ["ExpSyn"]


class ExpSyn(Module):
    """
    Exponential synapse module


    """

    def __init__(
        self,
        shape: tuple = None,
        tau: np.array = None,
        dt: float = 1e-3,
        max_window_length: int = 1e6,
        *args,
        **kwargs,
    ):
        """
        Initialise a module of exponential synapses

        Args:
            shape (Optional[tuple]): The number of synapses in this module ``(N,)``. If not provided, the shape will be extracted from ``tau``.
            tau (Optional[np.ndarray]): Concrete initialisation data for the time constants of the synapses, in seconds. Default: 100 ms for all synapses.
            dt (float): The timestep of this module, in seconds. Default: 1 ms.
            max_window_length (int):
        """
        # - Work out the shape of this module
        if shape is None:
            assert (
                tau is not None
            ), "You must provide either `shape` or else specify concrete parameters."
            shape = np.array(tau).shape

        # - Check that the shape is reasonable
        if np.size(shape) > 1:
            raise ValueError("The `shape` argument must be one-dimensional.")

        # - Call super-class initialisation
        super().__init__(shape=shape, *args, **kwargs)

        # - Record parameters
        self.dt: Union[float, SimulationParameter] = SimulationParameter(dt)
        """ Time step for this module """

        self.max_window_length: Union[int, SimulationParameter] = SimulationParameter(
            max_window_length
        )
        """ Maximum window length for convolution """

        self.tau: Union[np.array, Parameter] = Parameter(
            shape=self.shape,
            data=tau,
            family="taus",
            init_func=lambda s: 100e-3 * np.ones(s),
        )
        """ Time constant of each synapse """

        self.Isyn: Union[np.array, State] = State(
            shape=self.shape, init_func=np.zeros,
        )

    def _init_synapse_windows(self) -> None:
        # - Determine window length required
        window_length = np.clip(
            10 * np.max(self.tau) / self.dt, None, self.max_window_length
        )

        # - Compute window normalised time base
        time_base = [-np.arange(0, window_length) * self.dt] * self.size_out
        time_base = np.array(time_base) / np.atleast_2d(self.tau).T

        # - Compute exponentials
        self._window = np.exp(time_base).T

    def evolve(
        self, input_data: np.array, *args, **kwargs,
    ) -> (np.ndarray, dict, dict):
        # - Compute roll-over decay from last evolution
        rollover = np.zeros(input_data.shape)
        rollover[0, :] = self.Isyn
        rollover = sig.fftconvolve(
            rollover, self._window[: input_data.shape[0]], axes=0, mode="full"
        )

        # - Perform temporal convolution on input
        output_data = (
            sig.fftconvolve(
                input_data, self._window[: input_data.shape[0]], axes=0, mode="full"
            )
            + rollover
        )

        # - Record final state for use in next evolution
        self.Isyn = output_data[input_data.shape[0], :]

        # - Trim output to input shape
        output_data = output_data[: input_data.shape[0]]

        # - Return output along with new state
        return output_data, {"Isyn": self.Isyn}, {}

    @property
    def tau(self) -> float:
        return self._tau

    @tau.setter
    def tau(self, new_value):
        # - Set the value
        self._tau = new_value

        # - Re-generate windows
        self._init_synapse_windows()
