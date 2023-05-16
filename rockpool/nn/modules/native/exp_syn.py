"""
Implement a fast exponential synapse layer using convolution 
"""

# - Rockpool imports
from rockpool.nn.modules import Module
from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool.typehints import P_float

# - Other imports
import numpy as np
import scipy.signal as sig

from typing import Union, Optional

__all__ = ["ExpSyn"]


class ExpSyn(Module):
    """
    Exponential synapse module

    This module implements a layer of exponential synapses, operating under the update equations

    .. math::

        I_{syn} = I_{syn} + i(t)
        I_{syn} = I_{syn} * \exp(-dt / \tau)
        I_{syn} = I_{syn} + \sigma \zeta_t

    where :math:`i(t)` is the instantaneous input; :math:`\\tau` is the vector ``(N,)`` of time constants for each synapse in seconds; :math:`dt` is the update time-step in seconds; :math:`\\sigma` is the std. deviation after 1s of a Wiener process.

    This module uses fast convolutional logic to implement the update dynamics.
    """

    def __init__(
        self,
        shape: Union[int, tuple],
        tau: Optional[np.array] = None,
        noise_std: float = 0.0,
        dt: float = 1e-3,
        max_window_length: int = 1e6,
        spiking_input: bool = True,
        spiking_output: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialise a module of exponential synapses

        Args:
            shape (Union[int, tuple]): The number of synapses in this module ``(N,)``.
            tau (Optional[np.ndarray]): Concrete initialisation data for the time constants of the synapses, in seconds. Default: 10 ms individual for all synapses.
            noise_std (float): The std. dev after 1s of noise added independently to each synapse
            dt (float): The timestep of this module, in seconds. Default: 1 ms.
            max_window_length (int): The largest window to use when pre-generating synaptic kernels. Default: 1e6.
        """
        # - Work out the shape of this module
        if np.size(shape) > 1:
            raise ValueError(
                "The `shape` argument must be one-dimensional for an ExpSyn module."
            )

        # - Call super-class initialisation
        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        # - Record parameters
        self.dt: Union[float, SimulationParameter] = SimulationParameter(dt)
        """ Time step for this module """

        self.max_window_length: Union[int, SimulationParameter] = SimulationParameter(
            max_window_length
        )
        """ (int) Maximum window length for convolution """

        # - Initialise noise std dev
        self.noise_std: P_float = SimulationParameter(noise_std, cast_fn=np.array)
        """ (float) Noise std. dev after 1 second """

        self.tau: Union[np.array, Parameter] = Parameter(
            data=tau,
            family="taus",
            shape=[(self.size_in,), ()],
            init_func=lambda s: 10e-3 * np.ones(s),
        )
        """ (np.ndarray) Time constant of each synapse ``(Nin,)`` or ``()`` """

        self.isyn: Union[np.array, State] = State(
            shape=self.shape,
            init_func=np.zeros,
        )

    def _init_synapse_windows(self) -> None:
        # - Determine window length required
        window_length = np.clip(
            10 * np.max(self.tau) / self.dt, None, self.max_window_length
        )

        # - Compute window normalised time base
        time_base = [-np.arange(1, window_length + 1) * self.dt] * self.size_out
        time_base = np.array(time_base) / np.atleast_2d(self.tau).T

        # - Compute exponentials
        self._window = np.exp(time_base).T

    def evolve(
        self,
        input_data: np.array,
        *args,
        **kwargs,
    ) -> (np.ndarray, dict, dict):
        # - Expand states and data over batches
        input_data, (isyn, window) = self._auto_batch(
            input_data, (self.isyn, self._window)
        )
        n_batches, n_timesteps, _ = input_data.shape
        window = np.broadcast_to(
            self._window, (n_batches, self._window.shape[0], self.size_in)
        )

        # - Compute roll-over decay from last evolution
        rollover = np.zeros(input_data.shape)
        rollover[:, 0, :] = isyn
        rollover = sig.fftconvolve(
            rollover,
            window[:, :n_timesteps, :],
            axes=1,
            mode="full",
        )

        # - Perform temporal convolution on input
        output_data = (
            sig.fftconvolve(input_data, window[:, :n_timesteps, :], axes=1, mode="full")
            + rollover
        )

        # - Trim output to input shape
        output_data = output_data[:, :n_timesteps, :]

        # - Add noise
        if self.noise_std > 0.0:
            output_data += (
                self.noise_std * np.sqrt(self.dt) * np.random.randn(*output_data.shape)
            )

        # - Record final state for use in next evolution
        self.isyn = output_data[0, -1, :]

        # - Return output along with new state
        return output_data, self.state(), {}

    @property
    def tau(self) -> float:
        return self._tau

    @tau.setter
    def tau(self, new_value):
        # - Set the value
        self._tau = new_value

        # - Re-generate windows
        self._init_synapse_windows()
