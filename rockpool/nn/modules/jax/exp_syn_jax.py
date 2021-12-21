"""
An exponential synapse layer, with a Jax backend.
"""

# - Rockpool imports
from rockpool.nn.modules import JaxModule
from rockpool.parameters import Parameter, State, SimulationParameter

# - Other imports
import jax
import jax.numpy as np
from jax.lax import scan

from typing import Union

from rockpool import typehints as rt

__all__ = ["ExpSynJax"]


class ExpSynJax(JaxModule):
    """
    Exponential synapse module with a Jax backend

    This module simulates the dynamics of a number of synapses. The synapses evolve under the dynamics

    .. math::

        I_{syn}(t+1) = \alpha \cdot I_{syn}(t) + inp(t)

        \alpha = \frac{\tau}{\textrm{dt}}

    """

    def __init__(
        self,
        shape: Union[tuple, int],
        tau: Union[rt.FloatVector, rt.P_ndarray] = 100e-3,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Initialise an exponential synapse module

        Args:
            shape (Optional[tuple]): The number of units in this module ``(N,)``.
            tau (Optional[np.ndarray]): Concrete initialisation data to use for the time constants of the synapses. Default: shared 100 ms for all synapses.
            dt (float): The time step for simulation, in seconds. Default: 1 ms
        """
        # - Call super-class initialisation
        super().__init__(
            shape=shape, spiking_input=True, spiking_output=False, *args, **kwargs
        )

        # - Record parameters
        self.dt: Union[float, SimulationParameter] = SimulationParameter(dt)
        """ (float) Time step for this module """

        self.tau: rt.P_ndarray = Parameter(
            data=tau, shape=[(), self.size_out], family="taus", cast_fn=np.array
        )
        """ (np.ndarray) Time constant of each synapse """

        self.Isyn: Union[np.array, State] = State(
            shape=self.size_out, init_func=np.zeros,
        )
        """ (np.ndarray) Synaptic current state """

    def evolve(
        self, input_data: np.array, *args, **kwargs,
    ) -> (np.ndarray, dict, dict):
        # - Pre-compute synapse decay beta
        beta = np.exp(-self.dt / self.tau)

        # - Define synapse dynamics
        def forward(Isyn, input_t):
            Isyn = Isyn * beta + input_t
            return Isyn, Isyn

        # - Scan over the input
        Isyn, output = scan(forward, self.Isyn, input_data)

        # - Return output data and state
        return output, {"Isyn": Isyn}, {}
