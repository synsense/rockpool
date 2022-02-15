"""
Implement a exponential synapse module, using a Torch backend
"""

from typing import Optional, Tuple, Any, Union
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
import torch
import rockpool.parameters as rp

import rockpool.typehints as rt

__all__ = ["ExpSynTorch"]


class ExpSynTorch(TorchModule):
    """
    Exponential synapse module with a Torch backend

    This module implements a layer of exponential synapses, operating under the update equations

    .. math::

        I_{syn} = I_{syn} + i(t)
        I_{syn} = I_{syn} * \exp(-dt / \tau)
        I_{syn} = I_{syn} + \sigma \zeta_t

    where :math:`i(t)` is the instantaneous input; :math:`\\tau` is the vector ``(N,)`` of time constants for each synapse in seconds; :math:`dt` is the update time-step in seconds; :math:`\\sigma` is the std. deviation after 1s of a Wiener process.
    """

    def __init__(
        self,
        shape: Union[tuple, int],
        tau: Optional[rt.FloatVector] = None,
        noise_std: float = 0.0,
        dt: float = 1e-3,
        spiking_input: bool = True,
        spiking_output: bool = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate an exp. synapse module

        Args:
            shape (Union[int, tuple]): The number of synapses in this module ``(N,)``.
            tau (Optional[np.ndarray]): Concrete initialisation data for the time constants of the synapses, in seconds. Default: 10 ms individual for all synapses.
            noise_std (float): The std. dev after 1s of noise added independently to each synapse
            dt (float): The timestep of this module, in seconds. Default: 1 ms.
        """
        # Initialize super class
        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        # - To-float-tensor conversion utility
        to_float_tensor = lambda x: torch.tensor(x).float()

        # - Initialise tau
        self.tau: rt.P_tensor = rp.Parameter(
            tau,
            shape=[(self.size_out,), ()],
            family="taus",
            init_func=lambda s: torch.ones(*s) * 10e-3,
            cast_fn=to_float_tensor,
        )
        """ (torch.Tensor) Time constants of each synapse in seconds ``(N,)`` or ``()`` """

        # - Initialise noise std dev
        self.noise_std: rt.P_tensor = rp.SimulationParameter(
            noise_std, cast_fn=to_float_tensor
        )
        """ (float) Noise std. dev after 1 second """

        # - Initialise state
        self.isyn: rt.P_tensor = rp.State(
            shape=(self.size_out,),
            init_func=lambda s: torch.zeros(*s),
        )
        """ (torch.tensor) Synaptic current state for each synapse ``(1, N)`` """

        # - Store dt
        self.dt: rt.P_float = rp.SimulationParameter(dt)
        """ (float) Simulation time-step in seconds """

    def evolve(
        self, input_data: torch.Tensor, record: bool = False
    ) -> Tuple[Any, Any, Any]:

        # - Evolve the module
        output_data, _, _ = super().evolve(input_data, record)

        # - Return the result of evolution
        return output_data, self.state(), {}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the exponential synapse dynamics

        Parameters
        ----------
        data: Tensor
            Data takes the shape of (batch, time_steps, N)

        Returns
        -------
        out: Tensor
            Out of spikes with the shape (batch, time_steps, N)
        """
        # - Auto-batch over input data
        data, (isyn,) = self._auto_batch(data, (self.isyn,))
        n_batches, time_steps, _ = data.shape

        # - Build a tensor to compute and return internal state
        isyn_ts = torch.zeros(data.shape, device=data.device)

        # - Compute decay factor
        beta = torch.exp(-self.dt / self.tau)
        noise_zeta = self.noise_std * torch.sqrt(torch.tensor(self.dt))

        data = data + noise_zeta * torch.randn(data.shape)

        # - Loop over time
        for t in range(time_steps):
            isyn += data[:, t, :]
            isyn *= beta
            isyn_ts[:, t, :] = isyn

        # - Store the final state
        self.isyn = isyn[0].detach()

        # - Return the evolved synaptic current
        return isyn_ts
