"""
Implement a exponential synapse module, using a Torch backend
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

from typing import Optional, Tuple, Any
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
import torch
import rockpool.parameters as rp

import rockpool.typehints as rt

__all__ = ["ExpSynTorch"]


class ExpSynTorch(TorchModule):
    """
    An exponential synapse model

    This module implements the dynamics:

    .. math ::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t)

        where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`.
    """

    def __init__(
        self,
        shape: tuple = None,
        tau_syn: rt.FloatVector = 50e-3,
        dt: float = 1e-3,
        device: str = None,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        Instantiate an exp. synapse module

        Args:
            shape (tuple): Number of synapses that will be created. Example: shape = (5,).
            tau_syn (Optional[np.ndarray]): An optional array with concrete initialisation data for the synaptic time constants, in seconds. If not provided, 50ms will be used by default.
            dt (float): The time step for the forward-Euler ODE solver, in seconds. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: ``0.0``
            device (str): Defines the device on which the model will be processed. Default: ``None``, use the system default.
            dtype: Defines the torch data type of the tensors to use in this module. Default: ``None``.
        """
        # Initialize class variables
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            shape=shape,
            spiking_input=True,
            spiking_output=False,
            *args,
            **kwargs,
        )

        # - Permit a scalar tau_syn initialisation
        if np.size(tau_syn) == 1:
            tau_syn = torch.ones(self.size_out, **factory_kwargs) * tau_syn

        self.tau_syn: rt.P_tensor = rp.Parameter(
            tau_syn, shape=(self.size_out,), family="taus"
        )
        """ (torch.Tensor) Time constants of each synapse in seconds ``(N,)`` """

        self.isyn: rt.P_tensor = rp.State(
            shape=(
                1,
                self.size_out,
            ),
            init_func=lambda s: torch.zeros(*s, **factory_kwargs),
        )
        """ (torch.tensor) Synaptic current state for each synapse ``(1, N)`` """

        self.dt: rt.P_float = rp.SimulationParameter(dt)
        """ (float) Simulation time-step in seconds """

    def evolve(
        self, input_data: torch.Tensor, record: bool = False
    ) -> Tuple[Any, Any, Any]:

        # - Evolve the module
        output_data, states, _ = super().evolve(input_data, record)

        # - Build a record dictionary
        if record:
            record_dict = {
                "Isyn": self._isyn_rec,
            }
        else:
            record_dict = {}

        # - Return the result of evolution
        return output_data, states, record_dict

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the epxonential synapse dynamics

        Parameters
        ----------
        data: Tensor
            Data takes the shape of (batch, time_steps, N)

        Returns
        -------
        out: Tensor
            Out of spikes with the shape (batch, time_steps, N)
        """
        n_batches, time_steps, n_synapses = data.shape

        if n_synapses != self.size_out:
            raise ValueError(
                f"Input has wrong synapse dimensions. It is {n_synapses}, must be {self.size_out}"
            )

        # - Expand state over batches
        isyn = torch.ones(n_batches, 1) @ self.isyn

        # - Build a tensor to compute and return internal state
        self._isyn_rec = torch.zeros(data.shape, device=data.device)

        beta = torch.exp(-self.dt / self.tau_syn)

        # - Loop over time
        for t in range(time_steps):
            # Integrate input
            isyn = beta * isyn + data[:, t, :]
            self._isyn_rec[:, t, :] = isyn

        # - Store the final state
        self.isyn = isyn[0:1].detach()

        # - Return the evolved synaptic current
        return self._isyn_rec
