"""
Implement a exponential synapse module, using a Torch backend
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

from typing import Union, List, Tuple
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
import torch
import rockpool.parameters as rp
from typing import Optional, Tuple, Any

__all__ = ["ExpSynTorch"]

# - Define a float / array type
FloatVector = Union[float, torch.Tensor]

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
        tau_syn: Optional[FloatVector] = 0.05,
        dt: float = 1e-3,
        device = None,
        dtype = None,
        record: bool = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate an exp. synapse module

        Args:
            shape (tuple): Number of synapses that will be created. Example: shape = (5,).
            tau_syn (Optional[np.ndarray]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: 0.0
            device (str): Defines the device on which the model will be processed. Default: 'cpu'
            record (bool): If set to True, the module records the internal states and returns them with the output. Default: False
        """
        # Initialize class variables
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            shape=shape,
            spiking_input=True,
            spiking_output=False,
            *args,
            **kwargs,
        )

        self.n_synapses = shape[0]
        self.record = record

        if isinstance(tau_syn, torch.Tensor):
            self.tau_syn = rp.Parameter(tau_syn)
        else:
            self.tau_syn = rp.Parameter(torch.ones(1, self.n_synapses) * tau_syn, **factory_kwargs)

        self.isyn = rp.State(torch.zeros(1, self.n_synapses, **factory_kwargs))

        self.dt = rp.SimulationParameter(dt)

        self.beta = torch.exp(-self.dt / self.tau_syn)

    def evolve(self, input_data: torch.Tensor, record: bool = False) -> Tuple[Any, Any, Any]:

        output_data = self.forward(input_data)

        states = {
            "Isyn": self.isyn,
        }
        if self.record:
            record_dict = {
                "Isyn": self.isyn_rec,
            }
        else:
            record_dict = {}

        return output_data, states, record_dict

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the epxonential synapse dynamics

        Parameters
        ----------
        data: Tensor
            Data takes the shape of (batch, time_steps, n_synapses)

        Returns
        -------
        out: Tensor
            Out of spikes with the shape (batch, time_steps, n_synapses)

        """
        n_batches, time_steps, n_synapses = data.shape

        if n_synapses != self.n_synapses:
            raise ValueError(
                "Input has wrong synapse dimension. It is {}, must be {}".format(n_synapses, self.n_synapses)
            )

        isyn = torch.ones(n_batches,1) @ self.isyn
        beta = self.beta

        self.isyn_rec = torch.zeros(data.shape, device=data.device)

        for t in range(time_steps):
            # Integrate input
            isyn = beta*isyn + data[:,t,:]
            self.isyn_rec[:,t,:] = isyn

        self.isyn = isyn[0:1].detach()

        return self.isyn_rec
