"""
Implement a LIF Neuron Module, using a Torch backend
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
import torch.nn.functional as F
import rockpool.parameters as rp
from typing import Optional, Tuple, Any

__all__ = ["LIFNeuronTorch"]

# - Define a float / array type
FloatVector = Union[float, torch.Tensor]

class StepPWL(torch.autograd.Function):
    """
    Heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param torch.Tensor x: Input value

    :return torch.Tensor: output value and gradient function
    """
    @staticmethod
    def forward(ctx, data):
        ctx.save_for_backward(data)
        return torch.clamp(torch.floor(data + 1), 0)

    @staticmethod
    def backward(ctx, grad_output):
        data, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[data < -0.5] = 0
        return grad_input


class LIFNeuronTorch(TorchModule):
    """
    A leaky integrate-and-fire spiking neuron model

    This module implements the dynamics:

    .. math ::

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{in} + b + \\sigma\\zeta(t)

    where :math:`I_{in}(t)` is a :math:`N` vector containing a continuous signal of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` is the membrane time constant.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{tanh}(V_j + 1) / 2 + .5
    """
    def __init__(
        self,
        shape: tuple = None,
        tau_mem: Optional[FloatVector] = 0.1,
        bias: Optional[FloatVector] = 0,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        device = None,
        dtype = None,
        record: bool = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF Neuron module

        Args:
            shape (tuple): Number of neuron-synapse pairs that will be created. Example: shape = (5,)
            tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
            tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            bias (Optional[FloatVector]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, 0.0 will be used by default.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: 0.0
            device: Defines the device on which the model will be processed.
            dtype: Defines the data type of the tensors saved as attributes.
            record (bool): If set to True, the module records the internal states and returns them with the output. Default: False
        """
        # Initialize class variables

        super().__init__(
            shape=shape,
            spiking_input=False,
            spiking_output=True,
            *args,
            **kwargs,
        )

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.n_neurons = shape[0]
        self.record = record
        self.v_thresh = 0
        self.v_reset = -1
        self.noise_std = noise_std

        if isinstance(tau_mem, torch.Tensor):
            self.tau_mem = rp.Parameter(tau_mem)
        else:
            self.tau_mem = rp.Parameter(torch.ones(1, n_neurons, **factory_kwargs)  * tau_mem)

        if isinstance(bias, torch.Tensor):
            self.bias = rp.Parameter(bias)
        else:
            self.bias = rp.Parameter(torch.ones(1, n_neurons, **factory_kwargs) * bias)

        self.dt = rp.SimulationParameter(dt)

        self.vmem = rp.State(self.v_reset * torch.ones(1, n_neurons, **factory_kwargs))

        self.alpha = self.dt / self.tau_mem

    def evolve(self, input_data: torch.Tensor, record: bool = False) -> Tuple[Any, Any, Any]:

        output_data = self.forward(input_data)

        states = {
            "Vmem": self.vmem,
        }
        if self.record:
            record_dict = {
                "Vmem": self.vmem_rec,
            }
        else:
            record_dict = {}

        return output_data, states, record_dict

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        ----------
        data: Tensor
            Data takes the shape of (batch, time_steps, n_neurons)

        Returns
        -------
        out: Tensor
            Out of spikes with the shape (batch, time_steps, n_neurons)

        """
        n_batches, time_steps, n_neurons = data.shape

        if n_neurons != self.n_neurons:
            raise ValueError(
                "Input has wrong neuron dimension. It is {}, must be {}".format(n_neurons, self.n_neurons)
            )

        vmem = torch.ones(n_batches,1) @ self.vmem
        bias = torch.ones(n_batches,1) @ self.bias
        v_thresh = self.v_thresh
        v_reset = self.v_reset
        alpha = self.alpha
        step_pwl = StepPWL.apply
        noise_std = self.noise_std

        out_spikes = torch.zeros(data.shape, device=data.device)

        if self.record:
            self.vmem_rec = torch.zeros(data.shape)

        for t in range(time_steps):

            # - Membrane potentials
            dvmem = data[:,t,:] + bias - vmem
            vmem = vmem + alpha * dvmem + torch.randn(vmem.shape)*noise_std

            if self.record:
                self.vmem_rec[:,t,:] = vmem

            out_spikes[:,t,:] = step_pwl(vmem)
            vmem = vmem - out_spikes[:,t,:]

        self.vmem = vmem[0:1,:].detach()

        if self.record:
            self.vmem_rec.detach_()

        return out_spikes
