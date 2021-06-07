"""
Implement a LIF Module, using a Torch backend
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

__all__ = ["LIFTorch"]

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


class LIFTorch(TorchModule):
    """
    A leaky integrate-and-fire spiking neuron model

    This module implements the dynamics:

    .. math ::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t)

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

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
        n_neurons: int = None,
        tau_mem: Optional[FloatVector] = 0.1,
        tau_syn: Optional[FloatVector] = 0.05,
        bias: Optional[FloatVector] = 0,
        w_rec: torch.Tensor = None,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        device: str ="cpu",
        record: bool = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Args:
            n_neurons (int): Number of neuron-synapse pairs that will be created.
            tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
            tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            bias (Optional[FloatVector]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, 0.0 will be used by default.
            w_rec (Optional[FloatVector]): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(N, N)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: 0.0
            device (str): Defines the device on which the model will be processed. Default: 'cpu'
            record (bool): If set to True, the module records the internal states and returns them with the output. Default: False
        """
        # Initialize class variables

        super().__init__(
            shape=(n_neurons,n_neurons),
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        if w_rec == None:
            self.w_rec = torch.zeros(n_neurons,n_neurons)
        else:
            if w_rec.shape != (n_neurons,n_neurons):
                self.w_rec = rp.Parameter(w_rec)
            else:
                raise ValueError(
                    "Input has wrong dimension. It is {}, must be {}".format(w_rec.shape, (n_neurons,n_neurons))
                )

        self.n_neurons = n_neurons
        self.record = record
        self.v_thresh = 0
        self.v_reset = -1
        self.noise_std = noise_std

        if isinstance(tau_mem, torch.Tensor):
            self.tau_mem = rp.Parameter(tau_mem)
        else:
            self.tau_mem = rp.Parameter(torch.ones(1, n_neurons).to(device)  * tau_mem)

        if isinstance(tau_syn, torch.Tensor):
            self.tau_syn = rp.Parameter(tau_syn)
        else:
            self.tau_syn = rp.Parameter(torch.ones(1, n_neurons).to(device) * tau_syn)

        if isinstance(bias, torch.Tensor):
            self.bias = rp.Parameter(bias)
        else:
            self.bias = rp.Parameter(torch.ones(1, n_neurons).to(device) * bias)

        self.dt = rp.SimulationParameter(dt)

        self.isyn = rp.State(torch.zeros(1, n_neurons))
        self.vmem = rp.State(self.v_reset * torch.ones(1, n_neurons))

        self.alpha = self.dt / self.tau_mem
        self.beta = torch.exp(-self.dt / self.tau_syn)

    def evolve(self, input_data: torch.Tensor, record: bool = False) -> Tuple[Any, Any, Any]:

        output_data = self.forward(input_data)

        states = {
            "Isyn": self.isyn,
            "Vmem": self.vmem,
        }
        if self.record:
            record_dict = {
                "Isyn": self.isyn_rec,
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
        isyn = torch.ones(n_batches,1) @ self.isyn
        bias = torch.ones(n_batches,1) @ self.bias
        v_thresh = self.v_thresh
        v_reset = self.v_reset
        alpha = self.alpha
        beta = self.beta
        step_pwl = StepPWL.apply
        noise_std = self.noise_std

        out_spikes = torch.zeros(data.shape, device=data.device)

        if self.record:
            self.vmem_rec = torch.zeros(data.shape)
            self.isyn_rec = torch.zeros(data.shape)

        for t in range(time_steps):

            # Integrate input
            isyn = beta*isyn + data[:,t,:]

            # - Membrane potentials
            dvmem = isyn + bias - vmem
            vmem = vmem + alpha * dvmem + torch.randn(vmem.shape)*noise_std


            if self.record:
                self.vmem_rec[:,t,:] = vmem
                self.isyn_rec[:,t,:] = isyn

            out_spikes[:,t,:] = step_pwl(vmem)
            vmem = vmem - out_spikes[:,t,:]
            # - Apply spikes over the recurrent weights
            isyn += F.linear(out_spikes[:,t,:], self.w_rec)

        self.vmem = vmem[0:1,:].detach()
        self.isyn = isyn[0:1,:].detach()

        if self.record:
            self.vmem_rec.detach_()
            self.isyn_rec.detach_()

        return out_spikes
