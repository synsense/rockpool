"""
Implement a LIF Module with bit-shift decay, using a Torch backend
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
from torch.nn.parameter import Parameter
import rockpool.parameters as rp
from typing import Tuple, Any

__all__ = ["LIFTorch"]

class StepPWL(torch.autograd.Function):
    """

    """
    @staticmethod
    def forward(ctx, data):
        ctx.save_for_backward(data)
        return torch.floor(data + 1)

    @staticmethod
    def backward(ctx, grad_output):
        data, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[data < -0.5] = 0
        return grad_input


class ExpSynTorch(torch.nn.Module):
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
        n_synapses = None,
        tau_syn = None,
        bias = None,
        dt = 1e-4,
        device="cpu",
        record = False,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Args:
            shape (tuple): Either a single dimension ``N``, which defines a feed-forward layer of LIF neurons, or two dimensions ``(N, N)``, which defines a recurrent layer of LIF neurons.
            tau_mem (Optional[np.ndarray]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
            tau_syn (Optional[np.ndarray]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            bias (Optional[np.ndarray]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, 0.0 will be used by default.
            w_rec (Optional[np.ndarray]): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(N, N)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: 0.0
        """
        # Initialize class variables

        super().__init__(
            *args,
            **kwargs,
        )

        self.n_synapses = n_synapses

        if isinstance(tau_syn, list) or isinstance(tau_syn, np.ndarray):
            self.tau_syn = Parameter(torch.from_numpy(tau_syn).to(device))
        else:
            self.tau_syn = Parameter(torch.ones(1, n_synapses).to(device) * tau_syn)

        if isinstance(bias, list) or isinstance(bias, np.ndarray):
            self.bias = Parameter(torch.from_numpy(bias).to(device))
        else:
            self.bias = Parameter(torch.ones(1, n_synapses).to(device) * bias)

        self.record = record
        self.dt = dt
        self.register_buffer('isyn', torch.zeros(1, self.n_synapses).to(device))
        self.beta = torch.exp(-self.dt / self.tau_syn)


    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:

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

    def detach(self):
        """
        Detach gradients of the state variables
        Returns
        -------

        """
        self.isyn = self.isyn.detach()
        if hasattr(self, "n_spikes_out"):
            self.n_spikes_out = self.n_spikes_out.detach()

    def forward(self, data: torch.Tensor):
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics dynamics Leaky Integrate and Fire dynamics

        Parameters
        ----------
        data: Tensor
            Data takes the shape of (time_steps, batch, n_synapses, n_synapses)

        Returns
        -------
        out: Tensor
            Tensor of spikes with the shape (time_steps, batch, n_synapses)

        """
        n_batches, n_data, n_synapses = data.shape

        if n_synapses != self.n_synapses:
            raise ValueError(
                "Input has wrong synapse dimension. It is {}, must be {}".format(n_synapses, self.n_synapses)
            )

        state_shape = self.isyn.shape

        if state_shape[0] < n_batches:
            n_new = n_batches-state_shape[0]
            isyn_new = self.isyn[0:1,:].repeat(n_new, 1)
            self.isyn = torch.cat((self.isyn, isyn_new),0)

        if state_shape[0] > n_batches:
            self.isyn = self.isyn[0:n_batches,:]

        isyn = self.isyn
        bias = torch.ones(n_batches,1) @ self.bias
        beta = self.beta

        self.isyn_rec = torch.zeros(data.shape, device=data.device)

        for t in range(n_data):
            # Integrate input
            isyn = beta*isyn + data[:,t,:]
            self.isyn_rec[:,t,:] = isyn

        self.isyn = isyn.detach()

        return self.isyn_rec
