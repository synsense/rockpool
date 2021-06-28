"""
Implement a LIF Neuron Module, using a Torch backend
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

from typing import Union, Tuple, Any
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
import torch
import rockpool.parameters as rp

from rockpool.typehints import P_float, P_tensor

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
        (data,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[data < -0.5] = 0
        return grad_input


class LIFNeuronTorch(TorchModule):
    """
    A leaky integrate-and-fire spiking neuron model

    This module implements the dynamics:

    .. math ::

        \\tau_{mem} \\dot{V}_{mem} + V_{mem} = I_{in} + b + \\sigma\\zeta(t)

    where :math:`I_{in}(t)` is a :math:`N` vector containing a continuous input currents for each neuron; :math:`b` is a bias current for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` is the membrane time constant.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0.``, a firing threshold of ``0.``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``0.``).
    """

    def __init__(
        self,
        shape: tuple = None,
        tau_mem: FloatVector = 0.1,
        bias: FloatVector = 0.0,
        has_bias: bool = True,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF Neuron module

        Args:
            shape (tuple): Number of neuron-synapse pairs that will be created. Example: shape = (5,)
            tau_mem (FloatVector): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
            bias (FloatVector): An optional array with concrete initialisation data for the neuron bias currents. If not provided, 0.0 will be used by default.
            has_bias (bool): A flag indicating that the neurons should have a bias. Default: ``True``, neurons have a trainable bias. ``False``: Neurons have a bias fixed to zero.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: 0.0
            device: Defines the device on which the model will be processed.
            dtype: Defines the data type of the tensors saved as attributes.
        """
        # - Check shape argument
        if np.size(shape) == 1:
            shape = (np.array(shape).item(),)

        if np.size(shape) > 1:
            raise ValueError("`shape` must have only one dimension for LIFNeuronTorch")

        # - Initialize super-class
        super().__init__(
            shape=shape,
            spiking_input=False,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Determine arguments for building tensors
        factory_kwargs = {"device": device, "dtype": dtype}

        # - Reset and thresholds
        self._v_thresh: float = 0.0
        self._v_reset: float = -1.0

        # - Intialise parameters
        self.noise_std: P_float = rp.SimulationParameter(noise_std)
        """ (float) Std. Dev. of noise injected into neurons on each time-step """

        if np.size(tau_mem) == 1:
            self.tau_mem: P_tensor = rp.Parameter(
                torch.ones(size=(self.size_out,), **factory_kwargs)
                * np.array(tau_mem).item(),
                shape=(self.size_out,),
                family="taus",
            )
            """ (torch.Tensor) Vector `(N,)` with time constants of each neuron """
        else:
            self.tau_mem: P_tensor = rp.Parameter(
                tau_mem, shape=(self.size_out,), family="taus"
            )
            """ (torch.Tensor) Vector `(N,)` with time constants of each neuron """

        if has_bias:
            if np.size(bias) == 1:
                self.bias = rp.Parameter(
                    torch.ones(size=(self.size_out,), **factory_kwargs)
                    * np.array(bias).item(),
                    shape=(self.size_out,),
                    family="biases",
                )
                """ (torch.Tensor) Vector `(N,)` with bias of each neuron """

            else:
                self.bias: P_tensor = rp.Parameter(
                    bias, shape=(self.size_out,), family="biases"
                )
                """ (torch.Tensor) Vector `(N,)` with bias of each neuron """

        else:
            self.bias = torch.zeros(self.size_out, **factory_kwargs)

        self.dt: P_float = rp.SimulationParameter(dt)
        """ (float) Simulation time-step in seconds """

        self.vmem: P_tensor = rp.State(
            self._v_reset * torch.ones(1, self.size_out, **factory_kwargs)
        )
        """ (torch.Tensor) Vector `(N,)` of neuron membrane potentials """

        # - Attribute for recording state
        self._vmem_rec = None
        """ (torch.Tensor) Record of previous evolution """

    def evolve(
        self, input_data: torch.Tensor, record: bool = False
    ) -> Tuple[Any, Any, Any]:
        # - Call super-class evolve
        output_data, states, record_dict = super().evolve(input_data, record)

        # - Fill record dictionary
        if record:
            record_dict["Vmem"] = self._vmem_rec

        # - Return output
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
        # - Validate data shape
        n_batches, time_steps, n_neurons = data.shape

        if n_neurons != self.size_out:
            raise ValueError(
                f"Input has wrong neuron dimensions. It is {n_neurons}, must be {self.size_out}"
            )

        # - Expand state out by batch dimension
        vmem = torch.ones(n_batches, 1) @ self.vmem

        alpha = self.dt / self.tau_mem
        step_pwl = StepPWL.apply
        noise_std = self.noise_std

        # - Initialise output raster and state record
        out_spikes = torch.zeros(data.shape, device=data.device)
        self._vmem_rec = torch.zeros(data.shape)

        # - Loop over time
        for t in range(time_steps):
            # - Update membrane potential
            dvmem = data[:, t, :] + self.bias - vmem
            vmem = vmem + alpha * dvmem + torch.randn(vmem.shape) * noise_std

            # - Record state
            self._vmem_rec[:, t, :] = vmem

            # - Compute spikes and reset
            out_spikes[:, t, :] = step_pwl(vmem)
            vmem = vmem - out_spikes[:, t, :]

        # - Only retain state for first neuron
        self.vmem = vmem[0:1, :].detach()

        self._vmem_rec.detach_()
        return out_spikes
