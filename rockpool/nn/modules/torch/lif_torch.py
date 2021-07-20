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
import torch.nn.init as init
import rockpool.parameters as rp
from typing import Optional, Tuple, Any

from rockpool.typehints import FloatVector, P_float, P_tensor

__all__ = ["LIFTorch"]


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

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``0.``).

    :Surrogate signals:

    To facilitate gradient-based training, a surrogate :math:`U(t)` is generated from the membrane potentials of each neuron.

    .. math ::

        U_j = \\textrm{tanh}(V_j + 1) / 2 + .5
    """

    def __init__(
        self,
        shape: tuple = None,
        tau_mem: Optional[FloatVector] = 100e-3,
        tau_syn: Optional[FloatVector] = 50e-3,
        bias: Optional[FloatVector] = 0.0,
        has_bias: bool = True,
        w_syn: torch.Tensor = None,
        w_rec: torch.Tensor = None,
        has_rec: bool = False,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Args:
            shape (tuple): Either a single dimension ``(Nout,)``, which defines a feed-forward layer of LIF modules with equal amounts of synapses and neurons, or two dimensions ``(Nin, Nout)``, which defines a layer of ``Nin`` synapses and ``Nout`` LIF neurons.
            tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
            tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            bias (Optional[FloatVector]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, ``0.0`` will be used by default.
            has_bias (bool): When ``True`` the module provides a trainable bias. Default: ``True``
            w_syn (torch.Tensor): Defines the weights between the synapse outputs and the LIF neuron inputs. Must have shape ``(Nin, Nout)``.
            w_rec (torch.Tensor): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a matrix with shape ``(Nout, Nin)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            has_rec (bool): When ``True`` the module provides a trainable recurent weight matrix. Default ``False``, module is feed-forward.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: ``0.0``
            device: Defines the device on which the model will be processed.
            dtype: Defines the data type of the tensors saved as attributes.
        """
        # - Check shape argument
        if np.size(shape) == 1:
            shape = (np.array(shape).item(),)

        if np.size(shape) > 2:
            raise ValueError(
                "`shape` must be a one- or two-element tuple `(Nin, Nout)`."
            )

        # - Initialise superclass
        super().__init__(
            shape=shape,
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Default tensor construction parameters
        factory_kwargs = {"device": device, "dtype": dtype}

        # - Initialise recurrent weights
        w_rec_shape = tuple(reversed(shape))
        if has_rec:
            self.w_rec: P_tensor = rp.Parameter(
                w_rec,
                shape=w_rec_shape,
                init_func=lambda s: init.kaiming_uniform_(
                    torch.empty(s, **factory_kwargs)
                ),
                family="weights",
            )
            """ (Tensor) Recurrent weights `(Nout, Nin)` """
        else:
            if w_rec is not None:
                raise ValueError("`w_rec` may not be provided if `has_rec` is `False`")

            self.w_rec: torch.Tensor = torch.zeros(w_rec_shape, **factory_kwargs)
            """ (Tensor) Recurrent weights `(Nout, Nin)` """

        self.noise_std: P_float = rp.SimulationParameter(noise_std)
        """ (float) Noise std.dev. injected onto the membrane of each neuron during evolution """

        # - Permit scalar time constant initialisation
        if np.size(tau_mem) == 1:
            tau_mem = torch.ones(self.size_out, **factory_kwargs) * tau_mem

        self.tau_mem: P_tensor = rp.Parameter(
            tau_mem, shape=(self.size_out,), family="taus"
        )
        """ (Tensor) Membrane time constants `(Nout,)` """

        if np.size(tau_syn) == 1:
            tau_syn = torch.ones(self.size_in, **factory_kwargs) * tau_syn

        self.tau_syn: P_tensor = rp.Parameter(
            tau_syn, shape=(self.size_in,), family="taus"
        )
        """ (Tensor) Synaptic time constants `(Nout,)` """

        if has_bias:
            if np.size(bias) == 1:
                bias = torch.ones(self.size_out, **factory_kwargs) * bias

            self.bias: P_tensor = rp.Parameter(
                bias, shape=(self.size_out,), family="bias"
            )
            """ (Tensor) Neuron biases `(Nout)` """
        else:
            self.bias: float = 0.0
            """ (Tensor) Neuron biases `(Nout)` """

        self.w_syn = rp.Parameter(
            w_syn,
            shape=shape,
            init_func=lambda s: init.kaiming_uniform_(torch.empty(s, **factory_kwargs)),
            family="weights",
        )
        """ (Tensor) Input weights `(Nin, Nout)` """

        self.dt: P_float = rp.SimulationParameter(dt)
        """ (float) Euler simulator time-step in seconds"""

        self.isyn: P_tensor = rp.State(torch.zeros(1, self.size_in, **factory_kwargs))
        """ (Tensor) Synaptic currents `(Nin,)` """

        self.vmem: P_tensor = rp.State(
            -1.0 * torch.ones(1, self.size_out, **factory_kwargs)
        )
        """ (Tensor) Membrane potentials `(Nout,)` """

    def evolve(
        self, input_data: torch.Tensor, record: bool = False
    ) -> Tuple[Any, Any, Any]:
        # - Evolve with superclass evolution
        output_data, states, _ = super().evolve(input_data, record)

        # - Build state record
        record_dict = (
            {
                "isyn": self._isyn_rec,
                "vmem": self._vmem_rec,
            }
            if record
            else {}
        )

        return output_data, states, record_dict

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        ----------
        data: Tensor
            Data takes the shape of (batch, time_steps, n_synapses)

        Returns
        -------
        out: Tensor
            Out of spikes with the shape (batch, time_steps, n_neurons)

        """
        n_batches, time_steps, n_synapses = data.shape

        if n_synapses != self.size_in:
            raise ValueError(
                "Input has wrong neuron dimension. It is {}, must be {}".format(
                    n_synapses, self.size_in
                )
            )

        # - Replicate states out by batches
        vmem = torch.ones(n_batches, 1) @ self.vmem
        isyn = torch.ones(n_batches, 1) @ self.isyn
        n_neurons = self.size_out
        alpha = self.dt / self.tau_mem
        beta = torch.exp(-self.dt / self.tau_syn)
        step_pwl = StepPWL.apply

        # - Set up state record and output
        self._vmem_rec = torch.zeros(n_batches, time_steps, n_neurons)
        self._isyn_rec = torch.zeros(n_batches, time_steps, n_synapses)
        out_spikes = torch.zeros(n_batches, time_steps, n_neurons, device=data.device)

        # - Loop over time
        for t in range(time_steps):

            # Integrate input
            isyn = beta * isyn + data[:, t, :]

            # - Membrane potentials
            dvmem = F.linear(isyn, self.w_syn.T) + self.bias - vmem
            vmem = vmem + alpha * dvmem + torch.randn(vmem.shape) * self.noise_std

            self._vmem_rec[:, t, :] = vmem
            self._isyn_rec[:, t, :] = isyn

            out_spikes[:, t, :] = step_pwl(vmem)
            vmem = vmem - out_spikes[:, t, :]

            # - Apply spikes over the recurrent weights
            isyn = isyn + F.linear(out_spikes[:, t, :], self.w_rec.T)

        self.vmem = vmem[0:1, :].detach()
        self.isyn = isyn[0:1, :].detach()

        self._vmem_rec.detach_()
        self._isyn_rec.detach_()

        return out_spikes
