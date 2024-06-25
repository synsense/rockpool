"""
Implement a LIF Neuron Module, using a Torch backend
"""

from typing import Union, Tuple, Any
import numpy as np
from rockpool.nn.modules.torch.torch_module import TorchModule
import torch
import rockpool.parameters as rp

from rockpool.typehints import P_float, P_tensor, P_int

__all__ = ["LIFNeuronTorch"]

# - Define a float / array type
FloatVector = Union[float, torch.Tensor]


class StepPWL(torch.autograd.Function):
    """
    Heaviside step function with piece-wise linear surrogate to use as spike-generation surrogate
    """

    @staticmethod
    def forward(
        ctx,
        x,
        threshold=torch.tensor(1.0),
        window=torch.tensor(0.5),
        max_spikes_per_dt=torch.tensor(2.0**16),
    ):
        ctx.save_for_backward(x, threshold)
        ctx.window = window
        nr_spikes = ((x >= threshold) * torch.floor(x / threshold)).float()
        clamp_bool = (nr_spikes > max_spikes_per_dt).float()
        nr_spikes -= (nr_spikes - max_spikes_per_dt.float()) * clamp_bool
        return nr_spikes

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_threshold = grad_window = grad_max_spikes_per_dt = None

        mask = x >= (threshold - ctx.window)
        if ctx.needs_input_grad[0]:
            grad_x = grad_output / threshold * mask

        if ctx.needs_input_grad[1]:
            grad_threshold = -x * grad_output / (threshold**2) * mask

        return grad_x, grad_threshold, grad_window, grad_max_spikes_per_dt


def tau_to_decay(dt, tau):
    return torch.exp(-dt / tau).to(tau.device)


class LIFNeuronTorch(TorchModule):
    """
    A leaky integrate-and-fire spiking neuron model

    This module implements the dynamics:

    .. math ::

        V_{mem} *= \exp(-dt / \tau_{mem})

        V_{mem} += S_{in} + b + \sigma \zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` (or a weighed spike) for each input channel that emits a spike at time :math:`t`; :math:`b` is a bias current for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` is the membrane time constant.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        V_{mem, j} = V_{mem, j} - \\theta

    Neurons therefore share a common resting potential of ``0.``, individual firing thresholds :math:`\\theta`, and a subtractive reset of :math:`-\\theta`. Neurons each have an optional bias current `.bias` (default: ``0.``).
    """

    def __init__(
        self,
        shape: tuple = None,
        tau_mem: FloatVector = None,
        threshold: FloatVector = None,
        bias: FloatVector = 0.0,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        spike_generation_fn: torch.autograd.Function = StepPWL,
        learning_window: P_float = 0.5,
        max_spikes_per_dt: P_int = torch.tensor(2.0**16),
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF Neuron module

        Args:
            shape (tuple): Number of neuron-synapse pairs that will be created. Example: shape = (5,)
            tau_mem (FloatVector): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            threshold (FloatVector): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            bias (FloatVector): An optional array with concrete initialisation data for the neuron bias currents. If not provided, 0.0 will be used by default.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: `0.0`, no noise.
            spike_generation_fn (Callable): Function to call for spike production. Usually simple threshold crossing. Implements the surrogate gradient function in the backward call. (StepPWL or PeriodicExponential).
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

        self.n_neurons: P_int = rp.SimulationParameter(self.size_out)
        """ (int) Number of neurons """

        # - Reset and thresholds

        self._v_thresh: float = 0.0
        self._v_reset: float = -1.0

        # - Intialise parameters
        self.noise_std: P_float = rp.SimulationParameter(noise_std)
        """ (float) Std. Dev. of noise injected into neurons on each time-step """

        to_float_tensor = lambda x: torch.as_tensor(x, dtype=torch.float)

        self.tau_mem: P_tensor = rp.Parameter(
            tau_mem,
            family="taus",
            shape=[(self.n_neurons,), ()],
            init_func=lambda s: torch.ones(s) * 20e-3,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Membrane time constant for each neuron in seconds `(Nout,)` """

        self.threshold: P_tensor = rp.Parameter(
            threshold,
            shape=[(self.size_out,), ()],
            family="thresholds",
            init_func=torch.ones,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Firing threshold for each neuron `(Nout,)` """

        self.bias: P_tensor = rp.Parameter(
            bias,
            shape=[(self.size_out,), ()],
            family="bias",
            init_func=torch.zeros,
            cast_fn=to_float_tensor,
        )
        """ (Tensor) Neuron biases `(Nout,)` or `()` """

        self.dt: P_float = rp.SimulationParameter(dt)
        """ (float) Simulation time-step in seconds """

        self.vmem: P_tensor = rp.State(
            shape=self.n_neurons, init_func=torch.zeros, cast_fn=to_float_tensor
        )
        """ (Tensor) Membrane potentials `(Nout,)` """

        self.spike_generation_fn: P_Callable = rp.SimulationParameter(
            spike_generation_fn.apply
        )
        """ (Callable) Spike generation function with surrograte gradient """

        self.max_spikes_per_dt: P_float = rp.SimulationParameter(
            max_spikes_per_dt, cast_fn=to_float_tensor
        )
        """ (float) Maximum number of events that can be produced in each time-step """

        self.learning_window: P_tensor = rp.SimulationParameter(
            learning_window,
            cast_fn=to_float_tensor,
        )
        """ (float) Learning window cutoff for surrogate gradient function """

        # - Placeholders for state recordings
        self._record_dict = {}
        self._record = False

    def evolve(
        self, input_data: torch.Tensor, record: bool = False
    ) -> Tuple[Any, Any, Any]:
        # - Keep track of "record" flag for use by `forward` method
        self._record = record

        # - Evolve with superclass evolution
        output_data, _, _ = super().evolve(input_data, record)

        # - Obtain state record dictionary
        record_dict = self._record_dict if record else {}

        # - Clear record in order to avoid non-leaf tensors hanging around
        self._record_dict = {}

        return output_data, self.state(), record_dict

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        Args:
            input_data (torch.Tensor): Data takes the shape of (batch, time_steps, n_neurons)

        Raises:
            ValueError: Input has wrong neuron dimensions.

        Returns:
            torch.Tensor: Out of spikes with the shape (batch, time_steps, n_neurons)
        """

        # - Auto-batch over input data
        input_data, (vmem,) = self._auto_batch(
            input_data,
            (self.vmem,),
            ((self.size_out,),),
        )

        n_batches, n_timesteps, _ = input_data.shape

        # - Set up state record and output
        if self._record:
            self._record_dict["vmem"] = torch.zeros(
                n_batches, n_timesteps, self.size_out
            )
        self._record_dict["spikes"] = torch.zeros(
            n_batches, n_timesteps, self.size_out, device=input_data.device
        )

        noise_zeta = self.noise_std * torch.sqrt(torch.tensor(self.dt))

        # - Generate membrane noise trace
        noise_ts = noise_zeta * torch.randn(
            (n_batches, n_timesteps, self.size_out), device=vmem.device
        )

        alpha = tau_to_decay(self.dt, self.tau_mem)

        # - Loop over time
        for t in range(n_timesteps):
            # Decay membrane state
            vmem *= alpha.to(vmem.device)

            # Input spikes
            vmem += input_data[:, t, :]

            # Integrate membrane state and apply noise
            vmem = vmem + noise_ts[:, t, :] + self.bias

            # - Spike generation
            spikes = self.spike_generation_fn(
                vmem, self.threshold, self.learning_window, self.max_spikes_per_dt
            )

            # - Apply subtractive membrane reset
            vmem = vmem - spikes * self.threshold

            # - Maintain state record
            if self._record:
                self._record_dict["vmem"][:, t] = vmem

            # - Maintain output spike record
            self._record_dict["spikes"][:, t] = spikes

        # - Update states
        self.vmem = vmem[0].detach()

        # - Return output
        return self._record_dict["spikes"]
