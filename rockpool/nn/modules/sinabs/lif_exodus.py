"""
Implement a LIF Module, using an Exodus backend

Exodus is an accelerated CUDA-based simulator for LIF-like neuron dynamics, supporting gradient calculations.

This package implements the modules :py:class:`LIFExodus` and `LIFMembraneExodus`.
"""

from rockpool.nn.modules.torch.lif_torch import LIFBaseTorch
import torch
import warnings

from rockpool.typehints import *
from rockpool.parameters import Constant

from rockpool.graph import GraphModuleBase

from sinabs.exodus.spike import IntegrateAndFire
from sinabs.exodus.leaky import LeakyIntegrator

from sinabs.activation import Heaviside, SingleExponential

__all__ = ["LIFExodus", "LIFMembraneExodus", "LIFSlayer"]


class LIFExodus(LIFBaseTorch):
    def __init__(
        self,
        shape: tuple,
        tau_mem: P_float = 0.02,
        tau_syn: P_float = 0.05,
        threshold: P_float = 1.0,
        learning_window: P_float = 0.5,
        bias: P_float = 0.0,
        has_rec: bool = False,
        noise_std: P_float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module using the Exodus backend

        Uses the Exodus accelerated CUDA module to implement an LIF neuron. A CUDA device is required to instantiate this module.

        The output of evolving this module is the neuron spike events; synaptic currents and membrane potentials are available using the ``record = True`` argument to :py:meth:`~.LIFExodus.evolve`.

        Warnings:
            Exodus does not currently support training time constants, biases or thresholds. Use :py:class:`LIFTorch` or py:class:`LIFJax` to train those parameters.

            Exodus currently only supports a single membrane time constant.

            Exodus does not support neuron biases.

            Exodus does not support noise injection.

        Examples:
            Instantitate an LIF module with 2 neurons, with 2 synapses each (4 input channels).

            >>> mod = LIFExodus((4, 2))

            Specify the membrane and synapse time constants, as well as time-step ``dt``.

            >>> mod = LIFExodus((4, 2), tau_mem = 30e-3, tau_syn = 10e-3, dt = 10e-3)

        Args:
            shape (tuple): The shape of this module
            tau_syn (float): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            tau_mem (float): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            threshold (float): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            learning_window (float): Cutoff value for the surrogate gradient. Default: 0.5
            dt (float): Time step in seconds. Default: 1 ms.
        """

        assert isinstance(
            tau_mem, float
        ), "Exodus-backed LIF module must have a single membrane time constant"

        assert isinstance(
            threshold, float
        ), "Exodus-backed LIF module must have a single threshold"

        # - Check unused parameters
        if bias != 0.0:
            raise ValueError("`LIFExodus` does not support non-zero biases.")

        if has_rec:
            raise ValueError("`LIFExodus` does not support recurrent weights.")

        if noise_std != 0.0:
            raise ValueError("`LIFExodus` does not support injected noise.")

        # - Initialise superclass
        super().__init__(
            shape=shape,
            tau_syn=Constant(tau_syn),
            tau_mem=Constant(tau_mem),
            threshold=Constant(threshold),
            bias=Constant(0.0),
            has_rec=False,
            noise_std=0.0,
            learning_window=learning_window,
            *args,
            **kwargs,
        )

        # - Assign the surrogate gradient function
        self.spike_generation_fn = Heaviside(self.learning_window)

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
        # - Ensure input data is on GPU
        if not data.is_cuda:
            warnings.warn("Input data was not on a CUDA device. Moving it there now.")
        data = data.to("cuda")

        # - Replicate data and states out by batches
        data, (vmem, isyn, spikes) = self._auto_batch(
            data, (self.vmem, self.isyn, self.spikes)
        )

        # - Get input data size
        (n_batches, time_steps, n_connections) = data.shape

        # - Reshape input data to de-interleave synapses
        data = data.reshape(n_batches, time_steps, self.n_neurons, self.n_synapses)

        # Exponential leak
        # Generate buffer for synaptic current
        isyn_exodus = torch.zeros(
            n_batches * self.n_neurons, self.n_synapses, time_steps
        ).to(data.device)

        # - Broadcast parameters to full size for this module
        beta = self.beta.expand((self.n_neurons, self.n_synapses))
        threshold = self.threshold.expand((self.n_neurons,))
        alpha = self.alpha.expand((self.n_neurons,))

        for syn in range(self.n_synapses):
            # bring data into format expected by exodus
            inp = (
                data[:, :, :, syn]
                .movedim(1, -1)
                .reshape(n_batches * self.n_neurons, time_steps)
            )

            isyn_exodus[:, syn] = LeakyIntegrator.apply(
                inp.contiguous(),
                isyn[:, :, syn].flatten().contiguous(),
                beta[0, syn],
                True,
            )

        spikes, vmem_exodus = IntegrateAndFire.apply(
            isyn_exodus.sum(1),  # input
            threshold[0],  # membrane subtract
            alpha[0],  # alpha
            vmem.squeeze(),  # init state
            spikes.squeeze(),  # last activations
            threshold[0],  # threshold
            None,  # threshold low
            self.spike_generation_fn,
            None if torch.isinf(self.max_spikes_per_dt) else self.max_spikes_per_dt,
        )

        # Subtract spikes from Vmem as exodus subtracts them starting from the next timestep
        vmem_exodus.data = vmem_exodus.data - spikes.data * threshold[0]

        # Bring states to rockpool dimensions
        isyn_exodus = (
            isyn_exodus.reshape(n_batches, self.n_neurons, self.n_synapses, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )
        vmem_exodus = (
            vmem_exodus.reshape(n_batches, self.n_neurons, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )
        spikes = (
            spikes.reshape(n_batches, self.n_neurons, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )

        if self._record:
            self._record_vmem = vmem_exodus
            self._record_isyn = isyn_exodus

        self._record_spikes = spikes

        self.vmem = vmem_exodus[0, -1].detach()
        self.isyn = isyn_exodus[0, -1].detach()
        self.spikes = spikes[0, -1].detach()

        return self._record_spikes


class LIFMembraneExodus(LIFBaseTorch):
    def __init__(
        self,
        shape: tuple,
        tau_syn: P_float = 0.05,
        tau_mem: P_float = 0.02,
        *args,
        **kwargs,
    ):
        """
        Instantiate a module implementing an LIF membrane using the Exodus backend

        Uses the Exodus accelerated CUDA module to implement an LIF neuron membrane. A CUDA device is required to instantiate this module.

        The output of evolving this module is the neuron membrane potentials; synaptic currents are available using the ``record = True`` argument to :py:meth:`~.LIFExodus.evolve`.

        Warnings:
            Exodus does not currently support training time constants or biases.

            Exodus currently only supports a single membrane time constant.

            Exodus does not support neuron biases.

            Exodus does not support noise injection.

        Examples:
            Instantitate an LIF membrane module with 2 neurons, with 2 synapses each (4 input channels).

            >>> mod = LIFMembraneExodus((4, 2))

            Specify the membrane and synapse time constants, as well as time-step ``dt``.

            >>> mod = LIFMembraneExodus((4, 2), tau_mem = 30e-3, tau_syn = 10e-3, dt = 10e-3)

        Args:
            shape (tuple): The shape of this module
            tau_syn (float): An optional array with concrete initialisation data for the synapse time constants. If not provided, 50ms will be used by default.
            tau_mem (float): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            dt (float): Time-step of this module in seconds. Default: 1 ms.
        """

        # - Check input arguments
        assert isinstance(
            tau_mem, float
        ), "Exodus-backed LIF module must have a single membrane time constant"

        # - Remove unused parameters
        unused_arguments = ["threshold", "bias", "has_rec", "noise_std"]
        test_args = [arg in kwargs for arg in unused_arguments]
        if any(test_args):
            error_args = [arg for (arg, t) in zip(unused_arguments, test_args) if t]
            raise TypeError(
                f"The argument(s) {error_args} is/are not used in LIFMembraneExodus."
            )

        # - Initialise superclass
        super().__init__(
            shape=shape,
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(tau_syn),
            bias=Constant(0.0),
            has_rec=False,
            noise_std=0.0,
            *args,
            **kwargs,
        )

        # - Remove LIFBaseTorch attributes that do not apply
        delattr(self, "threshold")
        delattr(self, "bias")
        delattr(self, "learning_window")
        delattr(self, "spikes")
        delattr(self, "spike_generation_fn")
        delattr(self, "max_spikes_per_dt")
        delattr(self, "_record_spikes")
        delattr(self, "_record_irec")
        delattr(self, "_record_U")

        # - Check that CUDA is available
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for exodus-backed modules.")

        # - Move module to CUDA device
        self.cuda()

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

        # - Ensure input data is on GPU
        if not data.is_cuda:
            warnings.warn("Input data was not on a CUDA device. Moving it there now.")
        data = data.to("cuda")

        # - Replicate data and states out by batches
        data, (vmem, isyn) = self._auto_batch(data, (self.vmem, self.isyn))

        # - Get input data size
        (n_batches, time_steps, n_connections) = data.shape

        # - Reshape input data to de-interleave synapses
        data = data.reshape(n_batches, time_steps, self.n_neurons, self.n_synapses)

        # Generate buffer for synaptic current
        isyn_exodus = torch.zeros(
            n_batches * self.n_neurons, self.n_synapses, time_steps
        ).to(data.device)

        # - Broadcast parameters to full size for this module
        beta = self.beta.expand((self.n_neurons, self.n_synapses))
        alpha = self.alpha.expand((self.n_neurons,))

        for syn in range(self.n_synapses):
            # bring data into format expected by exodus
            inp = (
                data[:, :, :, syn]
                .movedim(1, -1)
                .reshape(n_batches * self.n_neurons, time_steps)
            )

            isyn_exodus[:, syn] = LeakyIntegrator.apply(
                inp.contiguous(),
                isyn[:, :, syn].flatten().contiguous(),
                beta[0, syn],
                True,
            )

        vmem_exodus = LeakyIntegrator.apply(
            isyn_exodus.sum(1),  # input
            vmem.flatten().contiguous(),
            alpha[0],  # alpha
            True,
        )

        # Bring states to rockpool dimensions
        isyn_exodus = (
            isyn_exodus.reshape(n_batches, self.n_neurons, self.n_synapses, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )
        vmem_exodus = (
            vmem_exodus.reshape(n_batches, self.n_neurons, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )

        if self._record:
            self._record_vmem = vmem_exodus
            self._record_isyn = isyn_exodus

        self.vmem = vmem_exodus[0, -1].detach()
        self.isyn = isyn_exodus[0, -1].detach()

        return vmem_exodus

    def as_graph(self) -> GraphModuleBase:
        raise NotImplementedError


class LIFSlayer(LIFExodus):
    """DEPRECATED: An LIF module with an Exodus backend"""

    def __init__(self, *args, **kwargs):
        """
        Instantiate an LIF module with an Exodus backend

        Warnings:
            This module is deprecated. Use :py:class:`LIFExodus` instead.
        """
        warnings.warn(
            "This module is deprecated. Use `LIFExodus` instead.", DeprecationWarning
        )
        super().__init__(*args, **kwargs)
