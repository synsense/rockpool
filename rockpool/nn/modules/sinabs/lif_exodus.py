"""
Implement a LIF Module, using an Exodus backend

Exodus is an accelerated CUDA-based simulator for LIF-like neuron dynamics, supporting gradient calculations.

This package implements the modules :py:class:`.LIFExodus`, :py:class:`.ExpSynExodus` and :py:class:`.LIFMembraneExodus`.
"""

from rockpool.nn.modules.torch.lif_torch import LIFBaseTorch
import torch
import warnings

from rockpool.typehints import *
from rockpool.parameters import Constant

from rockpool.graph import GraphModuleBase

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

if backend_available("sinabs"):
    from sinabs.activation import Heaviside, SingleExponential

    if backend_available("sinabs.exodus"):
        from sinabs.exodus.spike import IntegrateAndFire
        from sinabs.exodus.leaky import LeakyIntegrator
    else:
        IntegrateAndFire = missing_backend_shim("IntegrateAndFire", "sinabs.exodus")
        LeakyIntegrator = missing_backend_shim("LeakyIntegrator", "sinabs.exodus")

else:
    Heaviside = missing_backend_shim("Heaviside", "sinabs")
    SingleExponential = missing_backend_shim("SingleExponential", "sinabs")


__all__ = ["LIFExodus", "LIFMembraneExodus", "LIFSlayer", "ExpSynExodus"]


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
            Exodus does not currently support training thresholds.

            Exodus does not support noise injection.

        Examples:
            Instantitate an LIF module with 2 neurons, with 2 synapses each (4 input channels).

            >>> mod = LIFExodus((4, 2))

            Specify the membrane and synapse time constants, as well as time-step ``dt``.

            >>> mod = LIFExodus((4, 2), tau_mem = 30e-3, tau_syn = 10e-3, dt = 10e-3)

            Pass the model and data to the same cuda device, since it is required to use CUDA on this module.

            >>> data = torch.ones((1, 10, 4))
            >>> device = 'cuda: 1'
            >>> mod.to(device)
            >>> data = data.to(device)
            >>> output = mod(data)

        Args:
            shape (tuple): The shape of this module
            tau_syn (float): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            tau_mem (float): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            bias (float):
            threshold (float): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            learning_window (float): Cutoff value for the surrogate gradient. Default: 0.5
            dt (float): Time step in seconds. Default: 1 ms.
        """

        if has_rec:
            raise ValueError("`LIFExodus` does not support recurrent weights.")

        if noise_std != 0.0:
            raise ValueError("`LIFExodus` does not support injected noise.")

        # - Initialise superclass
        super().__init__(
            shape=shape,
            tau_syn=tau_syn,
            tau_mem=tau_mem,
            threshold=Constant(threshold),
            bias=bias,
            has_rec=False,
            noise_std=0.0,
            learning_window=learning_window,
            *args,
            **kwargs,
        )

        # - Assign the surrogate gradient function
        self.spike_generation_fn = Heaviside(self.learning_window)

        # - Check that CUDA is available
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for exodus-backed modules.")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        Args:
            data (torch.Tensor): Data takes the shape of (batch, time_steps, n_synapses)

        Returns:
            torch.Tensor: Out of spikes with the shape (batch, time_steps, n_neurons)
        """

        # - Replicate data and states out by batches
        data, (vmem, isyn, spikes) = self._auto_batch(
            data, (self.vmem, self.isyn, self.spikes)
        )

        # - Get input data size
        (n_batches, time_steps, n_connections) = data.shape

        # - Broadcast parameters to full size for this module
        beta = self.beta.expand((n_batches, self.n_neurons, self.n_synapses)).flatten()
        alpha = self.alpha.expand((n_batches, self.n_neurons)).flatten().contiguous()
        membrane_subtract = self.threshold.expand((n_batches, self.n_neurons)).flatten()
        threshold = (
            self.threshold.expand((n_batches, self.n_neurons)).flatten().contiguous()
        )

        # Bring data into format expected by exodus: (batches*neurons*synapses, timesteps)
        data = data.movedim(1, -1).flatten(0, -2)

        # Decay data by one timestep to match xylo behavior
        data = beta.unsqueeze(-1) * data

        # Synaptic dynamics: Calculate I_syn and bring to shape
        # (batches*neurons, synapses, timesteps)
        isyn_exodus = LeakyIntegrator.apply(
            data.contiguous(),  # Input
            beta.contiguous(),  # beta
            isyn.flatten().contiguous(),  # initial state
        ).reshape(-1, self.n_synapses, time_steps)

        # Add bias to isyn_exodus, to be added onto the membrane
        bias = self.bias.reshape((1, -1, 1, 1))
        bias = (
            bias.expand((n_batches, self.n_neurons, self.n_synapses, time_steps))
            .flatten(0, 1)
            .contiguous()
        )
        isyn_with_bias = isyn_exodus + bias

        # Membrane dynamics: Calculate v_mem
        spikes, vmem_exodus = IntegrateAndFire.apply(
            isyn_with_bias.sum(1).contiguous(),  # input
            alpha.contiguous(),  # alpha
            vmem.flatten().contiguous(),  # init state
            threshold,  # threshold
            membrane_subtract.contiguous(),  # membrane subtract
            None,  # threshold low
            self.spike_generation_fn,
            None if torch.isinf(self.max_spikes_per_dt) else self.max_spikes_per_dt,
        )

        # Subtract spikes from Vmem as exodus subtracts them starting from the next timestep
        vmem_exodus.data = vmem_exodus.data - spikes.data * threshold.unsqueeze(-1)

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

        self._record_dict["vmem"] = vmem_exodus
        self._record_dict["isyn"] = isyn_exodus
        self._record_dict["spikes"] = spikes

        self.vmem = vmem_exodus[0, -1].detach()
        self.isyn = isyn_exodus[0, -1].detach()
        self.spikes = spikes[0, -1].detach()

        return self._record_dict["spikes"]


class ExpSynExodus(LIFBaseTorch):
    def __init__(
        self,
        shape: tuple,
        tau: P_float = 0.05,
        noise_std: P_float = 0.0,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Instantiate an exponential synapse module using the Exodus backend

        Uses the Exodus accelerated CUDA module to implement an exponential synapse. A CUDA device is required to instantiate this module.

        The output of evolving this module is the synaptic currents.

        Warning:
            Exodus does not support noise injection.

        Examples:
            Instantitate an exponential synapse module with 2 synapses.

            >>> mod = LIFExodus(2)

            Specify the synaptic time constants, as well as time-step ``dt``.

            >>> mod = LIFExodus(2, tau_syn = 10e-3, dt = 10e-3)

            Specify multiple synaptic time constants.

            >>> mod = LIFExodus(2, tau_syn = [10e-3, 20e-3])

            Pass the model and data to the same cuda device, since it is required to use CUDA on this module.

            >>> data = torch.ones((1, 10, 4))
            >>> device = 'cuda: 1'
            >>> mod.to(device)
            >>> data = data.to(device)
            >>> output = mod(data)


        Args:
            shape (tuple): The shape of this module
            tau_syn (float): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            dt (float): Time step in seconds. Default: 1 ms.
        """

        # - Remove unused parameters
        unused_arguments = ["threshold", "has_rec", "noise_std", "bias", "tau_mem"]
        test_args = [arg in kwargs for arg in unused_arguments]
        if any(test_args):
            error_args = [arg for (arg, t) in zip(unused_arguments, test_args) if t]
            raise TypeError(
                f"The argument(s) {error_args} is/are not used in ExpSynExodus."
            )

        if noise_std != 0.0:
            raise ValueError("`ExpSynExodus` does not support injected noise.")

        # - Initialise superclass
        super().__init__(
            shape=shape,
            tau_syn=tau,
            has_rec=False,
            noise_std=0.0,
            dt=dt,
            *args,
            **kwargs,
        )

        # - Remove LIFBaseTorch attributes that do not apply
        delattr(self, "tau_mem")
        delattr(self, "vmem")
        delattr(self, "threshold")
        delattr(self, "bias")
        delattr(self, "learning_window")
        delattr(self, "spikes")
        delattr(self, "spike_generation_fn")
        delattr(self, "max_spikes_per_dt")

        # - Check that CUDA is available
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for exodus-backed modules.")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        forward method for processing data through this layer
        Adds inputs to the synaptic states

        Args:
            data (torch.Tensor): Data takes the shape of (batch, time_steps, n_synapses)

        Returns:
            torch.Tensor: Out of spikes with the shape (batch, time_steps, n_synapses)
        """

        # - Replicate data and states out by batches
        data, (isyn,) = self._auto_batch(data, (self.isyn,))

        # - Get input data size
        (n_batches, time_steps, n_connections) = data.shape

        # - Broadcast parameters to full size for this module
        beta = self.beta.expand((n_batches, self.n_neurons, self.n_synapses)).flatten()

        # Bring data into format expected by exodus: (batches*neurons*synapses, timesteps)
        data = data.movedim(1, -1).flatten(0, -2)

        # Decay data by one timestep to match xylo behavior
        data = beta.unsqueeze(-1) * data

        # Synaptic dynamics: Calculate I_syn and bring to shape
        # (batches*neurons, synapses, timesteps)
        isyn_exodus = LeakyIntegrator.apply(
            data.contiguous(),  # Input
            beta.contiguous(),  # beta
            isyn.flatten().contiguous(),  # initial state
        ).reshape(-1, self.n_synapses, time_steps)

        # Bring states to rockpool dimensions
        isyn_exodus = (
            isyn_exodus.reshape(n_batches, self.n_neurons, self.n_synapses, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )

        # Save synaptic currents and return
        self._record_dict["isyn"] = isyn_exodus.reshape(
            n_batches, time_steps, self.size_out
        )
        self.isyn = isyn_exodus[0, -1].detach()
        return self._record_dict["isyn"]


class LIFMembraneExodus(LIFBaseTorch):
    def __init__(
        self,
        shape: tuple,
        tau_syn: P_float = 0.05,
        tau_mem: P_float = 0.02,
        bias: P_float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Instantiate a module implementing an LIF membrane using the Exodus backend

        Uses the Exodus accelerated CUDA module to implement an LIF neuron membrane. A CUDA device is required to instantiate this module.

        The output of evolving this module is the neuron membrane potentials; synaptic currents are available using the ``record = True`` argument to :py:meth:`~.LIFExodus.evolve`.

        Warnings:
            Exodus does not support noise injection.

        Examples:
            Instantitate an LIF membrane module with 2 neurons, with 2 synapses each (4 input channels).

            >>> mod = LIFMembraneExodus((4, 2))

            Specify the membrane and synapse time constants, as well as time-step ``dt``.

            >>> mod = LIFMembraneExodus((4, 2), tau_mem = 30e-3, tau_syn = 10e-3, dt = 10e-3)

            Pass the model and data to the same cuda device, since it is required to use CUDA on this module.

            >>> data = torch.ones((1, 10, 4))
            >>> device = 'cuda: 1'
            >>> mod.to(device)
            >>> data = data.to(device)
            >>> output = mod(data)

        Args:
            shape (tuple): The shape of this module
            tau_syn (float): An optional array with concrete initialisation data for the synapse time constants. If not provided, 50ms will be used by default.
            tau_mem (float): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            dt (float): Time-step of this module in seconds. Default: 1 ms.
        """

        # - Remove unused parameters
        unused_arguments = ["threshold", "has_rec", "noise_std"]
        test_args = [arg in kwargs for arg in unused_arguments]
        if any(test_args):
            error_args = [arg for (arg, t) in zip(unused_arguments, test_args) if t]
            raise TypeError(
                f"The argument(s) {error_args} is/are not used in LIFMembraneExodus."
            )

        # - Initialise superclass
        super().__init__(
            shape=shape,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            bias=bias,
            has_rec=False,
            noise_std=0.0,
            *args,
            **kwargs,
        )

        # - Remove LIFBaseTorch attributes that do not apply
        delattr(self, "threshold")
        delattr(self, "learning_window")
        delattr(self, "spikes")
        delattr(self, "spike_generation_fn")
        delattr(self, "max_spikes_per_dt")

        # - Check that CUDA is available
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for exodus-backed modules.")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        forward method for processing data through this layer
        Adds synaptic inputs to the synaptic states and mimics the Leaky Integrate and Fire dynamics

        Args:
            data (torch.Tensor): Data takes the shape of (batch, time_steps, n_synapses)

        Returns:
            torch.Tensor: Out of spikes with the shape (batch, time_steps, n_neurons)
        """

        # - Replicate data and states out by batches
        data, (vmem, isyn) = self._auto_batch(data, (self.vmem, self.isyn))

        # - Get input data size
        (n_batches, time_steps, n_connections) = data.shape

        # - Broadcast parameters to full size for this module
        beta = self.beta.expand((n_batches, self.n_neurons, self.n_synapses)).flatten()
        alpha = self.alpha.expand((n_batches, self.n_neurons)).flatten().contiguous()

        # Bring data into format expected by exodus: (batches*neurons*synapses, timesteps)
        data = data.movedim(1, -1).flatten(0, -2)

        # Decay data by one timestep to match xylo behavior
        data = beta.unsqueeze(-1) * data

        # Synaptic dynamics: Calculate I_syn and bring to shape
        # (batches*neurons, synapses, timesteps)
        isyn_exodus = LeakyIntegrator.apply(
            data.contiguous(),  # Input
            beta.contiguous(),  # beta
            isyn.flatten().contiguous(),  # initial state
        ).reshape(-1, self.n_synapses, time_steps)

        # Add bias to isyn_exodus, to be added onto the membrane
        bias = self.bias.reshape((1, -1, 1, 1))
        bias = (
            bias.expand((n_batches, self.n_neurons, self.n_synapses, time_steps))
            .flatten(0, 1)
            .contiguous()
        )
        isyn_exodus = isyn_exodus + bias

        # Inteagrate onto a membrane
        vmem_exodus = LeakyIntegrator.apply(
            isyn_exodus.sum(1).contiguous(),  # input
            alpha.contiguous(),  # alpha
            vmem.flatten().contiguous(),  # initial state
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

        self._record_dict["vmem"] = vmem_exodus
        self._record_dict["isyn"] = isyn_exodus

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
