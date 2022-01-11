"""
Implement a LIF Module, using a Slayer backend
"""

from importlib import util

if util.find_spec("sinabs") is None or util.find_spec("sinabs.slayer") is None:
    raise ModuleNotFoundError(
        "'Slayer' backend not found. Modules that rely on Sinabs-Slayer will not be available."
    )

from rockpool.nn.modules.torch.lif_torch import LIFBaseTorch
import torch

from rockpool.typehints import *
from rockpool.parameters import Constant

from sinabs.slayer.spike import SpikeFunctionIterForward
from sinabs.slayer.leaky import LeakyIntegrator

__all__ = ["LIFSlayer"]


class LIFSlayer(LIFBaseTorch):
    def __init__(
        self,
        tau_mem: P_float = 0.02,
        tau_syn: P_float = 0.05,
        threshold: P_float = 1.0,
        has_rec: bool = False,
        noise_std: P_float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module using the Slayer backend

        Args:
            tau_mem (float): An optional array with concrete initialisation data for the membrane time constants. If not provided, 20ms will be used by default.
            threshold (float): An optional array specifying the firing threshold of each neuron. If not provided, ``1.`` will be used by default.
            has_rec (bool): Must be False
            noise_std (float): Must be 0
        """

        assert isinstance(
            tau_mem, float
        ), "Slayer-backed LIF module must have a single membrane time constant"
        assert isinstance(
            threshold, float
        ), "Slayer-backed LIF module must have a single threshold"

        assert has_rec == False, "Slayer-backed LIF module does not support recurrence"
        assert noise_std == 0.0, "Slayer-backed LIF module does not support noise"

        # - Initialise superclass
        super().__init__(
            tau_mem=Constant(tau_mem),
            threshold=Constant(threshold),
            tau_syn=Constant(tau_syn),
            bias=Constant(0.0),
            has_rec=False,
            noise_std=noise_std,
            *args,
            **kwargs,
        )

    # def forward_leak(self, inp: torch.Tensor, alpha, state):
    #
    #     return out_state

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
        # assert data.device == "cuda"
        (n_batches, time_steps, n_connections) = data.shape
        if n_connections != self.size_in:
            raise ValueError(
                "Input has wrong neuron dimension. It is {}, must be {}".format(
                    self.size_in, self.size_out
                )
            )

        # Bring input to expected format for rockpool
        data = data.reshape(n_batches, time_steps, self.n_neurons, self.n_synapses)

        # Replicate states out by batches
        vmem = torch.ones(n_batches, self.n_neurons).to(data.device) * self.vmem
        isyn = (
            torch.ones(n_batches, self.n_neurons, self.n_synapses).to(data.device)
            * self.isyn
        )
        spikes = torch.zeros(n_batches, self.n_neurons).to(data.device) * self.spikes

        # Exponential leak
        # Generate buffer for synaptic current
        isyn_slayer = torch.zeros(
            n_batches * self.n_neurons, self.n_synapses, time_steps
        ).to(data.device)

        beta = torch.broadcast_to(self.beta, (self.n_neurons, self.n_synapses))
        threshold = torch.broadcast_to(self.threshold, (self.n_neurons,))
        alpha = torch.broadcast_to(self.alpha, (self.n_neurons,))

        for syn in range(self.n_synapses):
            # bring data into format expected by slayer
            inp = (
                data[:, :, :, syn]
                .movedim(1, -1)
                .reshape(n_batches * self.n_neurons, time_steps)
            )

            isyn_slayer[:, syn] = LeakyIntegrator.apply(
                inp, isyn[:, :, syn].flatten().contiguous(), beta[0, syn]
            )

        spikes, vmem_slayer = SpikeFunctionIterForward.apply(
            isyn_slayer.sum(1),  # input
            threshold[0],  # membrane subtract
            alpha[0],  # alpha
            vmem.squeeze(),  # init state
            spikes.squeeze(),  # last activations
            threshold[0],  # threshold
            None,  # threshold low
            self.learning_window,  # learning window
            1.0,  # scale grads
        )

        # Bring states to rockpool dimensions
        isyn_slayer = (
            isyn_slayer.reshape(n_batches, self.n_neurons, self.n_synapses, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )
        vmem_slayer = (
            vmem_slayer.reshape(n_batches, self.n_neurons, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )
        spikes = (
            spikes.reshape(n_batches, self.n_neurons, time_steps)
            .movedim(-1, 1)
            .to(data.device)
        )

        vmem_slayer = vmem_slayer - spikes

        # recording
        if self._record:
            # recording
            self._record_Vmem = vmem_slayer
            self._record_Isyn = isyn_slayer

        self._record_spikes = spikes

        self.vmem = vmem_slayer[0, -1].detach()
        self.isyn = isyn_slayer[0, -1].detach()
        self.spikes = spikes[0, -1].detach()

        return self._record_spikes
