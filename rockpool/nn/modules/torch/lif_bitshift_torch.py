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

import rockpool.parameters as rp

from typing import Tuple, Any

from rockpool.typehints import P_int, P_float, P_tensor

__all__ = ["LIFBitshiftTorch"]


class ThresholdSubtract(torch.autograd.Function):
    """
    Subtract from membrane potential on reaching threshold
    """

    @staticmethod
    def forward(ctx, data, threshold=1, window=0.5):
        ctx.save_for_backward(data.clone())
        ctx.threshold = threshold
        ctx.window = window
        return (data > 0) * torch.floor(data / threshold)

    @staticmethod
    def backward(ctx, grad_output):
        (membranePotential,) = ctx.saved_tensors

        vmem_shifted = membranePotential - ctx.threshold / 2
        vmem_periodic = vmem_shifted % ctx.threshold
        vmem_below = vmem_shifted * (membranePotential < ctx.threshold)
        vmem_above = vmem_periodic * (membranePotential >= ctx.threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = (
            torch.exp(-torch.abs(vmem_new - ctx.threshold / 2) / ctx.window)
            / ctx.threshold
        )

        return grad_output * spikePdf, None, None


class Bitshift(torch.autograd.Function):
    """
    Subtract from membrane potential on reaching threshold
    """

    @staticmethod
    def forward(ctx, data, dash, tau):

        scale = 10000
        data_ = (data * scale).float()
        dv = torch.floor((data_ / 2 ** dash)).float()
        dv[dv == 0] = 1 * torch.sign(data_[dv == 0])
        v = (data_ - torch.floor(dv)) / scale
        ctx.save_for_backward(data.clone(), v.clone(), tau)
        return v

    @staticmethod
    def backward(ctx, grad_output):
        (data, v, tau) = ctx.saved_tensors
        grad_input = grad_output * tau

        return grad_input, None, None


# helper functions
def calc_bitshift_decay(tau, dt):
    bitsh = torch.log2(tau / dt)
    bitsh[bitsh < 0] = 0
    return bitsh


class LIFBitshiftTorch(TorchModule):
    def __init__(
        self,
        n_neurons: int,
        n_synapses: int,
        batch_size: int,
        tau_mem: Union[float, List],
        tau_syn: Union[float, List],
        threshold: float,
        learning_window: float,
        dt=1,
        device="cuda",
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        n_neurons: int
            number of neurons
        n_synapses: int
            number of synapses
        tau_mem: float / (n_neurons)
            decay time constant in units of simulation time steps
        tau_syn: float / (n_synapses, n_neurons)
            decay time constant in units of simulation time steps
        threshold: float
            Spiking threshold
        learning_window: float
            Learning window around spike threshold for surrogate gradient calculation
        dt: float
            Resolution of the simulation in seconds.
        """
        # Initialize class variables
        # torch.nn.Module.__init__(self)
        super().__init__(
            shape=(n_neurons * n_synapses * batch_size, n_neurons), *args, **kwargs
        )

        self.n_neurons: P_int = rp.SimulationParameter(n_neurons)
        self.n_synapses: P_int = rp.SimulationParameter(n_synapses)
        self.batch_size: P_int = rp.SimulationParameter(batch_size)

        if isinstance(tau_mem, float):
            self.tau_mem: P_tensor = rp.Parameter(
                torch.from_numpy(np.array([tau_mem])).to(device), "taus"
            )
        else:
            self.tau_mem: P_tensor = rp.Parameter(
                torch.from_numpy(np.array(tau_mem)).to(device), "taus"
            )

        if isinstance(tau_syn, float):
            self.tau_syn: P_tensor = rp.Parameter(
                torch.from_numpy(np.array([tau_syn])).to(device), "taus"
            )
        else:
            self.tau_syn: P_tensor = rp.Parameter(
                torch.from_numpy(np.array(tau_syn)).to(device), "taus"
            )

        self.threshold: P_tensor = rp.Parameter(torch.Tensor([threshold]).to(device))
        self.learning_window: P_tensor = rp.Parameter(
            torch.Tensor([learning_window]).to(device)
        )
        self.dt: P_float = rp.SimulationParameter(dt, "dt")

        self.vmem: P_tensor = rp.State(
            torch.zeros((self.batch_size, self.n_neurons)).to(device)
        )
        self.isyn: P_tensor = rp.State(
            torch.zeros((self.batch_size, self.n_synapses, self.n_neurons)).to(device)
        )

        self.alpha_mem: P_tensor = rp.Parameter(
            calc_bitshift_decay(self.tau_mem, self.dt).to(device)
        )
        self.alpha_syn: P_tensor = rp.Parameter(
            calc_bitshift_decay(self.tau_syn, self.dt).to(device)
        )

        self.propagator_mem: P_tensor = rp.Parameter(
            torch.exp(-self.dt / self.tau_mem).to(device)
        )
        self.propagator_syn: P_tensor = rp.Parameter(
            torch.exp(-self.dt / self.tau_syn).to(device)
        )

        # determine if cpp lif was compiled
        try:
            import torch_lif_cpp

            self.forward = self.lif_cpp_forward
        except:
            self.threshold_subtract = ThresholdSubtract().apply
            self.bitshift_decay = Bitshift().apply

        # placeholders for recordings
        self._record_Vmem = None
        self._record_Isyn = None

        self._record_dict = {}
        self._record = False

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:

        self._record = record

        output_data, _, _ = super().evolve(input_data, record)

        states = {
            "Isyn": self.isyn,
            "Vmem": self.vmem,
        }

        record_dict = (
            {
                "Isyn": self._record_Isyn,
                "Vmem": self._record_Vmem,
            }
            if record
            else {}
        )

        return output_data, states, record_dict

    def lif_cpp_forward(self, data):
        import torch_lif_cpp

        out, self._record_Vmem, self._record_Isyn = torch_lif_cpp.forward(
            data.double(),
            self.vmem.double(),
            self.isyn.double(),
            self.alpha_mem.double(),
            self.alpha_syn.double(),
            self.propagator_mem.double(),
            self.propagator_syn.double(),
            self.threshold.double().item(),
            self.learning_window.double().item(),
            self._record,
        )

        self.vmem = self._record_Vmem[-1]
        self.isyn = self._record_Isyn[-1]

        # Output spike count
        self.n_spikes_out = out

        return out.float()

    def detach(self):
        """
        Detach gradients of the state variables
        Returns
        -------

        """
        self.vmem = self.vmem.detach()
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
            Data takes the shape of (time_steps, batch, n_synapses, n_neurons)

        Returns
        -------
        out: Tensor
            Tensor of spikes with the shape (time_steps, batch, n_neurons)

        """
        (time_steps, n_batches, n_synapses, n_neurons) = data.shape

        vmem = self.vmem
        isyn = self.isyn
        alpha_mem = self.alpha_mem
        alpha_syn = self.alpha_syn
        threshold = self.threshold
        learning_window = self.learning_window
        out_spikes = torch.zeros((time_steps, n_batches, n_neurons), device=data.device)

        if self._record:
            self._record_Vmem = torch.zeros(
                (time_steps, n_batches, n_neurons), device=data.device
            )
            self._record_Isyn = torch.zeros(
                (time_steps, n_batches, n_synapses, n_neurons), device=data.device
            )

        for t in range(time_steps):

            # Spike generation
            out = self.threshold_subtract(vmem, threshold, learning_window)
            out_spikes[t] = out

            # Membrane reset
            vmem = vmem - out * threshold

            if self._record:
                # recording
                self._record_Vmem[t] = vmem
                self._record_Isyn[t] = isyn

            # Integrate input
            isyn = isyn + data[t]

            # Leak
            vmem = self.bitshift_decay(vmem, alpha_mem, self.propagator_mem)
            isyn = self.bitshift_decay(isyn, alpha_syn, self.propagator_syn)

            # State propagation
            vmem = vmem + isyn.sum(1)  # isyn shape (batch, syn, neuron)

        self.vmem = vmem
        self.isyn = isyn

        # Output spike count
        self.n_spikes_out = out_spikes

        return out_spikes

    def inject(self, data):
        """
        Inject static current as used in the signal to spike preprocessing step.
        """
        time_steps = data.shape[0]
        n_batches = data.shape[1]
        out_spikes = torch.zeros((time_steps, self.n_neurons)).to(data.device)
        for t in range(time_steps):
            # threshold crossing
            spikes = (self.vmem // self.threshold).int()
            out_spikes[t] = spikes

            # threshold subtract
            self.vmem = self.vmem - spikes * self.threshold

            # update vmem
            self.vmem = self.vmem + data[t]

            # decay
            self.vmem = self.vmem * self.alpha_mem

        return out_spikes

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        return input_shape
