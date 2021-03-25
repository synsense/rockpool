from importlib import util
if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

import torch

from typing import Union, List, Tuple
import numpy as np

from rockpool.nn.modules.torch.torch_module import TorchModule

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_opset9 import floor, div, relu

import torch.nn as nn

import rockpool.parameters as rp

from typing import Iterable, Tuple, Any, Callable


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
        (data,) = ctx.saved_tensors
        grad_input = grad_output * ((data >= (ctx.threshold - ctx.window)).float())
        return grad_input, None, None

    def symbolic(g, data, threshold=1, window=0.5):
        x = relu(g, data)
        x = div(g, x, torch.tensor(threshold))
        x = floor(g, x)
        return x


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
        grad_input[torch.isnan(grad_input)] = 0

        return grad_input, None, None


# helper functions
def calc_bitshift_decay(tau, dt):
    bitsh = torch.log2(tau / dt)
    bitsh[bitsh < 0] = 0
    return bitsh


class LIFLayer(TorchModule):
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
        torch.nn.Module.__init__(self)

        self.n_neurons = rp.SimulationParameter(n_neurons)
        self.n_synapses = rp.SimulationParameter(n_synapses)
        self.batch_size = rp.SimulationParameter(batch_size)

        if isinstance(tau_mem, float):
            self.tau_mem = rp.Parameter(
                torch.from_numpy(np.array([tau_mem])).to(device), "taus"
            )
        else:
            self.tau_mem = rp.Parameter(
                torch.from_numpy(np.array(tau_mem)).to(device), "taus"
            )

        if isinstance(tau_syn, float):
            self.tau_syn = rp.Parameter(
                torch.from_numpy(np.array([tau_syn])).to(device), "taus"
            )
        else:
            self.tau_syn = rp.Parameter(
                torch.from_numpy(np.array(tau_syn)).to(device), "taus"
            )

        self.threshold = rp.Parameter(torch.Tensor([threshold]).to(device))
        self.learning_window = rp.Parameter(torch.Tensor([learning_window]).to(device))
        self.dt = rp.SimulationParameter(dt, "dt")

        self.vmem = rp.State(torch.zeros((self.batch_size, self.n_neurons)).to(device))
        self.isyn = rp.State(
            torch.zeros((self.batch_size, self.n_synapses, self.n_neurons)).to(device)
        )

        self.alpha_mem = rp.Parameter(
            calc_bitshift_decay(self.tau_mem, self.dt).to(device)
        )
        self.alpha_syn = rp.Parameter(
            calc_bitshift_decay(self.tau_syn, self.dt).to(device)
        )

        # determine if cpp lif was compiled
        try:
            import torch_lif_cpp

            self.forward = self.lif_cpp_forward
        except:
            self.threshold_subtract = ThresholdSubtract().apply
            self.bitshift_decay = Bitshift().apply

        # placeholders for recordings
        self.vmem_rec = None
        self.isyn_rec = None

        self.record = False

    def evolve(self, input_data, record: bool = False) -> Tuple[Any, Any, Any]:

        output_data, _, _ = super().evolve(input_data, record)

        states = {
            "Isyn": self.isyn,
            "Vmem": self.vmem,
        }

        record_dict = {
            "Isyn": self.isyn_rec,
            "Vmem": self.vmem_rec,
        }

        return output_data, states, record_dict

    def lif_cpp_forward(self, data):
        import torch_lif_cpp

        out, self.vmem_rec, self.isyn_rec = torch_lif_cpp.forward(
            data.double(),
            self.vmem.double(),
            self.isyn.double(),
            self.alpha_mem.double(),
            self.alpha_syn.double(),
            self.tau_mem.double(),
            self.tau_syn.double(),
            self.threshold.double().item(),
            self.learning_window.double().item(),
            self.record,
        )

        self.vmem = self.vmem_rec[-1]
        self.isyn = self.isyn_rec[-1]

        # Output spike count
        self.n_spikes_out = out

        return out

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

        if self.record:
            self.vmem_rec = torch.zeros(
                (time_steps, n_batches, n_neurons), device=data.device
            )
            self.isyn_rec = torch.zeros(
                (time_steps, n_batches, n_synapses, n_neurons), device=data.device
            )

        for t in range(time_steps):

            # Spike generation
            out = self.threshold_subtract(vmem, threshold, learning_window)
            out_spikes[t] = out

            # Membrane reset
            vmem = vmem - out * threshold
            isyn = isyn + data[t]

            # Leak
            vmem = self.bitshift_decay(vmem, alpha_mem, self.tau_mem)
            isyn = self.bitshift_decay(isyn, alpha_syn, self.tau_syn)

            # State propagation
            vmem = vmem + isyn.sum(1)  # isyn shape (batch, syn, neuron)

            if self.record:
                # recording
                self.vmem_rec[t] = vmem
                self.isyn_rec[t] = isyn

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
