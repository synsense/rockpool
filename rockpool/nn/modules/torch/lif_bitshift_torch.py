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

from rockpool.typehints import P_int, P_float, P_tensor, FloatVector, P_bool, P_str

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

        scale = 10000 / torch.max(torch.abs(data))
        if torch.isnan(scale) or torch.isinf(scale):
            scale = 10000

        data_ = (data * scale).float()
        dv = torch.floor((data_ / 2 ** dash)).float()
        dv[dv == 0] = 1 * torch.sign(data_[dv == 0])
        v = (data_ - torch.floor(dv)) / scale
        ctx.save_for_backward(tau)

        return v

    @staticmethod
    def backward(ctx, grad_output):
        [tau] = ctx.saved_tensors
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
        shape: tuple,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        has_bias: P_bool = False,
        bias: FloatVector = None,
        threshold: FloatVector = 1.0,
        learning_window: P_float = 1.0,
        dt: P_float = 1e-3,
        device: P_str = "cuda",
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        shape: tuple 
            Input and output dimensions. (n_neurons * n_synapses, n_neurons)
        tau_mem: float / (n_neurons)
            Decay time constant in units of simulation time steps
        tau_syn: float / (n_synapses, n_neurons)
            Decay time constant in units of simulation time steps
        threshold: float
            Spiking threshold
        learning_window: float
            Learning window around spike threshold for surrogate gradient calculation
        has_bias: bool 
            Bias / current injection to the membrane 
        bias: FloatVector
            Inital values for the bias
        dt: float
            Resolution of the simulation in seconds.
        device:
            Device. Either 'cuda' or 'cpu'.
        """

        # - Check shape argument
        if np.size(shape) == 1:
            shape = (np.array(shape).item(),)

        if np.size(shape) > 2:
            raise ValueError(
                "`shape` must be a one- or two-element tuple `(Nin, Nout)`."
            )

        super().__init__(shape=shape, 
                         spiking_input=True,
                         spiking_output=True,
                         *args, 
                         **kwargs)

        # Initialize class variables
        if isinstance(tau_syn, float):
            n_synapses = 1
        else:
            n_synapses = len(tau_syn)

        self.n_synapses: P_int = rp.SimulationParameter(n_synapses)
        self.n_neurons: P_int = rp.SimulationParameter(shape[0] // n_synapses)

        if isinstance(tau_mem, float):
            self.tau_mem: P_tensor = rp.SimulationParameter(
                torch.from_numpy(np.array([tau_mem])).to(device), "taus"
            )
        else:
            self.tau_mem: P_tensor = rp.SimulationParameter(
                torch.from_numpy(np.array(tau_mem)).to(device), "taus"
            )

        if isinstance(tau_syn, float):
            self.tau_syn: P_tensor = rp.SimulationParameter(
                torch.from_numpy(np.array([tau_syn])).to(device), "taus"
            )
        else:
            self.tau_syn: P_tensor = rp.SimulationParameter(
                torch.from_numpy(np.array(tau_syn)).to(device), "taus"
            )

        if isinstance(threshold, float):
            self.threshold: P_tensor = rp.SimulationParameter(torch.Tensor([threshold]).to(device))
        else:
            self.threshold: P_tensor = rp.SimulationParameter(torch.from_numpy(np.array(threshold)).to(device))

        bias: P_tensor = torch.zeros(self.n_neurons).to(device)
        if has_bias:
            self.bias: P_tensor = rp.Parameter(torch.nn.parameter.Parameter(bias))
        else:
            self.bias: P_tensor = rp.Parameter(bias)

        self.learning_window: P_tensor = rp.SimulationParameter(
            torch.Tensor([learning_window]).to(device)
        )
        self.dt: P_float = rp.SimulationParameter(dt, "dt")

        self.vmem: P_tensor = rp.State(
            torch.zeros(self.n_neurons).to(device)
        )
        self.isyn: P_tensor = rp.State(
            torch.zeros((self.n_synapses, self.n_neurons)).to(device)
        )

        self.alpha_mem: P_tensor = rp.SimulationParameter(
            calc_bitshift_decay(self.tau_mem, self.dt).unsqueeze(1).to(device)
        )
        self.alpha_syn: P_tensor = rp.SimulationParameter(
            calc_bitshift_decay(self.tau_syn, self.dt).unsqueeze(1).to(device)
        )

        self.propagator_mem: P_tensor = rp.SimulationParameter(
            torch.exp(-self.dt / self.tau_mem).unsqueeze(1).to(device)
        )
        self.propagator_syn: P_tensor = rp.SimulationParameter(
            torch.exp(-self.dt / self.tau_syn).unsqueeze(1).to(device)
        )

        # determine if cpp lif was compiled
        try:
            # TODO
            asd
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
            self.bias.double(),
            self.threshold.double().item(),
            self.learning_window.double().item(),
            self._record,
        )

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
            Data takes the shape of (n_batches, time_steps, n_synapses, n_neurons)

        Returns
        -------
        out: Tensor
            Tensor of spikes with the shape (n_batches, time_steps, n_neurons)

        """
        (n_batches, time_steps, n_connections) = data.shape
        if n_connections != self.size_in:
            raise ValueError(
                "Input has wrong neuron dimension. It is {}, must be {}".format(
                    n_synapses, self.size_in
                )
            )

        data = data.reshape(n_batches, time_steps, self.n_synapses, self.n_neurons)

        # - Replicate states out by batches
        vmem = torch.ones(n_batches, self.n_neurons).to(data.device) * self.vmem
        isyn = torch.ones(n_batches, self.n_synapses, self.n_neurons).to(data.device) * self.isyn
        bias = torch.ones(n_batches, self.n_neurons).to(data.device) * self.bias

        alpha_mem = self.alpha_mem
        alpha_syn = self.alpha_syn
        threshold = self.threshold
        learning_window = self.learning_window
        out_spikes = torch.zeros((n_batches, time_steps, self.n_neurons), device=data.device)

        if self._record:
            self._record_Vmem = torch.zeros(
                (n_batches, time_steps, self.n_neurons), device=data.device
            )
            self._record_Isyn = torch.zeros(
                (n_batches, time_steps, self.n_synapses, self.n_neurons), device=data.device
            )

        for t in range(time_steps):

            # Leak
            vmem = self.bitshift_decay(vmem, alpha_mem, self.propagator_mem)
            isyn = self.bitshift_decay(isyn, alpha_syn, self.propagator_syn)

            # Integrate input
            isyn = isyn + data[:, t]

            # State propagation
            vmem = vmem + isyn.sum(1) + bias

            # Spike generation
            out = self.threshold_subtract(vmem, threshold, learning_window)
            out_spikes[:, t] = out

            # Membrane reset
            vmem = vmem - out * threshold

            if self._record:
                # recording
                self._record_Vmem[:, t] = vmem.detach()
                self._record_Isyn[:, t] = isyn.detach()


        self.vmem = vmem[0].detach()
        self.isyn = isyn[0].detach()

        #self._record_Vmem.detach()
        #self._record_Isyn.detach()

        # Output spike count
        self.n_spikes_out = out_spikes

        return out_spikes


    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        return input_shape
