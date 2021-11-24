"""
Implement a LIF Module with bit-shift decay, using a Torch backend
"""

from importlib import util

if util.find_spec("torch") is None:
    raise ModuleNotFoundError(
        "'Torch' backend not found. Modules that rely on Torch will not be available."
    )

from typing import Union, List, Tuple, Callable, Optional, Any
import numpy as np
import torch
from rockpool.nn.modules.torch.lif_torch import LIFTorch, StepPWL, PeriodicExponential 
import rockpool.parameters as rp
from rockpool.typehints import * 

__all__ = ["LIFBitshiftTorch"]


# helper functions
def calc_bitshift_decay(tau, dt):
    bitsh = torch.log2(tau / dt)
    bitsh[bitsh < 0] = 0
    return bitsh


class LIFBitshiftTorch(LIFTorch):
    def __init__(
        self,
        shape: tuple,
        tau_mem: FloatVector,
        tau_syn: FloatVector,
        has_bias: P_bool = True,
        bias: FloatVector = 0.0,
        threshold: FloatVector = 1.0,
        has_rec: P_bool = False,
        w_rec: torch.Tensor = None,
        noise_std: P_float = 0.0,
        spike_generation_fn: Callable = StepPWL, 
        learning_window: P_float = 0.5,
        weight_init_func: Optional[Callable[[Tuple], torch.tensor]] = None,
        dt: P_float = 1e-3,
        device: P_str = None,
        *args,
        **kwargs,
    ):

        """
        Instantiate an LIF module with Bitshift decay.

        Args:
            shape (tuple): Either a single dimension ``(Nout,)``, which defines a feed-forward layer of LIF modules with equal amounts of synapses and neurons, or two dimensions ``(Nin, Nout)``, which defines a layer of ``Nin`` synapses and ``Nout`` LIF neurons.
            tau_mem (Optional[FloatVector]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
            tau_syn (Optional[FloatVector]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            has_bias (bool): When ``True`` the module provides a trainable bias. Default: ``True``
            bias (Optional[FloatVector]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, ``0.0`` will be used by default.
            threshold (FloatVector): An optional array specifying the firing threshold of each neuron. If not provided, ``0.`` will be used by default.
            has_rec (bool): When ``True`` the module provides a trainable recurrent weight matrix. Default ``False``, module is feed-forward.
            w_rec (torch.Tensor): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a matrix with shape ``(Nout, Nin)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: ``0.0``
            spike_generation_fn (Callable): Function to call for spike production. Usually simple threshold crossing. Implements the suroogate gradient function in the backward call. (StepPWL or PeriodicExponential).
            learning_window (float): Cutoff value for the surrogate gradient. 
            weight_init_func (Optional[Callable[[Tuple], torch.tensor]): The initialisation function to use when generating weights. Default: ``None`` (Kaiming initialisation)
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            device: Defines the device on which the model will be processed.
        """


        super().__init__(shape=shape, 
                         tau_mem=tau_mem,
                         tau_syn=tau_syn,
                         has_bias=has_bias,
                         bias=bias,
                         threshold=threshold,
                         has_rec=has_rec,
                         w_rec=w_rec,
                         noise_std=noise_std,
                         spike_generation_fn=spike_generation_fn, 
                         learning_window=learning_window,
                         weight_init_func=weight_init_func,
                         dt=dt,
                         device=device,
                         *args, 
                         **kwargs)

        self.dash_mem: P_tensor = rp.SimulationParameter(
            calc_bitshift_decay(self.tau_mem, self.dt).unsqueeze(1).to(device)
        )
        self.dash_syn: P_tensor = rp.SimulationParameter(
            calc_bitshift_decay(self.tau_syn, self.dt).unsqueeze(1).to(device)
        )


    def _decay_isyn(self, v):
        return  v - (v / (2 ** self.dash_syn))

    def _decay_vmem(self, v):
        return v - (v / (2 ** self.dash_mem))

