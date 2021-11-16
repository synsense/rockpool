"""
Implements the WaveSense architecture from Weidel et al 2021 [1]
[1] 
"""

from rockpool.nn.modules.torch import TorchModule, LinearTorch, LIFTorch, ExpSynTorch
from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool.nn.modules.torch.lif_torch import StepPWL, PeriodicExponential
from rockpool.graph import AliasConnection, GraphHolder, connect_modules

import torch

from typing import List

__all__ = ["WaveBlock", "WaveSenseNet"]


class WaveBlock(TorchModule):
    """
    Implements a single WaveBlock
                          ▲
           To next block  │            ┌────────────┐
       ┌──────────────────┼────────────┤ WaveBlock  ├───┐
       │                  │            └────────────┘   │
       │ Residual path   .─.                            │
       │    ─ ─ ─ ─ ─ ─▶( + )                           │
       │    │            `─'                            │
       │                  ▲                             │
       │    │             │                             │
       │               .─────.                          │
       │    │         ( Spike )                         │
       │               `─────'                          │
       │    │             ▲                             │
       │                  │                             │
       │    │       ┌──────────┐                        │
       │            │  Linear  │                        │
       │    │       └──────────┘         Skip path      │    Skip
       │                  ▲       ┌──────┐    .─────.   │ connections
       │    │             ├──────▶│Linear│──▶( Spike )──┼──────────▶
       │                  │       └──────┘    `─────'   │
       │    │          .─────.                          │
       │              ( Spike )                         │
       │    │          `─────'                          │
       │                 ╲┃╱                            │
       │    │             ┃ Dilation                    │
       │            ┌──────────┐                        │
       │    │       │  Linear  │                        │
       │            └──────────┘                        │
       │    │             ▲                             │
       │     ─ ─ ─ ─ ─ ─ ─│                             │
       └──────────────────┼─────────────────────────────┘
                          │ From previous block
                          │
    """

    def __init__(
        self,
        Nchannels: int = 16,
        Nskip: int = 32,
        dilation: int = None,
        kernel_size: int = 2,
        has_bias: bool = False,
        tau_mem: float = 10e-3,
        base_tau_syn: float = 10e-3,
        threshold: float = 0.0,
        neuron_model=LIFTorch,
        dt: float = 1e-3,
        device: str = "cuda",
        *args,
        **kwargs,
    ):
        """
        Implementation of the WaveBlock as used in the WaveSense model. It received (Nchannels) input channels and outputs (Nchannels, Nskip) channels.

        Args:
            :param int Nchannels:           Dimensionality of the residual connection
            :param int Nskip:               Dimensionality of the skip connection
            :param int dilation:            Determins the synaptic time constant of the dilation layer $dilation * base_tau_syn$
            :param int kernel_size:         Number of synapses the time dilation layer in the WaveBlock
            :param bool has_bias:           If the network can use biases to train
            :param float tau_mem:           Membrane potential time constant of all neurons in WaveSense
            :param float base_tau_syn:      Base synaptic time constant. Each synapse has this time constant, except the second synapse in the dilation layer which caclulates the time constant as $dilations * base_tau_syn$
            :param float threshold:         Threshold of all neurons in WaveSense
            :param NeuronType neuron_model: Neuronmodel to use. Either LIFTorch as standard LIF implementation, LIFBitshiftTorch for hardware compatibility or LIFSlayer for speedup
            :param float dt:                Temporal resolution of the simulation
            :param str device:              Torch device, cuda or cpu
            *args:
            **kwargs:
        """
        # - Determine module shape
        shape = (Nchannels, Nchannels)

        # - Initialise superclass
        super().__init__(
            shape=(Nchannels, Nchannels),
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Add parameters
        self.neuron_model = neuron_model

        # - Dilation layers
        tau_syn = torch.arange(0, dilation * kernel_size, dilation) * base_tau_syn
        tau_syn = torch.clamp(tau_syn, base_tau_syn, tau_syn.max())

        self.lin1 = LinearTorch(
            shape=(Nchannels, Nchannels * kernel_size), has_bias=False, device=device
        )

        self.spk1 = self.neuron_model(
            shape=(Nchannels * kernel_size, Nchannels),
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            has_bias=has_bias,
            threshold=threshold,
            has_rec=False,
            w_rec=None,
            noise_std=0,
            gradient_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
            device=device,
        )

        # - Remapping output layers
        self.lin2_res = LinearTorch(
            shape=(Nchannels, Nchannels), has_bias=False, device=device
        )

        self.spk2_res = self.neuron_model(
            shape=(Nchannels, Nchannels),
            tau_mem=tau_mem,
            tau_syn=tau_syn.min().item(),
            has_bias=has_bias,
            threshold=threshold,
            has_rec=False,
            w_rec=None,
            noise_std=0,
            gradient_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
            device=device,
        )

        # - Skip output layers
        self.lin2_skip = LinearTorch(
            shape=(Nchannels, Nskip), has_bias=False, device=device
        )

        self.spk2_skip = self.neuron_model(
            shape=(Nskip, Nskip),
            tau_mem=tau_mem,
            tau_syn=tau_syn.min().item(),
            has_bias=has_bias,
            threshold=threshold,
            dt=dt,
            device=device,
        )

        # - Internal record dictionary
        self._record_dict = {}

        self.submods = []
        self.submods.append(self.lin1)
        self.submods.append(self.spk1)
        self.submods.append(self.lin2_res)
        self.submods.append(self.spk2_res)
        self.submods.append(self.lin2_skip)
        self.submods.append(self.spk2_skip)

    def forward(self, data: torch.tensor) -> (torch.tensor, dict, dict):
        # Expecting data to be of the format (batch, time, Nchannels)
        (n_batches, t_sim, Nchannels) = data.shape

        # - Pass through dilated weight layer
        out, _, self._record_dict["lin1"] = self.lin1(data, record=True)

        # - Pass through dilated spiking layer
        hidden, _, self._record_dict["spk1"] = self.spk1(
            out, record=True
        )  # (t_sim, n_batches, Nchannels)

        # - Pass through output linear weights
        out_res, _, self._record_dict["lin2_res"] = self.lin2_res(hidden, record=True)

        # - Pass through output spiking layer
        out_res, _, self._record_dict["spk2_res"] = self.spk2_res(out_res, record=True)

        # - Hidden -> skip outputs
        out_skip, _, self._record_dict["lin2_skip"] = self.lin2_skip(
            hidden, record=True
        )

        # - Pass through skip output spiking layer
        out_skip, _, self._record_dict["spk2_skip"] = self.spk2_skip(
            out_skip, record=True
        )

        # - Combine output and residual connections (pass-through)
        res_out = out_res + data

        return res_out, out_skip

    def evolve(self, input, record: bool = False):
        # - Use super-class evolve
        output, new_state, _ = super().evolve(input, record)

        # - Get state record from property
        record_dict = self._record_dict if record else {}

        return output, new_state, record_dict

    def as_graph(self):
        mod_graphs = []

        for mod in self.submods:
            mod_graphs.append(mod.as_graph())

        connect_modules(mod_graphs[0], mod_graphs[1])
        connect_modules(mod_graphs[1], mod_graphs[2])
        connect_modules(mod_graphs[2], mod_graphs[3])  # skip_res
        connect_modules(mod_graphs[1], mod_graphs[4])
        connect_modules(mod_graphs[4], mod_graphs[5])  # skip_add

        AliasConnection(
            mod_graphs[0].input_nodes, mod_graphs[3].output_nodes, name=f"residual_loop"
        )

        multiple_out = mod_graphs[3].output_nodes
        multiple_out.extend(mod_graphs[5].output_nodes)

        return GraphHolder(
            mod_graphs[0].input_nodes,
            multiple_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
        )


class WaveSenseNet(TorchModule):
    """
    Implement a WaveSense network
                                                         Threshold
                                                         on output
                                                .───────.
                                               (Low-pass )────▶
                                                `───────'
                                                    ▲
                                                    │
                                              ┌──────────┐
                                              │  Linear  │
                                              └──────────┘
                                                    ▲
                                                    │
                                                 .─────.
                                                ( Spike )
    ┌──────────────────────┐         Skip        `─────'
    │                      ├┐      outputs          ▲
    │   WaveBlock stack    │├┬───┐                  │
    │                      ││├┬──┤      .─.   ┌──────────┐
    └┬─────────────────────┘││├──┴┬───▶( + )─▶│  Linear  │
     └┬─────────────────────┘││───┘     `─'   └──────────┘
      └┬─────────────────────┘│
       └──────────────────────┘
                   ▲
                   │
                .─────.
               ( Spike )
                `─────'
                   ▲
                   │
             ┌──────────┐
             │  Linear  │
             └──────────┘
                   ▲  Spiking
                   │   input
    """

    def __init__(
        self,
        dilations: List,
        n_classes: int = 2,
        n_channels_in: int = 16,
        n_channels_res: int = 16,
        n_channels_skip: int = 32,
        n_hidden: int = 32,
        kernel_size: int = 2,
        has_bias: bool = False,
        smooth_output: bool = True,
        tau_mem: float = 20e-3,
        base_tau_syn: float = 20e-3,
        tau_lp: float = 20e-3,
        threshold: float = 0.0,
        neuron_model=LIFTorch,
        dt: float = 1e-3,
        device: str = None,
        *args,
        **kwargs,
    ):
        """
        Implementation of the WaveSense network as described in https://arxiv.org/abs/2111.01456.

        Args:
            :param List dilations:          List of dilations which determines the number of WaveBlockes used and the synaptic time constant of the dilation layer $dilations * base_tau_syn$.
            :param int n_classes:           Output dimensionality, usually one per class
            :param int n_channels_in:       Input dimensionality / number of input features
            :param int n_channels_res:      Dimensionality of the residual connection in each WaveBlock
            :param int n_channels_skip:     Dimensionality of the skip connection
            :param int n_hidden:            Number of neurons in the hidden layer of the readout
            :param int kernel_size:         Number of synapses the dilated layer in the WaveBlock
            :param bool has_bias:           If the network can use biases to train
            :param bool smooth_output:      If the output of the network is smoothed with an exponential kernel
            :param float tau_mem:           Membrane potential time constant of all neurons in WaveSense
            :param float base_tau_syn:      Base synaptic time constant. Each synapse has this time constant, except the second synapse in the dilation layer which caclulates the time constant as $dilations * base_tau_syn$
            :param float tau_lp:            Time constant of the smooth output
            :param float threshold:         Threshold of all neurons in WaveSense
            :param NeuronType neuron_model: Neuronmodel to use. Either LIFTorch as standard LIF implementation, LIFBitshiftTorch for hardware compatibility or LIFSlayer for speedup
            :param float dt:                Temporal resolution of the simulation
            :param str device:              Torch device, cuda or cpu
            *args:
            **kwargs:
        """
        # - Determine network shape and initialise
        shape = (n_channels_in, n_classes)
        super().__init__(
            shape=shape, spiking_input=True, spiking_output=True, *args, **kwargs
        )
        self.n_channels_res = n_channels_res
        self.n_channels_skip = n_channels_skip

        self.neuron_model = neuron_model

        # - Input mapping layers
        self.lin1 = LinearTorch(
            shape=(n_channels_in, n_channels_res), has_bias=False, device=device
        )

        self.spk1 = self.neuron_model(
            shape=(n_channels_res, n_channels_res),
            tau_mem=tau_mem,
            tau_syn=base_tau_syn,
            has_bias=has_bias,
            threshold=threshold,
            has_rec=False,
            w_rec=None,
            noise_std=0,
            gradient_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
            device=device,
        )

        # - WaveBlock layers
        self._num_dilations = len(dilations)
        for i, dilation in enumerate(dilations):
            wave = WaveBlock(
                n_channels_res,
                n_channels_skip,
                dilation=dilation,
                kernel_size=kernel_size,
                has_bias=has_bias,
                tau_mem=tau_mem,
                base_tau_syn=base_tau_syn,
                threshold=threshold,
                dt=dt,
                device=device,
            )
            self.__setattr__(f"wave{i}", wave)

        # Dense readout layers
        self.hidden = LinearTorch(
            shape=(n_channels_skip, n_hidden), has_bias=False, device=device
        )

        self.spk2 = self.neuron_model(
            shape=(n_hidden, n_hidden),
            tau_mem=tau_mem,
            tau_syn=base_tau_syn,
            has_bias=has_bias,
            threshold=threshold,
            has_rec=False,
            w_rec=None,
            noise_std=0,
            gradient_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
            device=device,
        )

        self.readout = LinearTorch(
            shape=(n_hidden, n_classes), has_bias=False, device=device
        )

        # - low pass filter is not compatible with xylo unless we give tau_syn 0
        # Smoothing output
        # self.smooth_output = SimulationParameter(smooth_output)
        # """ bool: Perform low-pass filtering of the readout """
        #
        # if smooth_output:
        #     self.lp = ExpSynTorch(n_classes, tau_syn=tau_lp, dt=dt, device=device)

        self.spk_out = self.neuron_model(
            shape=(n_classes, n_classes),
            tau_mem=tau_lp,
            tau_syn=tau_lp,
            has_bias=has_bias,
            threshold=threshold,
            has_rec=False,
            w_rec=None,
            noise_std=0,
            gradient_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
            device=device,
        )

        # - Record dt
        self.dt = SimulationParameter(dt)
        """ float: Time-step in seconds """

        # Dictionary for recording state
        self._record_dict = {}

        self.submods = []
        self.submods.append(self.lin1)
        self.submods.append(self.spk1)

        for index in range(self._num_dilations):
            wave = self.modules()[f"wave{index}"]
            self.submods.append(wave)

        self.submods.append(self.hidden)
        self.submods.append(self.spk2)
        self.submods.append(self.readout)
        self.submods.append(self.spk_out)

    def forward(self, data: torch.Tensor):
        # Expected data shape
        (n_batches, t_sim, n_channels_in) = data.shape

        # - Input mapping layers
        out, _, self._record_dict["lin1"] = self.lin1(data, record=True)

        # Pass through spiking layer
        out, _, self._record_dict["spk1"] = self.spk1(
            out, record=True
        )  # (t_sim, n_batches, Nchannels)

        # Pass through each wave block in turn
        skip = 0
        for wave_index in range(self._num_dilations):
            wave_block = self.modules()[f"wave{wave_index}"]
            (out, skip_new), _, self._record_dict[f"wave{wave_index}"] = wave_block(
                out, record=True
            )
            skip = skip_new + skip

        # Dense layers
        out, _, self._record_dict["hidden"] = self.hidden(skip, record=True)
        out, _, self._record_dict["spk2"] = self.spk2(out, record=True)

        # Final readout layer
        out, _, self._record_dict["readout"] = self.readout(out, record=True)

        # - low pass filter is not compatible with xylo unless we give tau_syn 0
        # # Smooth the output if requested
        # if self.smooth_output:
        #     out, _, self._record_dict["lp"] = self.lp(out, record=True)
        #     self._record_dict["lp_output"] = out.detach()

        out, _, self._record_dict["spk_out"] = self.spk_out(out, record=True)

        return out

    def evolve(self, input_data, record: bool = False):
        output, new_state, _ = super().evolve(input_data, record=record)

        record_dict = self._record_dict if record else {}
        return output, new_state, record_dict

    def trainable_parameters(self):
        return [p for p in list(self.parameters().astorch()) if p.requires_grad]

    def as_graph(self):
        mod_graphs = []

        for mod in self.submods:
            mod_graphs.append(mod.as_graph())

        connect_modules(mod_graphs[0], mod_graphs[1])
        connect_modules(mod_graphs[1], mod_graphs[2])

        block_mod_start = 2

        for i in range(self._num_dilations - 1):
            connect_modules(
                mod_graphs[block_mod_start + i],
                mod_graphs[block_mod_start + i + 1],
                range(self.n_channels_res),
                None,
            )
            AliasConnection(
                mod_graphs[block_mod_start + i].output_nodes[self.n_channels_res :],
                mod_graphs[block_mod_start + i + 1].output_nodes[self.n_channels_res :],
                name="skip_add",
            )

        connect_modules(
            mod_graphs[-5],
            mod_graphs[-4],
            range(self.n_channels_res, self.n_channels_res + self.n_channels_skip),
            None,
        )
        connect_modules(mod_graphs[-4], mod_graphs[-3])
        connect_modules(mod_graphs[-3], mod_graphs[-2])
        connect_modules(mod_graphs[-2], mod_graphs[-1])

        return GraphHolder(
            mod_graphs[0].input_nodes,
            mod_graphs[-1].output_nodes,
            f"{type(self).__name__}_{self.name}_{id(self)}",
        )


# - for quick test, just unmute these lines
# Net = WaveSenseNet(
#     dilations=[2, 4, 8],
#     n_classes=2,
#     n_channels_in=16,
#     n_channels_res=16,
#     n_channels_skip=32,
#     n_hidden=32,
#     kernel_size=2,
# )
#
# from rockpool.devices.xylo import *
#
# WaveSense_graph = Net.as_graph()
# WaveSense_specs = mapper(
#     WaveSense_graph, weight_dtype="float", threshold_dtype="float", dash_dtype="float"
# )
# del WaveSense_specs["mapped_graph"]
# del WaveSense_specs["dt"]
# xylo_conf, is_valid, message = config_from_specification(**WaveSense_specs)
# print("Valid config: ", is_valid, message)
