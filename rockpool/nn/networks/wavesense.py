"""
Implements the WaveSense architecture for temporal signal processing

The WaveSense architecture is described in Weidel et al 2021 [https://arxiv.org/abs/2111.01456]

.. rubric:: See Also

:ref:`/tutorials/wavesense_training.ipynb`

"""

from rockpool.nn.modules import TorchModule, LinearTorch, LIFTorch, ExpSynTorch
from rockpool.parameters import Parameter, State, SimulationParameter, Constant
from rockpool.nn.modules.torch.lif_torch import StepPWL, PeriodicExponential
from rockpool.graph import AliasConnection, GraphHolder, connect_modules

import torch

from typing import List, Union, Callable, Optional
from rockpool.typehints import P_tensor

__all__ = ["WaveBlock", "WaveSenseNet"]


class WaveSenseBlock(TorchModule):
    """
    Implements a single WaveSenseBlock
                          ▲
           To next block  │       ┌─────────────────┐
       ┌──────────────────┼───────┤ WaveSenseBlock  ├───┐
       │                  │       └─────────────────┘   │
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
        bias: P_tensor = Constant(0.0),
        tau_mem: float = Constant(10e-3),
        base_tau_syn: float = Constant(10e-3),
        threshold: float = Constant(1.0),
        neuron_model: TorchModule = LIFTorch,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Implementation of the WaveBlock as used in the WaveSense model. It received (Nchannels) input channels and outputs (Nchannels, Nskip) channels.

        Args:
            :param int Nchannels:           Dimensionality of the residual connection. Default: ``16``
            :param int Nskip:               Dimensionality of the skip connection. Default: ``32``
            :param int dilation:            Determines the synaptic time constant of the dilation layer $dilation * base_tau_syn$. Default: ``None``
            :param int kernel_size:         Number of synapses the time dilation layer in the WaveBlock. Default: ``2``
            :param P_tensor bias:           Bias for the network to train. Default: No trainable bias
            :param float tau_mem:           Membrane potential time constant of all neurons in WaveSense. Default: 10ms
            :param float base_tau_syn:      Base synaptic time constant. Each synapse has this time constant, except the second synapse in the dilation layer which caclulates the time constant as $dilations * base_tau_syn$. Default: 10ms
            :param float threshold:         Threshold of all spiking neurons. Default: `0.`
            :param TorchModule neuron_model: Neuron model to use. Either :py:class:`.LIFTorch` as standard LIF implementation, :py:class:`.LIFBitshiftTorch` for hardware compatibility or :py:class:`.LIFSlayer` for speedup
            :param float dt:                Temporal resolution of the simulation. Default: 1ms
        """
        # - Initialise superclass
        super().__init__(
            shape=(Nchannels, Nchannels),
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Add parameters
        self.neuron_model: Union[Callable, SimulationParameter] = SimulationParameter(
            neuron_model
        )
        """ Neuron model used by this WaveSense network """

        # - Dilation layers
        tau_syn = torch.arange(0, dilation * kernel_size, dilation) * base_tau_syn
        tau_syn = torch.clamp(tau_syn, base_tau_syn, tau_syn.max()).repeat(Nchannels, 1)

        self.lin1 = LinearTorch(
            shape=(Nchannels, Nchannels * kernel_size), has_bias=False
        )
        with torch.no_grad():
            self.lin1.weight.data = self.lin1.weight.data * dt / tau_syn.max()

        self.spk1 = self.neuron_model(
            shape=(Nchannels * kernel_size, Nchannels),
            tau_mem=Constant(tau_syn.max().item()),
            tau_syn=Constant(tau_syn),
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
        )

        # - Remapping output layers
        self.lin2_res = LinearTorch(shape=(Nchannels, Nchannels), has_bias=False)
        with torch.no_grad():
            self.lin2_res.weight.data = self.lin2_res.weight.data * dt / tau_syn.min()

        self.spk2_res = self.neuron_model(
            shape=(Nchannels, Nchannels),
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(tau_syn.min()),
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
        )

        # - Skip output layers
        self.lin2_skip = LinearTorch(shape=(Nchannels, Nskip), has_bias=False)
        with torch.no_grad():
            self.lin2_skip.weight.data = self.lin2_skip.weight.data * dt / tau_syn.min()

        self.spk2_skip = self.neuron_model(
            shape=(Nskip, Nskip),
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(tau_syn.min()),
            bias=bias,
            threshold=Constant(threshold),
            dt=dt,
        )

        # - Internal record dictionary
        self._record = False
        self._record_dict = {}

    def forward(self, data: torch.tensor) -> (torch.tensor, dict, dict):
        # Expecting data to be of the format (batch, time, Nchannels)
        (n_batches, t_sim, Nchannels) = data.shape

        # - Pass through dilated weight layer
        out, _, self._record_dict["lin1"] = self.lin1(data, record=self._record)
        self._record_dict["lin1_output"] = out if self._record else []

        # - Pass through dilated spiking layer
        hidden, _, self._record_dict["spk1"] = self.spk1(
            out, record=self._record
        )  # (t_sim, n_batches, Nchannels)
        self._record_dict["spk1_output"] = out if self._record else []

        # - Pass through output linear weights
        out_res, _, self._record_dict["lin2_res"] = self.lin2_res(
            hidden, record=self._record
        )
        self._record_dict["lin2_res_output"] = out_res if self._record else []

        # - Pass through output spiking layer
        out_res, _, self._record_dict["spk2_res"] = self.spk2_res(
            out_res, record=self._record
        )
        self._record_dict["spk2_res_output"] = out_res if self._record else []

        # - Hidden -> skip outputs
        out_skip, _, self._record_dict["lin2_skip"] = self.lin2_skip(
            hidden, record=self._record
        )
        self._record_dict["lin2_skip_output"] = out_skip if self._record else []

        # - Pass through skip output spiking layer
        out_skip, _, self._record_dict["spk2_skip"] = self.spk2_skip(
            out_skip, record=self._record
        )
        self._record_dict["spk2_skip_output"] = out_skip if self._record else []

        # - Combine output and residual connections (pass-through)
        res_out = out_res + data

        return res_out, out_skip

    def evolve(self, input, record: bool = False):

        self._record = record

        # - Use super-class evolve
        output, new_state, _ = super().evolve(input, self._record)

        # - Get state record from property
        record_dict = self._record_dict if self._record else {}

        return output, new_state, record_dict

    def as_graph(self):
        mod_graphs = []

        for mod in self.modules().values():
            mod_graphs.append(mod.as_graph())

        connect_modules(mod_graphs[0], mod_graphs[1])
        connect_modules(mod_graphs[1], mod_graphs[2])
        connect_modules(mod_graphs[2], mod_graphs[3])  # skip_res
        connect_modules(mod_graphs[1], mod_graphs[4])
        connect_modules(mod_graphs[4], mod_graphs[5])  # skip_add

        AliasConnection(
            mod_graphs[0].input_nodes,
            mod_graphs[3].output_nodes,
            name=f"residual_loop",
            computational_module=None,
        )

        multiple_out = mod_graphs[3].output_nodes
        multiple_out.extend(mod_graphs[5].output_nodes)

        return GraphHolder(
            mod_graphs[0].input_nodes,
            multiple_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
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
    │ WaveSenseBlock stack │├┬───┐                  │
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
        bias: P_tensor = Constant(0.0),
        smooth_output: bool = True,
        tau_mem: float = Constant(20e-3),
        base_tau_syn: float = Constant(20e-3),
        tau_lp: float = Constant(20e-3),
        threshold: float = Constant(1.0),
        neuron_model: TorchModule = LIFTorch,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Implementation of the WaveSense network as described in https://arxiv.org/abs/2111.01456.

        Args:
            :param List dilations:          List of dilations which determines the number of WaveBlockes used and the synaptic time constant of the dilation layer $dilations * base_tau_syn$.
            :param int n_classes:           Output dimensionality, usually one per class. Default: ``2``
            :param int n_channels_in:       Input dimensionality / number of input features. Default: ``16``
            :param int n_channels_res:      Dimensionality of the residual connection in each WaveBlock. Default: ``16``
            :param int n_channels_skip:     Dimensionality of the skip connection. Default: ``32``
            :param int n_hidden:            Number of neurons in the hidden layer of the readout. Default: ``32``
            :param int kernel_size:         Number of synapses the dilated layer in the WaveBlock. Default: ``2``
            :param P_tensor bias:           Bias for the network to train. Default: No trainable bias
            :param bool smooth_output:      If the output of the network is smoothed with an exponential kernel. Default: ``True``, use a low-pass filter on the output.
            :param float tau_mem:           Membrane potential time constant of all neurons in WaveSense. Default: 20ms
            :param float base_tau_syn:      Base synaptic time constant. Each synapse has this time constant, except the second synapse in the dilation layer which caclulates the time constant as $dilations * base_tau_syn$. Default: 20ms
            :param float tau_lp:            Time constant of the smooth output. Default: 20ms
            :param float threshold:         Threshold of all neurons in WaveSense. Default: `1.0`
            :param TorchModule neuron_model: Neuron model to use. Either :py:class:`.LIFTorch` as standard LIF implementation, :py:class:`.LIFBitshiftTorch` for hardware compatibility or :py:class:`.LIFSlayer` for speedup. Default: :py:class:`.LIFTorch`
            :param float dt:                Temporal resolution of the simulation. Default: 1ms
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
        self.lin1 = LinearTorch(shape=(n_channels_in, n_channels_res), has_bias=False)
        with torch.no_grad():
            self.lin1.weight.data = self.lin1.weight.data * dt / base_tau_syn

        self.spk1 = self.neuron_model(
            shape=(n_channels_res, n_channels_res),
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(base_tau_syn),
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
        )

        # - WaveBlock layers
        self._num_dilations = len(dilations)
        for i, dilation in enumerate(dilations):
            wave = WaveSenseBlock(
                n_channels_res,
                n_channels_skip,
                dilation=dilation,
                kernel_size=kernel_size,
                bias=bias,
                tau_mem=Constant(tau_mem),
                base_tau_syn=Constant(base_tau_syn),
                threshold=Constant(threshold),
                neuron_model=neuron_model,
                dt=dt,
            )
            self.__setattr__(f"wave{i}", wave)

        # Dense readout layers
        self.hidden = LinearTorch(shape=(n_channels_skip, n_hidden), has_bias=False)
        with torch.no_grad():
            self.hidden.weight.data = self.hidden.weight.data * dt / base_tau_syn

        self.spk2 = self.neuron_model(
            shape=(n_hidden, n_hidden),
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(base_tau_syn),
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
        )

        self.readout = LinearTorch(shape=(n_hidden, n_classes), has_bias=False)
        with torch.no_grad():
            self.readout.weight.data = self.readout.weight.data * dt / tau_lp

        self.spk_out = self.neuron_model(
            shape=(n_classes, n_classes),
            tau_mem=Constant(tau_lp),
            tau_syn=Constant(tau_lp),
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            learning_window=0.5,
            dt=dt,
        )

        # - Record dt
        self.dt = SimulationParameter(dt)
        """ float: Time-step in seconds """

        # Dictionary for recording state
        self._record = False
        self._record_dict = {}

    def forward(self, data: torch.Tensor):
        # Expected data shape
        (n_batches, t_sim, n_channels_in) = data.shape

        # - Input mapping layers
        out, _, self._record_dict["lin1"] = self.lin1(data, record=self._record)
        self._record_dict["lin1_output"] = out if self._record else []

        # Pass through spiking layer
        out, _, self._record_dict["spk1"] = self.spk1(
            out, record=self._record
        )  # (t_sim, n_batches, Nchannels)
        self._record_dict["spk1_output"] = out if self._record else []

        # Pass through each wave block in turn
        skip = 0
        for wave_index in range(self._num_dilations):
            wave_block = self.modules()[f"wave{wave_index}"]
            (out, skip_new), _, self._record_dict[f"wave{wave_index}"] = wave_block(
                out, record=self._record
            )
            self._record_dict[f"wave{wave_index}_output"] = out if self._record else []
            skip = skip_new + skip

        # Dense layers
        out, _, self._record_dict["hidden"] = self.hidden(skip, record=self._record)
        self._record_dict["hidden_output"] = out if self._record else []
        out, _, self._record_dict["spk2"] = self.spk2(out, record=self._record)
        self._record_dict["spk2_output"] = out if self._record else []

        # Final readout layer
        out, _, self._record_dict["readout"] = self.readout(out, record=self._record)
        self._record_dict["readout_output"] = out if self._record else []

        out, _, self._record_dict["spk_out"] = self.spk_out(out, record=self._record)

        # - low pass filter is not compatible with xylo unless we give tau_syn 0
        # - Smooth the output if requested
        # if self.smooth_output:
        #     out, _, self._record_dict["lp"] = self.lp(out, record=self.record)

        return out

    def evolve(self, input_data, record: bool = False):
        # - Store "record" state
        self._record = record

        # - Evolve network
        output, new_state, _ = super().evolve(input_data, record=self._record)

        # - Get recording dictionary
        record_dict = self._record_dict if self._record else {}

        # - Return
        return output, new_state, record_dict

    def as_graph(self):
        # - Convert all modules to graph representation
        mod_graphs = {k: m.as_graph() for k, m in self.modules().items()}

        # - Connect modules
        connect_modules(mod_graphs["lin1"], mod_graphs["spk1"])
        connect_modules(mod_graphs["spk1"], mod_graphs["wave0"])

        for i in range(self._num_dilations - 1):
            connect_modules(
                mod_graphs[f"wave{i}"],
                mod_graphs[f"wave{i+1}"],
                range(self.n_channels_res),
            )

            AliasConnection(
                mod_graphs[f"wave{i}"].output_nodes[self.n_channels_res :],
                mod_graphs[f"wave{i+1}"].output_nodes[self.n_channels_res :],
                name="skip_add",
                computational_module=None,
            )
        if self._num_dilations == 1:
            connect_modules(
                mod_graphs[f"wave{0}"],
                mod_graphs["hidden"],
                range(self.n_channels_res, self.n_channels_res + self.n_channels_skip),
                None,
            )
        else:
            connect_modules(
                mod_graphs[f"wave{i+1}"],
                mod_graphs["hidden"],
                range(self.n_channels_res, self.n_channels_res + self.n_channels_skip),
                None,
            )
        connect_modules(mod_graphs["hidden"], mod_graphs["spk2"])
        connect_modules(mod_graphs["spk2"], mod_graphs["readout"])
        connect_modules(mod_graphs["readout"], mod_graphs["spk_out"])

        return GraphHolder(
            mod_graphs["lin1"].input_nodes,
            mod_graphs["spk_out"].output_nodes,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
        )


import torch.nn as nn
from torch.nn.functional import pad

# Define model
class WaveBlock(nn.Module):
    def __init__(
        self, n_channels_res, n_channels_skip, kernel_size, dilation, bias=False
    ):
        super().__init__()

        self.dilation = dilation

        # Dilation layer
        self.conv1_tanh = nn.Conv1d(
            n_channels_res,
            n_channels_res,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        self.tanh1 = nn.Tanh()

        self.conv1_sig = nn.Conv1d(
            n_channels_res,
            n_channels_res,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        self.sig1 = nn.Sigmoid()

        # 1x1 projection layer
        self.conv2 = nn.Conv1d(
            n_channels_res,
            n_channels_res,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )
        self.relu2 = nn.ReLU()

        self.conv_skip = nn.Conv1d(
            n_channels_res,
            n_channels_skip,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        self.relu_skip = nn.ReLU()

    def forward(self, data):

        tanh = self.tanh1(self.conv1_tanh(pad(data, [self.dilation, 0])))
        sig = self.sig1(self.conv1_sig(pad(data, [self.dilation, 0])))
        out1 = tanh * sig
        out2 = self.conv2(out1)

        skip_out = self.conv_skip(out1)

        res_out = data + out2
        return res_out, skip_out


class WaveNet(nn.Module):
    def __init__(
        self,
        n_classes=2,
        n_channels_in=64,
        n_channels_res=16,
        n_channels_skip=32,
        n_hidden=128,
        bias=True,
        dilations=[1, 2, 4, 8, 16, 1, 2, 4, 8, 16],
        kernel_size=2,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(n_channels_in, n_channels_res, kernel_size=1, bias=bias)
        self.relu1 = nn.ReLU()

        self.wavelayers = []
        for i, d in enumerate(dilations):
            self.wavelayers.append(
                WaveBlock(
                    n_channels_res,
                    n_channels_skip,
                    kernel_size=kernel_size,
                    dilation=d,
                    bias=bias,
                )
            )
            self.add_module(f"wave{i}", self.wavelayers[-1])

        # DNN
        self.dense = nn.Conv1d(n_channels_skip, n_hidden, kernel_size=1, bias=bias)
        self.relu_dense = nn.ReLU()

        self.readout = nn.Conv1d(n_hidden, n_classes, kernel_size=1, bias=bias)

        TorchModule.from_torch(self)

    def forward(self, data):

        # move dimensions such that Torch conv layers understand them correctly
        data = data.movedim(1, 2)

        out = self.relu1(self.conv1(data))

        skip = None
        for i, layer in enumerate(self.wavelayers):
            if skip is None:
                out, skip = layer(out)
            else:
                out, skip_new = layer(out)
                skip = skip + skip_new

        # Dense readout
        out = self.relu_dense(self.dense(skip))
        out = self.readout(out)

        # revert order of data back to rockpool standard
        out = out.movedim(2, 1)

        return out
