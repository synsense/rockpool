"""
Implements the WaveSense architecture from Weidel et al 2021 [1]
[1] 
"""

from rockpool.nn.modules import TorchModule, LinearTorch, LIFTorch, ExpSynTorch
from rockpool.parameters import Parameter, State, SimulationParameter

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
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """

        Args:
            Nchannels:
            Nskip:
            dilation:
            kernel_size:
            has_bias:
            tau_mem:
            base_tau_syn:
            threshold:
            dt:
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

        # - Dilation layers
        tau_syn = (
            torch.arange(0, dilation * kernel_size, dilation) * base_tau_syn
        ).float()
        tau_syn = torch.clamp(tau_syn, base_tau_syn, tau_syn.max())
        tau_syn = torch.cat(tuple([tau_syn] * Nchannels))

        self.lin1 = LinearTorch((Nchannels, Nchannels * kernel_size), has_bias=has_bias)
        with torch.no_grad():
            # normalize for tau_syn
            self.lin1.weight = self.lin1.weight / (tau_syn * 1000)
        self.lin1.weight.requires_grad = True

        if has_bias:
            self.lin1.bias.requires_grad = True

        self.spk1 = LIFTorch(
            (Nchannels * kernel_size, Nchannels),
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            has_bias=has_bias,
            threshold=threshold,
            dt=dt,
        )

        # - Build a synapse-to-neuron mapping matrix
        self._w_syn_mapping = torch.tensor(
            [
                [0] * ind * kernel_size
                + [1] * kernel_size
                + [0] * (Nchannels - ind - 1) * kernel_size
                for ind in range(Nchannels)
            ]
        ).T

        # - Remapping output layers
        self.lin2_res = LinearTorch((Nchannels, Nchannels), has_bias=has_bias)
        with torch.no_grad():
            # normalize for tau_syn
            self.lin2_res.weight = self.lin2_res.weight / (tau_syn.min().item() * 1000)
        self.lin2_res.weight.requires_grad = True


        self.lin2_res.weight.requires_grad = True
        if has_bias:
            self.lin2_res.bias.requires_grad = True

        self.spk2_res = LIFTorch(
            (Nchannels,),
            tau_mem=tau_mem,
            tau_syn=tau_syn.min().item(),
            has_bias=has_bias,
            threshold=threshold,
            dt=dt,
        )

        # - Skip output layers
        self.lin2_skip = LinearTorch((Nchannels, Nskip), has_bias=has_bias)
        with torch.no_grad():
            # normalize for tau_syn
            self.lin2_skip.weight = self.lin2_skip.weight / (tau_syn.min().item() * 1000)
        self.lin2_skip.weight.requires_grad = True
        if has_bias:
            self.lin2_skip.bias.requires_grad = True

        self.spk2_skip = LIFTorch(
            (Nskip,),
            tau_mem=tau_mem,
            tau_syn=tau_syn.min().item(),
            has_bias=has_bias,
            threshold=threshold,
            dt=dt,
        )

        # - Internal record dictionary
        self._record_dict = {}

    def forward(self, data: torch.tensor) -> (torch.tensor, dict, dict):
        # Expecting data to be of the format (batch, time, n_neurons)
        (n_batches, t_sim, n_neurons) = data.shape

        # - Pass through dilated weight layer
        out, _, self._record_dict["lin1"] = self.lin1(data, record=True)
        self._record_dict["lin1_output"] = out

        # - Enforce the synapse-to-neuron mapping matrix for the dilation spiking layer
        self.spk1.w_syn = self._w_syn_mapping

        # - Pass through dilated spiking layer
        hidden, _, self._record_dict["spk1"] = self.spk1(
            out, record=True
        )  # (t_sim, n_batches, n_neurons)
        self._record_dict["spk1_output"] = hidden

        # - Pass through output linear weights
        out_res, _, self._record_dict["lin2_res"] = self.lin2_res(hidden, record=True)
        self._record_dict["lin2_res_output"] = out_res

        # - Pass through output spiking layer
        out_res, _, self._record_dict["spk2_res"] = self.spk2_res(out_res, record=True)
        self._record_dict["spk2_res_output"] = out_res

        # - Hidden -> skip outputs
        out_skip, _, self._record_dict["lin2_skip"] = self.lin2_skip(
            hidden, record=True
        )
        self._record_dict["lin2_skip_output"] = out_skip

        # - Pass through skip output spiking layer
        out_skip, _, self._record_dict["spk2_skip"] = self.spk2_skip(
            out_skip, record=True
        )
        self._record_dict["spk2_skip_output"] = out_skip

        # - Combine output and residual connections (pass-through)
        res_out = out_res + data

        return res_out, out_skip

    def evolve(self, input, record: bool = False):
        # - Use super-class evolve
        output, new_state, _ = super().evolve(input, record)

        # - Get state record from property
        record_dict = self._record_dict if record else {}

        return output, new_state, record_dict


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
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """

        Args:
            dilations:
            n_classes:
            n_channels_in:
            n_channels_res:
            n_channels_skip:
            n_hidden:
            kernel_size:
            has_bias:
            smooth_output:
            tau_mem:
            base_tau_syn:
            tau_lp:
            threshold:
            dt:
            *args:
            **kwargs:
        """
        # - Determine network shape and initialise
        shape = (n_channels_in, n_classes)
        super().__init__(
            shape=shape, spiking_input=True, spiking_output=True, *args, **kwargs
        )

        # - Input mapping layers
        self.lin1 = LinearTorch((n_channels_in, n_channels_res), has_bias=has_bias)
        with torch.no_grad():
            # normalize for tau_syn
            self.lin1.weight = self.lin1.weight / (base_tau_syn * 1000)
        self.lin1.weight.requires_grad = True
        if has_bias:
            self.lin1.bias.requires_grad = True

        self.spk1 = LIFTorch(
            n_channels_res,
            tau_mem=tau_mem,
            tau_syn=base_tau_syn,
            has_bias=has_bias,
            threshold=threshold,
            dt=dt,
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
            )
            self.__setattr__(f"wave{i}", wave)

        # Dense readout layers
        self.dense = LinearTorch((n_channels_skip, n_hidden), has_bias=has_bias)
        with torch.no_grad():
            # normalize for tau_syn
            self.dense.weight = self.dense.weight / (base_tau_syn * 1000)
        self.dense.weight.requires_grad = True

        if has_bias:
            self.dense.bias.requires_grad = True

        self.spk2 = LIFTorch(
            n_hidden,
            tau_mem=tau_mem,
            tau_syn=base_tau_syn,
            has_bias=has_bias,
            threshold=threshold,
            dt=dt,
        )
        self.readout = LinearTorch((n_hidden, n_classes), has_bias=has_bias)
        self.readout.weight.requires_grad = True
        if has_bias:
            self.readout.bias.requires_grad = True

        # Smoothing output
        self.smooth_output = SimulationParameter(smooth_output)
        """ bool: Perform low-pass filtering of the readout """

        if smooth_output:
            with torch.no_grad():
                # normalize for tau_syn
                self.readout.weight = self.readout.weight / (tau_lp  * 1000)
            self.lp = ExpSynTorch(n_classes, tau_syn=tau_lp, dt=dt)

        # - Record dt
        self.dt = SimulationParameter(dt)
        """ float: Time-step in seconds """

        # Dictionary for recording state
        self._record_dict = {}

    def forward(self, data: torch.Tensor):
        # Expected data shape
        (n_batches, t_sim, n_channels_in) = data.shape

        # - Input mapping layers
        out, _, self._record_dict["lin1"] = self.lin1(data, record=True)
        self._record_dict["lin1_output"] = out.detach()

        # Pass through spiking layer
        out, _, self._record_dict["spk1"] = self.spk1(
            out, record=True
        )  # (t_sim, n_batches, n_neurons)
        self._record_dict["spk1_output"] = out.detach()

        # Pass through each wave block in turn
        skip = 0
        for wave_index in range(self._num_dilations):
            wave_block = self.modules()[f"wave{wave_index}"]
            (out, skip_new), _, self._record_dict[f"wave{wave_index}"] = wave_block(
                out, record=True
            )
            skip = skip_new + skip

        # Dense layers
        out, _, self._record_dict["dense"] = self.dense(skip, record=True)
        self._record_dict["dense_output"] = out.detach()
        out, _, self._record_dict["spk2"] = self.spk2(out, record=True)
        self._record_dict["spk2_output"] = out.detach()

        # Final readout layer
        out, _, self._record_dict["readout"] = self.readout(out, record=True)
        self._record_dict["readout_output"] = out.detach()

        # Smooth the output if requested
        if self.smooth_output:
            out, _, self._record_dict["lp"] = self.lp(out, record=True)
            self._record_dict["lp_output"] = out.detach()

        return out

    def evolve(self, input_data, record: bool = False):
        output, new_state, _ = super().evolve(input_data, record=record)

        record_dict = self._record_dict if record else {}
        return output, new_state, record_dict

    def trainable_parameters(self):
        return [p for p in list(self.parameters().astorch()) if p.requires_grad]
