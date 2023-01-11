"""
Implements the SynNet architecture for temporal signal processing

The SynNet architecture is described in Bos et al 2022 [https://arxiv.org/abs/2208.12991]

"""

from rockpool.nn.modules import TorchModule, Module
from rockpool.nn.modules import LinearTorch, LIFTorch, TimeStepDropout
from rockpool.parameters import Constant
from rockpool.nn.modules.torch.lif_torch import tau_to_bitshift, bitshift_to_tau
from rockpool.nn.modules.torch.lif_torch import PeriodicExponential
from rockpool.graph import GraphHolder, connect_modules

import torch

from typing import List

__all__ = ["SynNet"]


class SynNet(TorchModule):
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        size_hidden_layers: List = [60],
        time_constants_per_layer: List = [10],
        tau_syn_base: float = 0.002,
        tau_mem: float = 0.002,
        tau_syn_out: float = 0.002,
        quantize_time_constants: bool = True,
        threshold: float = 1.0,
        threshold_out: float = 100.0,
        train_threshold: bool = False,
        neuron_model: TorchModule = LIFTorch,
        max_spikes_per_dt: int = 31,
        max_spikes_per_dt_out: int = 1,
        p_dropout: float = 0.0,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Args:
            :param int n_classes:                 number of output classes
            :param int n_channels:                number of input channels
            :param List size_hidden_layers:       list of number of neurons per layer, list has length
                                                  number_of_layers
            :param List time_constants_per_layer: list of number of time synaptic constants per layer, list has length
                                                  number_of_layers, the unit of each time constant is seconds
            :param float tau_syn_base:            smallest synaptic time constants of hidden neurons in seconds
            :param float tau_syn_out:             synaptic time constants of output neurons in seconds
            :param float tau_mem:                 membrane time constant of all neurons in seconds
            :param bool quantize_time_constants:  determines if time constants are rounded to deployment values
            :param float threshold:               threshold of hidden neurons
            :param float threshold_out:               threshold of readout neurons
            :param bool train_threshold:          determines of threshold is trained
            :param TorchModule neuron_model:      neuron model of all neurons
            :param max_spikes_per_dt:             maximum number of spikes per time step of all neurons apart from
                                                  output neurons
            :param max_spikes_per_dt_out:         maximum number of spikes per time step of output neurons
            :param float p_dropout:               probability that one time step from one neuorn is dropped
            :param float dt:                      time step for simulation in seconds, determines time step on hardware,
                                                  currently the values of the time step and the time constants are not
                                                  independent, thus it should be chosen carefully to allow for
                                                  interpretable time constants
        """

        super().__init__(
            shape=(n_channels, n_classes),
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        if len(size_hidden_layers) != len(time_constants_per_layer):
            raise ValueError(
                "lists for hidden layer sizes and number of time constants per layer need to have the same length"
            )
        if tau_syn_base <= dt:
            raise ValueError("the base synaptic time constant tau_syn_base needs to be larger than the time step dt")

        # round time constants to the values they will take when deploying to Xylo
        if quantize_time_constants:
            tau_mem_bitshift = torch.round(tau_to_bitshift(dt, torch.tensor(tau_mem))[0]).int()
            tau_mem = bitshift_to_tau(dt, tau_mem_bitshift)[0].item()
            tau_syn_out_bitshift = torch.round(tau_to_bitshift(dt, torch.tensor(tau_syn_out))[0]).int()
            tau_syn_out = bitshift_to_tau(dt, tau_syn_out_bitshift)[0].item()

        # calculate how often time constants are repeated within a layer
        tau_repititions = []
        for i, (n_hidden, n_tau) in enumerate(
            zip(size_hidden_layers, time_constants_per_layer)
        ):
            tau_repititions.append(int(n_hidden / n_tau) + min(1, n_hidden % n_tau))

        self.lins = []
        self.spks = []
        self.dropouts = []
        n_channels_in = n_channels
        for i, (n_hidden, n_tau) in enumerate(
            zip(size_hidden_layers, time_constants_per_layer)
        ):

            taus = [
                torch.tensor([(tau_syn_base/dt)**j*dt for j in range(1, n_tau + 1)])
                for _ in range(tau_repititions[i])
            ]
            tau_syn_hidden = torch.hstack(taus)
            # if size of layer is not a multiple of the time constants connections of different time constants are
            # removed starting from the largest one
            tau_syn_hidden = tau_syn_hidden[:n_hidden]
            # orund time constants to the values they will take when deploying to Xylo
            if quantize_time_constants:
                tau_syn_hidden_bitshift = [torch.round(tau_to_bitshift(dt, tau_syn)[0]).int() for tau_syn in tau_syn_hidden]
                tau_syn_hidden = torch.tensor(
                    [bitshift_to_tau(dt, dash_syn)[0].item() for dash_syn in tau_syn_hidden_bitshift])

            if not train_threshold:
                thresholds = Constant([threshold for _ in range(n_hidden)])
            else:
                thresholds = [threshold for _ in range(n_hidden)]

            self.lins.append(
                LinearTorch(shape=(n_channels_in, n_hidden), has_bias=False)
            )
            n_channels_in = n_hidden
            with torch.no_grad():
                self.lins[-1].weight.data = (
                    self.lins[-1].weight.data / tau_syn_hidden
                )
            setattr(self, "lin" + str(i), self.lins[-1])
            self.spks.append(
                neuron_model(
                    shape=(n_hidden, n_hidden),
                    tau_mem=Constant(tau_mem),
                    tau_syn=Constant(tau_syn_hidden),
                    bias=Constant(0.0),
                    threshold=thresholds,
                    spike_generation_fn=PeriodicExponential,
                    dt=dt,
                    max_spikes_per_dt=max_spikes_per_dt,
                )
            )
            setattr(self, "spk" + str(i), self.spks[-1])

            self.dropouts.append(TimeStepDropout(shape=(n_hidden), p=p_dropout))

        self.lin_out = LinearTorch(shape=(n_hidden, n_classes), has_bias=False)
        with torch.no_grad():
            self.lin_out.weight.data = self.lin_out.weight.data / tau_syn_out
        setattr(self, "lin_out", self.lin_out)

        self.spk_out = neuron_model(
            shape=(n_classes, n_classes),
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(tau_syn_out),
            bias=Constant(0.0),
            threshold=Constant([threshold_out for _ in range(n_classes)]),
            spike_generation_fn=PeriodicExponential,
            max_spikes_per_dt=max_spikes_per_dt_out,
            dt=dt,
        )
        setattr(self, "spk_out", self.spk_out)

        # Dictionary for recording state
        self._record = False
        self._record_dict = {}

    def forward(self, data: torch.Tensor):

        out = data
        for i, (lin, spk, dropout) in enumerate(
            zip(self.lins, self.spks, self.dropouts)
        ):
            out, _, self._record_dict["lin" + str(i)] = lin(out, record=self._record)
            out, _, self._record_dict["spk" + str(i)] = spk(out, record=self._record)
            out, _, _ = dropout(out)

        # readout layer
        out, _, self._record_dict["lin_out"] = self.lin_out(out, record=self._record)
        out, _, self._record_dict["spk_out"] = self.spk_out(out, record=self._record)

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
        # convert all modules to graph representation
        mod_graphs = {k: m.as_graph() for k, m in Module.modules(self).items()}

        # connect modules
        for i, (lin, spk) in enumerate(zip(self.lins, self.spks)):
            connect_modules(mod_graphs["lin" + str(i)], mod_graphs["spk" + str(i)])
            if i == len(self.lins) - 1:
                connect_modules(mod_graphs["spk" + str(i)], mod_graphs["lin_out"])
            else:
                connect_modules(
                    mod_graphs["spk" + str(i)], mod_graphs["lin" + str(i + 1)]
                )
        connect_modules(mod_graphs["lin_out"], mod_graphs["spk_out"])

        return GraphHolder(
            mod_graphs["lin0"].input_nodes,
            mod_graphs["spk_out"].output_nodes,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
        )
