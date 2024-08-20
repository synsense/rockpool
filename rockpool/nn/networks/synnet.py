"""
Implements the SynNet architecture for temporal signal processing

The SynNet architecture is described in Bos & Muir 2022 [https://arxiv.org/abs/2208.12991] and Bos & Muir 2024 [https://arxiv.org/abs/2406.15112]
"""

from rockpool.nn.modules import TorchModule
from rockpool.nn.modules import LinearTorch, LIFTorch, TimeStepDropout, LIFExodus
from rockpool.parameters import Constant
from rockpool.nn.modules.torch.lif_torch import tau_to_bitshift, bitshift_to_tau
from rockpool.nn.modules.torch.lif_torch import PeriodicExponential
from rockpool.nn.combinators.sequential import TorchSequential

import torch

from copy import copy

from typing import List, Type, Union, Optional, Dict

THRESHOLD_OUT = 2**15 - 1

__all__ = ["SynNet"]


class SynNet(TorchModule):
    """
    Define a ``SynNet`` architecture network

    This class wraps a ``SynNet`` network, as defined in [1, 2].
    This is a feedforward spiking network architecture, with a range of synaptic time constants in each layer.
    By default the time constants are constant (not trainable). This is modifiable with the ``train_time_constants`` argument.

    [1] Bos & Muir 2022. "Sub-mW Neuromorphic SNN audio processing applications with Rockpool and Xylo." ESSCIRC2022. https://arxiv.org/abs/2208.12991

    [2] Bos & Muir 2024. "Micro-power spoken keyword spotting on Xylo Audio 2." arXiv. https://arxiv.org/abs/2406.15112
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        size_hidden_layers: List = [60],
        time_constants_per_layer: List = [10],
        tau_syn_base: float = 2e-3,
        tau_mem: float = 2e-3,
        tau_syn_out: float = 2e-3,
        quantize_time_constants: bool = True,
        train_time_constants: bool = False,
        threshold: float = 1.0,
        threshold_out: Union[float, List[float]] = None,
        train_threshold: bool = False,
        neuron_model: Type = LIFTorch,
        max_spikes_per_dt: int = 31,
        max_spikes_per_dt_out: int = 1,
        p_dropout: float = 0.0,
        dt: float = 1e-3,
        output: str = "spikes",
        neuron_kwargs: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        """
        Define a ``SynNet`` architecture network

        Args:
            n_classes (int):                 number of output classes
            n_channels (int):                number of input channels
            size_hidden_layers (List[int]):       list of number of neurons per layer, list has length ``number_of_layers``. Default: ``[60]``
            time_constants_per_layer (List[float]): list of number of time synaptic constants per layer, list has length ``number_of_layers``. Default: ``[10]``
            tau_syn_base (float):            smallest synaptic time constants of hidden neurons in seconds. Default: 2ms, ``2e-3``
            tau_syn_out (float):             synaptic time constants of output neurons in seconds. Default: 2ms, ``2e-3``
            tau_mem (float):                 membrane time constant of all neurons in seconds. Default: 2ms, ``2e-3``
            quantize_time_constants (bool):  If ``True``, initial time constants will be rounded to values compatibe with Xylo deployment. Default: ``True``
            train_time_constants (bool): If ``True``, time constants will be trainable. Default: ``False``, do not train time constants
            threshold (float):               threshold of hidden neurons. Default: ``1.0``
            threshold_out (Union[List[float], float]):   thresholds of readout neurons, can only be set if output is spikes. Default: ``None``, use the same value as ``threshold``
            train_threshold (bool):          If ``True``, the threshold will be trainable. If ``False``, the threshold will be constant. Default: ``False``.
            neuron_model (Type):      neuron model used for all neurons. Default: :py:class:`LIFTorch`
            max_spikes_per_dt (int):             maximum number of spikes per time step of all neurons apart from output neurons. Default: ``31``
            max_spikes_per_dt_out (int):         maximum number of spikes per time step of output neurons. Default: ``1``
            p_dropout (float):               probability that each time step from each neuron is dropped during training. Default: ``0.0``
            dt (float):                      time step for simulation in seconds. Currently the values of the time step and the time constants are not independent, thus it should be chosen carefully to allow for interpretable time constants. Default: 1ms, ``1e-3``
            output (str):                    specification of output variable, one of ``['spikes', 'vmem']``. Default: ``spikes``
            neuron_kwargs (Optional[Dict]): If supplied, keyword arguments from this dictionary will be passed to ``neuron_model`` on instantiation. Default: If :py:class:`LIFTorch` or :py:class:`LIFExodus` are used, the arguments will be ``{'spike_generation_fn':PeriodicExponential, 'max_spikes_per_dt': max_spikes_per_dt_out}``
        """

        # - Initialise superclass
        super().__init__(
            shape=(n_channels, n_classes),
            spiking_input=True,
            spiking_output=output == "spikes",
            *args,
            **kwargs,
        )

        if len(size_hidden_layers) != len(time_constants_per_layer):
            raise ValueError(
                "lists for hidden layer sizes and number of time constants per layer need to have the same length"
            )
        if tau_syn_base <= dt:
            raise ValueError(
                "the base synaptic time constant tau_syn_base needs to be larger than the time step dt"
            )

        if output not in ["spikes", "vmem"]:
            raise ValueError("output variable ", output, " not defined")
        if output == "vmem" and threshold_out is not None:
            raise ValueError(
                "threshold of readout neurons is not applied if output is vmem (membrane potential)"
            )

        # - Select the output threshold
        if output == "vmem":
            threshold_out = THRESHOLD_OUT
        else:
            threshold_out = threshold_out if threshold_out is not None else threshold

        self.output: str = output
        """ str: The output generated by this network. One of ``['spike', 'vmem']`` """

        self.dt: float = dt
        """ float: The time constant used by this network on initialisation """

        # round time constants to the values they will take when deploying to Xylo
        if quantize_time_constants:
            tau_mem_bitshift = torch.round(
                tau_to_bitshift(dt, torch.tensor(tau_mem))[0]
            ).int()
            tau_mem = bitshift_to_tau(dt, tau_mem_bitshift)[0].item()
            tau_syn_out_bitshift = torch.round(
                tau_to_bitshift(dt, torch.tensor(tau_syn_out))[0]
            ).int()
            tau_syn_out = bitshift_to_tau(dt, tau_syn_out_bitshift)[0].item()

        # - Make ``tau_mem`` constant, if it should not be trainable
        tau_mem = tau_mem if train_time_constants else Constant(tau_mem)

        # calculate how often time constants are repeated within a layer
        tau_repetitions = []
        for i, (n_hidden, n_tau) in enumerate(
            zip(size_hidden_layers, time_constants_per_layer)
        ):
            tau_repetitions.append(int(n_hidden / n_tau) + min(1, n_hidden % n_tau))

        # - Define an empty Sequential network, to add each layer to
        self.seq = TorchSequential()
        """ Sequential: The network itself, as a ``Sequential`` Module """

        # - Generate neuron arguments
        if neuron_model in [LIFTorch, LIFExodus]:
            default_neuron_kwargs = {
                "spike_generation_fn": PeriodicExponential,
                "max_spikes_per_dt": max_spikes_per_dt,
            }

            if neuron_kwargs is not None:
                default_neuron_kwargs.update(neuron_kwargs)

            default_out_neuron_kwargs = copy(default_neuron_kwargs)
            default_out_neuron_kwargs.update(
                {
                    "max_spikes_per_dt": max_spikes_per_dt_out,
                }
            )

        # - Generate each set of weights and neurons in turn
        n_channels_in = n_channels
        lif_names = []
        for i, (n_hidden, n_tau) in enumerate(
            zip(size_hidden_layers, time_constants_per_layer)
        ):
            # - Generate time constants
            taus = [
                torch.tensor(
                    [(tau_syn_base / dt) ** j * dt for j in range(1, n_tau + 1)]
                )
                for _ in range(tau_repetitions[i])
            ]
            tau_syn_hidden = torch.hstack(taus)

            # if size of layer is not a multiple of the time constants connections of different time constants are
            # removed starting from the largest one
            tau_syn_hidden = tau_syn_hidden[:n_hidden]

            # round time constants to the values they will take when deploying to Xylo
            if quantize_time_constants:
                tau_syn_hidden_bitshift = [
                    torch.round(tau_to_bitshift(dt, tau_syn)[0]).int()
                    for tau_syn in tau_syn_hidden
                ]
                tau_syn_hidden = torch.tensor(
                    [
                        bitshift_to_tau(dt, dash_syn)[0].item()
                        for dash_syn in tau_syn_hidden_bitshift
                    ]
                )

            # - Generate a linear weight module
            lyr_weights = LinearTorch(shape=(n_channels_in, n_hidden), has_bias=False)
            n_channels_in = n_hidden

            # - Normalise weights by time constant and add to network
            with torch.no_grad():
                lyr_weights.weight.data = lyr_weights.weight.data * dt / tau_syn_hidden
            self.seq.append(lyr_weights, f"{i}_linear")

            # - Add the neuron layer to the network
            self.seq.append(
                neuron_model(
                    shape=(n_hidden, n_hidden),
                    tau_mem=tau_mem,
                    tau_syn=(
                        tau_syn_hidden
                        if train_time_constants
                        else Constant(tau_syn_hidden)
                    ),
                    bias=Constant(0.0),
                    threshold=threshold if train_threshold else Constant(threshold),
                    dt=dt,
                    **default_neuron_kwargs,
                ),
                f"{i}_neurons",
            )
            lif_names.append(f"{i}_neurons")

            # - Incorporate a dropout layer, if requested
            if p_dropout > 0.0:
                self.seq.append(
                    TimeStepDropout(shape=(n_hidden), p=p_dropout), f"{i}_dropout"
                )

        # - Add the output weight layer
        lyr_weights = LinearTorch(shape=(n_hidden, n_classes), has_bias=False)
        with torch.no_grad():
            lyr_weights.weight.data = lyr_weights.weight.data * dt / tau_syn_out

        self.seq.append(lyr_weights, "out_linear")

        # - Add the output neuron layer
        self.seq.append(
            neuron_model(
                shape=(n_classes, n_classes),
                tau_mem=Constant(tau_mem),
                tau_syn=Constant(tau_syn_out),
                bias=Constant(0.0),
                threshold=Constant(threshold_out),
                dt=dt,
                **default_out_neuron_kwargs,
            ),
            "out_neurons",
        )
        lif_names.append(f"out_neurons")

        # Record names of neuron and output layers
        self.lif_names: List[str] = lif_names
        """ List[str]: A list of the neuron models present in this network, in evolution order. """

        self.label_last_LIF: str = lif_names[-1]
        """ str: The name of the readout neuron layer in this network. """

        # Dictionary for recording state
        self._record: bool = False
        """ bool: If ``True``, record the state trace during evolution """

        self._record_dict: dict = {}
        """ dict: The internal set of recorded state traces, if requested """

    def forward(self, data: torch.Tensor):
        # - Evolve the Sequential network
        out, _, record_dict = self.seq(data, record=self._record)

        # - If "vmem" output is requested, use this instead of spiking in the final layer
        if self.output == "vmem":
            out = record_dict[self.label_last_LIF]["vmem"]

        # - Modify the record dictionary to store the output
        if self._record:
            for key in record_dict.keys():
                if "output" in key:
                    record_dict[key] = out

        # - Keep a copy of the record dictionary
        self._record_dict = record_dict if self._record else {}

        # - Return the model output
        return out

    def evolve(self, input_data, record: bool = False):
        # - Store "record" state
        self._record = record or self.output == "vmem"

        # - Evolve network
        output, new_state, record_dict = super().evolve(input_data, record=self._record)

        # - Get recording dictionary
        record_dict = record_dict if record else {}

        # - Return
        return output, new_state, record_dict

    def as_graph(self):
        # - Return the graph from the ``Sequential`` network
        return self.seq.as_graph()
