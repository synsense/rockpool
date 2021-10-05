"""
Dynap-SE1 visualisation aid utility functions

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
01/10/2021
"""

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from rockpool.devices.dynapse.dynapse1_neuron_synapse_jax import (
    DynapSE1NeuronSynapseJax,
)

from rockpool.timeseries import TSEvent

from rockpool.devices.dynapse.utils import custom_spike_train

from typing import (
    Any,
    Dict,
    Type,
    Union,
    List,
    Optional,
    Tuple,
)

import numpy as np

ArrayLike = Union[np.ndarray, List, Tuple]
Numeric = Union[int, float, complex, np.number]
NeuronKey = Tuple[np.uint8, np.uint8, np.uint16]
NeuronConnection = Tuple[np.uint16, np.uint16]
NeuronConnectionSynType = Tuple[np.uint16, np.uint16, np.uint8]

_SAMNA_AVAILABLE = True

try:
    from samna.dynapse1 import (
        Dynapse1Configuration,
        Dynapse1Destination,
        Dynapse1Synapse,
        Dynapse1SynType,
        Dynapse1Neuron,
    )
except ModuleNotFoundError as e:
    Dynapse1Configuration = Any
    Dynapse1Destination = Any
    Dynapse1Synapse = Any
    Dynapse1SynType = Any
    Dynapse1Neuron = Any
    print(
        e,
        "\nFigure module cannot interact with samna objects",
    )
    _SAMNA_AVAILABLE = False


class Figure:
    @staticmethod
    def _decode_syn_type(
        syn_type: Union[Dynapse1SynType, str, np.uint8],
        idx_dict: Optional[Dict[str, int]] = None,
    ) -> Tuple[int, str]:
        """
        _decode_syn_type is a helper function to decode the synapse type.
        One can provide a Dynapse1SynType, str, or np.uint8, and _decode_syn_type returns the name
        and the index place of the synapse where it stored in the weight matrices

        :param syn_type: the type of the synapse either in samna format or in string format
        :type syn_type: Union[Dynapse1SynType, str, np.uint8]
        :param idx_dict: the index mapping from synapse type to index stored in the weight matrices. If one does not provide `idx_dict`, the function return -1 for all synapses. defaults to None
        :type idx_dict: Optional[Dict[str, int]], optional
        :raises TypeError: If the type of the syn_type parameter not one of the accepted types
        :return: syn_name, syn_idx
            :syn_name: name of the synapse
            :syn_idx: index of the synapse in the weight matrices
        :rtype: Tuple[int, str]
        """

        syn_idx = -1

        if isinstance(syn_type, str):
            syn_name = syn_type.upper()

        elif isinstance(syn_type, (int, np.uint8)):
            syn_name = Dynapse1SynType(syn_type).name

        elif isinstance(syn_type, Dynapse1SynType):
            syn_name = syn_type.name

        else:
            raise TypeError(
                "Please provide a proper syn_type: Dynapse1SynType, str, int"
            )

        if idx_dict is not None:
            syn_idx = idx_dict[syn_name]

        return syn_name, syn_idx

    @staticmethod
    def select_input_channels(
        input_ts: TSEvent,
        weighted_mask: ArrayLike,
        virtual: bool = True,
        idx_map: Optional[Union[Dict[int, np.uint16], Dict[int, NeuronKey]]] = None,
        title: Optional[str] = None,
    ) -> Tuple[TSEvent, List[str]]:

        """
        select_input_channels helper function to select and label channels from a TSEvent object.
        Given a weighted mask and input ts, it creates a new TSEvent object with selected channels
        and label them in accordance with given index map.

        :param input_ts: TSEvent object to be processed and clipped
        :type input_ts: TSEvent
        :param weighted_mask: A channel mask with non-binary values
        :type weighted_mask: np.ndarray
        :param virtual: Indicates if the pre-synaptic neruon is spike-generator(virtual) or real in-device neuron, defaults to True
        :type virtual: bool, optional
        :param idx_map: to map the matrix indexes of the neurons to a NeuronKey or neuron UID to be used in the label, defaults to None
        :type idx_map: Optional[Union[Dict[int, np.uint16], Dict[int, NeuronKey]]], optional
        :param title: The name of the resulting input spike train, defaults to None
        :type title: Optional[str], optional
        :raises ValueError: "Weighted mask should include as many elements as number of channels in the input_ts!"
        :return: spikes_ts, labels
            :spikes_ts: selected spike trains
            :labels: list of string labels generated for the channels in the following format : `<NeuronType>[<NeuronID>]<Repetition>`
                :NeuronType: can be 's' or 'n'. 's' means spike generator and 'n' means real in-device neuron
                :NeuronID: can be NeuronKey indicating chipID, coreID and neuronID of the neuron, can be universal neruon ID or matrix index.
                :Repetition: represents the number of synapse indicated in the weighted mask
                    n[(3, 0, 20)]x3 -> real neuron in chip 3, core 0, with neuronID 20, connection repeated 3 times
                    n[3092]x2 -> real neuron with UID 3092, connection repeated twice
                    s[0]x1 -> virtual neuron(spike generator), connection repeated once
        :rtype: Tuple[TSEvent, List[str]]
        """

        if not isinstance(weighted_mask, np.ndarray):
            weighted_mask = np.array(weighted_mask)

        if len(weighted_mask) != input_ts.num_channels:
            raise ValueError(
                "Weighted mask should include as many elements as number of channels in the input_ts!"
            )

        # Temporary storage lists
        spikes = []
        labels = []

        # Get the index map keys depending on their relative order
        if idx_map is not None:
            virtual_key, real_key = idx_map.keys()

        # Create an empty TSEvent object to append channels
        spikes_ts = custom_spike_train(
            times=np.array([]), channels=None, duration=0, name=title
        )

        # Select the channels of the TSEvent object with a weighted mask
        nonzero_idx = np.argwhere(weighted_mask).flatten()
        if nonzero_idx.size:
            input_st = input_ts.clip(channels=nonzero_idx, remap_channels=True)
            spikes.append(input_st)

            # Spike generator or in-device neuron
            n = "s" if virtual else "n"

            # Map indexes to NeuronKey or NeuronUID depending on the type of the idx_map
            if idx_map is not None:
                key = virtual_key if virtual else real_key
                name = list(map(lambda idx: f"{n}{[idx_map[key][idx]]}", nonzero_idx))

            else:
                name = list(map(lambda idx: f"{n}[{idx}]", nonzero_idx))

            count = list(map(lambda c: f"{c}", weighted_mask[nonzero_idx]))

            labels.extend(list(map(lambda t: f"{t[0]}x{t[1]}", zip(name, count))))

        # Merge spike trains in one TSEvent object
        for ts in spikes:
            spikes_ts = spikes_ts.append_c(ts)

        return spikes_ts, labels
