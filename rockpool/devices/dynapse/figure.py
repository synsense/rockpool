"""
Dynap-SE1 visualisation aid utility functions

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
01/10/2021
"""

import matplotlib
import matplotlib.pyplot as plt
from rockpool.devices.dynapse.dynapse1_neuron_synapse_jax import (
    DynapSE1NeuronSynapseJax,
)

from rockpool.timeseries import TSEvent, TSContinuous

from rockpool.devices.dynapse.utils import custom_spike_train

from typing import (
    Any,
    Dict,
    Union,
    List,
    Optional,
    Tuple,
)

from rockpool.typehints import (
    FloatVector,
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
        :raises ValueError: Weighted mask should include as many elements as number of channels in the input_ts!
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

    @staticmethod
    def spike_input_post(
        mod: DynapSE1NeuronSynapseJax,
        input_ts: TSEvent,
        output_ts: TSEvent,
        post: int,
        syn_type: Union[Dynapse1SynType, str, np.uint8],
        virtual: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Tuple[TSEvent, List[str]]:
        """
        spike_input_post gather together all the input spikes of a pre-synaptic neuron

        :param mod: The module to investigate
        :type mod: DynapSE1NeuronSynapseJax
        :param input_ts: Input spike trains fed to DynapSE1NeuronSynapseJax object
        :type input_ts: TSEvent
        :param output_ts: Output spike trains of DynapSE1NeuronSynapseJax object
        :type output_ts: TSEvent
        :param post: matrix index of the post synaptic neuron defined inside the `mod`
        :type post: int
        :param syn_type: the listening synapse of post-synaptic neuron of interest (e.g. "AMPA", "GABA_A", ...)
        :type syn_type: Union[Dynapse1SynType, str, np.uint8]
        :param virtual: Gather only the virtual connections or not. Gather both if None, defaults to None
        :type virtual: Optional[bool], optional
        :return: spikes_ts, labels
            :spikes_ts: input spike trains to post-synaptic neuron
            :labels: list of string labels generated for the channels in the following format : `<NeuronType>[<NeuronID>]<Repetition>`
                :NeuronType: can be 's' or 'n'. 's' means spike generator and 'n' means real in-device neuron
                :NeuronID: can be NeuronKey indicating chipID, coreID and neuronID of the neuron, can be universal neruon ID or matrix index.
                :Repetition: represents the number of synapse indicated in the weighted mask
                    n[(3, 0, 20)]x3 -> real neuron in chip 3, core 0, with neuronID 20, connection repeated 3 times (idx_map provided)
                    n[3092]x2 -> real neuron with UID 3092, connection repeated twice (idx_map provided)
                    s[0]x1 -> virtual neuron(spike generator), connection repeated once
        :rtype: Tuple[TSEvent, List[str]]
        """

        def gather_st(
            _ts: TSEvent, weight: FloatVector, virtual: bool
        ) -> Tuple[TSEvent, List[str]]:
            """
            gather_st extracts input spike trains to post-synaptic neuron from the given
            ts_event object depending on the weight matrix provided.
            In the weight matrix the organization is as follows : w[pre][post][syn_type].

            :param _ts: Input or output `TSEvent` object fed to or produced by the simulation module
            :type _ts: TSEvent
            :param weight: the weight matrix encoding the connectivity between the neruons w[pre][post][syn_type]
            :type weight: FloatVector
            :param virtual: Label the neurons as virtual or not.
            :type virtual: bool
            :return: spikes_ts, labels (as the same as the encapsulating function)
            :rtype: Tuple[TSEvent, List[str]]
            """

            mask = weight[:, post, syn_idx]
            spikes_ts, labels = Figure.select_input_channels(
                _ts, mask, virtual, *args, **kwargs
            )
            return spikes_ts, labels

        if syn_type is not None:
            _, syn_idx = Figure._decode_syn_type(
                syn_type, mod.SYN.default_idx_dict_no_ahp
            )

        # Gather external AND recurrent input spike trains
        if virtual is None:
            external_ts, external_labels = gather_st(input_ts, mod.w_in, True)
            recurrent_ts, recurrent_labels = gather_st(output_ts, mod.w_rec, False)

            # Merge external and recurrent inputs
            spikes_ts = external_ts.append_c(recurrent_ts)
            labels = external_labels + recurrent_labels

        # Gather external OR recurrent input spike trains
        else:
            _ts = input_ts if virtual else output_ts
            _weight = mod.w_in if virtual else mod.w_rec
            spikes_ts, labels = gather_st(_ts, _weight, virtual)

        return spikes_ts, labels

    @staticmethod
    def plot_spikes_label(
        spikes_ts: TSEvent,
        labels: List[str],
        ax: Optional[matplotlib.axes.Axes] = None,
        cmap: Optional[str] = "rainbow",
        *args,
        **kwargs,
    ) -> matplotlib.collections.PathCollection:
        """
        plot_spikes_label helper function used for plotting the spike train with labeled channels

        :param spikes_ts: spike train to be plotted
        :type spikes_ts: TSEvent
        :param labels: Channel labels
        :type labels: List[str]
        :param ax: The sub-plot axis to plot the figure, defaults to None
        :type ax: Optional[matplotlib.axes.Axes], optional
        :param cmap: matplotlib color map. For full list, please check https://matplotlib.org/stable/tutorials/colors/colormaps.html, defaults to "rainbow"
        :type cmap: Optional[str], optional
        :raises ValueError: `labels` should include as many elements as number of channels in the `spikes_ts`
        :return: `PathCollection` object returned by scatter plot
        :rtype: matplotlib.collections.PathCollection
        """
        if len(labels) > spikes_ts.num_channels:
            raise ValueError(
                "`labels` should include as many elements as number of channels in the `spikes_ts`"
            )

        if ax is not None:
            plt.sca(ax)

        # Empty figure if no incoming spikes
        if spikes_ts:
            scatter = spikes_ts.plot(c=spikes_ts.channels, cmap=cmap, *args, **kwargs)
            plt.yticks(range(len(labels)), labels)

        else:
            scatter = spikes_ts.plot()

        return scatter

    @staticmethod
    def plot_Isyn(
        Isyn_record: np.ndarray,
        dt: float = 1e-3,
        name: Optional[str] = "$I_{syn}$",
        ax: Optional[matplotlib.axes.Axes] = None,
        margin: Optional[float] = 0.2,
        *args,
        **kwargs,
    ) -> TSContinuous:
        """
        plot_Isyn converts an `Isyn_record` obtained from the record dictionary to a TSContinuous object and plot

        :param Isyn_record: Synaptic currents of the neurons recorded with respect to time (T,N)
        :type Isyn_record: np.ndarray
        :param dt: The discrete time resolution of recording, defaults to 1e-3
        :type dt: float, optional
        :param name: title of the figure, name of the `TSContinuous` object, defaults to "$I_{syn}$"
        :type name: Optional[str], optional
        :param ax: The sub-plot axis to plot the figure, defaults to None
        :type ax: Optional[matplotlib.axes.Axes], optional
        :param margin: The margin(ratio) between the edges of the figure and edges of the lines, defaults to 0.2
        :type margin: Optional[float], optional
        :return: Isyn current in `TSContinuous` object format
        :rtype: TSContinuous
        """
        f_margin = 1.0 + margin if margin is not None else 1.0

        if ax is not None:
            plt.sca(ax)

        Isyn = TSContinuous.from_clocked(Isyn_record, dt=dt, name=name)

        # Plotting
        Isyn.plot(stagger=Isyn.max * f_margin, *args, **kwargs)
        plt.ylabel("Current(A)")

        return Isyn

    def plot_Imem(
        Imem_record: np.ndarray,
        Ispkthr: Optional[Union[float, np.ndarray]] = None,
        dt: float = 1e-3,
        name: str = "$I_{mem}$",
        margin: Optional[float] = 0.2,
        ax: Optional[matplotlib.axes.Axes] = None,
        line_ratio: float = 0.3,
        *args,
        **kwargs,
    ) -> Tuple[TSContinuous, TSContinuous]:
        """
        plot_Imem converts an `Imem_record` obtained from the record dictionary to a TSContinuous object and plot

        :param Imem_record: Membrane currents of the neurons recorded with respect to time (T,N)
        :type Imem_record: np.ndarray
        :param Ispkthr: Spike threshold for neurons. Both a single float number for global spike threshold and an array of numbers for neuron-specific thresholds can be provided. Plotted with dashed lines if provided, defaults to None
        :type Ispkthr: Optional[float], optional
        :param dt: The discrete time resolution of the recording, defaults to 1e-3
        :type dt: float, optional
        :param name: title of the figure, name of the `TSContinuous` object, defaults to "$I_{mem}$"
        :type name: str, optional
        :param margin: The margin between the edges of the figure and edges of the lines, defaults to 0.2
        :type margin: Optional[float], optional
        :param ax: The sub-plot axis to plot the figure, defaults to None
        :type ax: Optional[matplotlib.axes.Axes], optional
        :param line_ratio: the ratio between Imem lines and the Ispkthr lines, defaults to 0.3
        :type line_ratio: float, optional
        :return: Imem, Ispkthr
            :Imem: Imem current in `TSContinuous` object format
            :Ispkthr: Ispkthr threshold current in `TSContinuous` object format
        :rtype: Tuple[TSContinuous, TSContinuous]
        """
        f_margin = 1.0 + margin if margin is not None else 1.0

        if ax is not None:
            plt.sca(ax)

        Imem = TSContinuous.from_clocked(Imem_record, dt=dt, name=name)

        # Plotting
        _lines = Imem.plot(stagger=Imem.max * f_margin, *args, **kwargs)
        plt.ylabel("Current(A)")

        # Spike threshold lines
        if Ispkthr is not None:
            linewidth = _lines[0]._linewidth * line_ratio
            Ispkthr = np.ones_like(Imem_record) * Ispkthr
            Ispkthr = TSContinuous.from_clocked(Ispkthr, dt=dt, name="$I_{spkthr}$")
            Ispkthr.plot(
                stagger=Imem.max * f_margin, linestyle="dashed", linewidth=linewidth
            )

        return Imem, Ispkthr
