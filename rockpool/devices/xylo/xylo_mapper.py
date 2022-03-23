"""
Mapper package for Xylo

- Create a graph using the :py:meth:`~.graph.GraphModule.as_graph` API
- Call :py:func:`.mapper`

"""


import numpy as np

import copy

from rockpool.graph import (
    GraphModuleBase,
    GenericNeurons,
    AliasConnection,
    LinearWeights,
    SetList,
    bag_graph,
    find_modules_of_subclass,
    find_recurrent_modules,
)
from rockpool.devices.xylo import XyloHiddenNeurons, XyloOutputNeurons, XyloNeurons


from typing import List, Callable, Set, Optional, Union

__all__ = ["mapper", "DRCError"]


class DRCError(ValueError):
    pass


def output_nodes_have_neurons_as_source(graph: GraphModuleBase) -> None:
    # - All output nodes must have a source that is a neuron
    for n in graph.output_nodes:
        for s in n.source_modules:
            if not isinstance(s, GenericNeurons):
                raise DRCError(
                    f"All network outputs must be directly from neurons.\nA network output node {n} has a source {s} which is not a neuron."
                )


def input_to_neurons_is_a_weight(graph: GraphModuleBase) -> None:
    # - Every neuron module must have weights on the input
    neurons = find_modules_of_subclass(graph, GenericNeurons)

    for n in neurons:
        for inp in n.input_nodes:
            for sm in inp.source_modules:
                if not isinstance(sm, LinearWeights):
                    raise DRCError(
                        f"All neurons must receive inputs only from weight nodes.\nA neuron node {n} has a source module {sm} which is not a LinearWeight."
                    )


def first_module_is_a_weight(graph: GraphModuleBase) -> None:
    # - The first module after the input must be a set of weights
    for inp in graph.input_nodes:
        for sink in inp.sink_modules:
            if not isinstance(sink, LinearWeights):
                raise DRCError(
                    f"The network input must go first through a weight.\nA network input node {inp} has a sink module {sink} which is not a LinearWeight."
                )


def le_16_input_channels(graph: GraphModuleBase) -> None:
    if len(graph.input_nodes) > 16:
        raise DRCError(
            f"Xylo only supports up to 16 input channels. The network requires {len(graph.input_nodes)} input channels."
        )


def le_8_output_channels(graph: GraphModuleBase) -> None:
    if len(graph.output_nodes) > 8:
        raise DRCError(
            f"Xylo only supports up to 8 output channels. The network requires {len(graph.output_nodes)} output channels."
        )


def all_neurons_have_same_dt(graph: GraphModuleBase) -> None:
    neurons: SetList[GenericNeurons] = find_modules_of_subclass(graph, GenericNeurons)

    dt: Optional[float] = None
    for n in neurons:
        if hasattr(n, "dt"):
            dt = n.dt if dt is None else dt
            if dt is not None and n.dt is not None and not np.isclose(dt, n.dt):
                raise DRCError("All neurons in the network must share a common `dt`.")

    if dt is None:
        raise DRCError(
            "The network must specify a `dt` for at least one neuron module."
        )


def output_neurons_cannot_be_recurrent(graph: GraphModuleBase) -> None:
    recurrent_modules = find_recurrent_modules(graph)

    output_neurons = SetList()
    for n in graph.output_nodes:
        for s in n.source_modules:
            if isinstance(s, GenericNeurons):
                output_neurons.add(s)

    rec_output_neurons = set(output_neurons).intersection(recurrent_modules)
    if len(rec_output_neurons) > 0:
        raise DRCError(
            f"Output neurons may not be recurrent.\nFound output neurons {rec_output_neurons} that are recurrent."
        )


def no_consecutive_weights(graph: GraphModuleBase) -> None:
    all_weights = find_modules_of_subclass(graph, LinearWeights)

    for w in all_weights:
        for i_n in w.input_nodes:
            for sm in i_n.source_modules:
                if isinstance(sm, LinearWeights):
                    raise DRCError(
                        f"Inputs to linear weights may not be linear weights.\nFound linear weights {sm} as source module -> to linear weights {w}."
                    )

        for o_n in w.output_nodes:
            for sm in o_n.sink_modules:
                if isinstance(sm, LinearWeights):
                    raise DRCError(
                        f"Outputs of linear weights may not be linear weights.\nFound linear weights {w} with output sink module -> {sm}."
                    )


def alias_inputs_must_be_neurons(graph: GraphModuleBase) -> None:
    all_aliases = find_modules_of_subclass(graph, AliasConnection)

    for a in all_aliases:
        for i_n in a.input_nodes:
            for source in i_n.source_modules:
                if not isinstance(source, (GenericNeurons, AliasConnection)):
                    raise DRCError(
                        f"Inputs to alias connections must be neurons or another alias.\nFound source module {source} as source -> to aliases {a}."
                    )


def alias_output_nodes_must_have_neurons_as_input(graph: GraphModuleBase) -> None:
    all_aliases = find_modules_of_subclass(graph, AliasConnection)

    for a in all_aliases:
        for o_n in a.output_nodes:
            for source in o_n.source_modules:
                if not isinstance(source, (GenericNeurons, AliasConnection)):
                    raise DRCError(
                        f"Alias connections must have neurons as the last block before the output.\nFound aliases {a} with module {source} as the last module in the graph."
                    )


def at_least_two_neuron_layers_needed(graph: GraphModuleBase) -> None:
    all_neurons = find_modules_of_subclass(graph, GenericNeurons)

    if len(all_neurons) < 2:
        raise DRCError(
            "At least two layers of neurons are required to map to hidden and output layers on Xylo."
        )


xylo_drc: List[Callable[[GraphModuleBase], None]] = [
    output_nodes_have_neurons_as_source,
    input_to_neurons_is_a_weight,
    first_module_is_a_weight,
    le_16_input_channels,
    le_8_output_channels,
    all_neurons_have_same_dt,
    output_neurons_cannot_be_recurrent,
    no_consecutive_weights,
    alias_inputs_must_be_neurons,
    alias_output_nodes_must_have_neurons_as_input,
    at_least_two_neuron_layers_needed,
]
""" List[Callable[[GraphModuleBase], None]]: The collection of design rules for Xylo """


def check_drc(
    graph: GraphModuleBase, design_rules: List[Callable[[GraphModuleBase], None]]
):
    """
    Perform a design rule check over a graph

    Args:
        graph (GraphModuleBase): A graph to check
        design_rules (List[Callable[[GraphModuleBase], None]]): A list of functions, each of which performs a DRC over a graph

    Raises:
        DRCError: If a design rule is violated
    """
    for dr in design_rules:
        try:
            dr(graph)
        except DRCError as e:
            raise DRCError(
                f"Design rule {dr.__name__} triggered an error:\n"
                + "".join([f"{msg}" for msg in e.args])
            )


def assign_ids_to_class(
    graph: GraphModuleBase, cls, available_ids: List[int]
) -> List[int]:
    """
    Assign IDs from a list to a class of graph module

    This function sets the :py:attr:`~.graph.GraphModule.hw_ids` attribute for all :py:class:`.graph.GraphModule` s of a chosen subclass, by assigning them greedily from a list. The allocated IDs are removed from the ``available`` list, are set in the graph modules, and are returned as a list.

    Examples:

        >>> output_ids = list(range(16))
        >>> allocated_ids = assign_ids_to_class(graph, XyloOutputNeurons, output_ids)
        >>> print(allocated_ids)
        [0, 1, 2, 3, 4, 5]
        >>> print(output_ids)
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    Args:
        graph (GraphModuleBase): The graph to search over
        cls: The :py:class:`~.graph.GraphModule` subclass to search for, to assign IDs to
        available_ids (List[int]): A list of integer unique HW IDs that can be allocated from. These IDs will be allocated to the graph modules.

    Returns:
        List[int]: The hardware IDs that were allocated to the graph modules
    """
    # - Build a list of ids that are allocated
    allocated_ids = []

    # - Get all modules of the defined class
    modules = find_modules_of_subclass(graph, cls)

    # - Allocate HW ids to these modules
    for m in modules:
        num_needed_ids = len(m.output_nodes)
        if len(available_ids) < num_needed_ids:
            raise DRCError(
                f"Exceeded number of available resources for graph module {m}."
            )

        # - Allocate the IDs and remove them from the available list
        m.hw_ids = available_ids[:num_needed_ids]
        allocated_ids.extend(m.hw_ids)
        del available_ids[:num_needed_ids]

        # - Annotate the original computational module with the allocated hardware IDs, if possible
        if m.computational_module is not None:
            m.computational_module._hw_ids = m.hw_ids

    return allocated_ids


def mapper(
    graph: GraphModuleBase,
    weight_dtype: Union[np.dtype, str] = "float",
    threshold_dtype: Union[np.dtype, str] = "float",
    dash_dtype: Union[np.dtype, str] = "float",
) -> dict:
    """
    Map a computational graph onto the Xylo v1 architecture

    This function performs a DRC of the computational graph to ensure it can be mapped onto the Xylo v1 architecture.

    Warnings:
        :py:func:`mapper` operates **in-place** on the graph, and may modify it. If you need the un-mapped graph, you may need to call :py:meth:`.Module.as_graph` again on your :py:class:`.Module`.

    It then allocates neurons and converts the network weights into a specification for Xylo. This specification can be used to create a config object with :py:func:`~rockpool.devices.xylo.config_from_specification`.

    Args:
        graph (GraphModuleBase): The graph to map
        weight_dtype (Union[np.dtype, str]): Data type for mapped weight parameters. Default: ``"float"``
        threshold_dtype (Union[np.dtype, str]): Data type for mapped threshold parameters. Default: ``"float"``
        dash_dtype (Union[np.dtype, str]): Data type for mapped dash (bitshift time constant) parameters. Default: ``"float"``

    Returns:
        dict: A dictionary of specifications for Xylo v1, containing the mapped computational graph
    """
    # - Check design rules
    check_drc(graph, xylo_drc)

    # --- Replace neuron modules with known graph classes ---

    # - Get output spiking layer from output nodes
    output_neurons: Set[GenericNeurons] = set()
    for on in graph.output_nodes:
        for sm in on.source_modules:
            if isinstance(sm, GenericNeurons):
                output_neurons.add(sm)

    # - Replace these output neurons with XyloOutputNeurons
    for on in output_neurons:
        try:
            XyloOutputNeurons._convert_from(on)
        except Exception as e:
            raise DRCError(f"Error replacing output neuron module {on}.") from e

    # - Replace all other neurons with XyloHiddenNeurons
    nodes, modules = bag_graph(graph)

    for m in modules:
        if isinstance(m, GenericNeurons) and not isinstance(m, XyloOutputNeurons):
            try:
                XyloHiddenNeurons._convert_from(m)
            except Exception as e:
                raise DRCError(f"Error replacing module {m}.") from e

    # --- Assign neurons to HW neurons ---

    # - Enumerate hidden neurons
    available_hidden_neuron_ids = list(range(1000))
    try:
        allocated_hidden_neurons = assign_ids_to_class(
            graph, XyloHiddenNeurons, available_hidden_neuron_ids
        )
    except Exception as e:
        raise DRCError("Failed to allocate HW resources for hidden neurons.") from e

    # - Enumerate output neurons
    available_output_neuron_ids = list(range(1000, 1008))
    try:
        allocated_output_neurons = assign_ids_to_class(
            graph, XyloOutputNeurons, available_output_neuron_ids
        )
    except Exception as e:
        raise DRCError("Failed to allocate HW resources for output neurons.") from e

    # - Enumerate input channels
    input_channels = list(range(len(graph.input_nodes)))

    # - How many synapses are we using for hidden neurons?
    hidden_neurons: SetList[XyloHiddenNeurons] = find_modules_of_subclass(
        graph, XyloHiddenNeurons
    )
    num_hidden_synapses = 1
    for hn in hidden_neurons:
        if len(hn.input_nodes) > len(hn.output_nodes):
            num_hidden_synapses = 2

    # --- Map weights and build Xylo weight matrices ---

    # - Build an input weight matrix
    input_weight_mod: LinearWeights = graph.input_nodes[0].sink_modules[0]
    target_neurons: XyloNeurons = input_weight_mod.output_nodes[0].sink_modules[0]
    # ^ Since DRC passed, we know this is valid

    weight_num_synapses = (
        2 if len(target_neurons.input_nodes) > len(target_neurons.output_nodes) else 1
    )

    target_ids = target_neurons.hw_ids
    these_dest_indices = [allocated_hidden_neurons.index(id) for id in target_ids]

    # - Allocate and assign the input weights
    w_in = np.zeros(
        (len(input_channels), len(allocated_hidden_neurons), num_hidden_synapses),
        weight_dtype,
    )
    w_in[
        np.ix_(input_channels, these_dest_indices, list(range(weight_num_synapses)))
    ] = input_weight_mod.weights.reshape(
        (len(input_channels), len(these_dest_indices), weight_num_synapses)
    )

    # - Build a recurrent weight matrix
    w_rec = np.zeros(
        (
            len(allocated_hidden_neurons),
            len(allocated_hidden_neurons),
            num_hidden_synapses,
        ),
        weight_dtype,
    )
    w_rec_source_ids = allocated_hidden_neurons
    w_rec_dest_ids = allocated_hidden_neurons

    # - Build an output weight matrix
    w_out = np.zeros(
        (len(allocated_hidden_neurons), len(allocated_output_neurons)), weight_dtype
    )
    w_out_source_ids = allocated_hidden_neurons
    w_out_dest_ids = allocated_output_neurons

    # - Get all weights
    weights: SetList[LinearWeights] = find_modules_of_subclass(graph, LinearWeights)
    weights.remove(input_weight_mod)

    # - For each weight module, place the weights in the right place
    for w in weights:
        # - Find the destination neurons
        sm = SetList(
            [
                sm
                for n in w.output_nodes
                for sm in n.sink_modules
                if isinstance(sm, XyloNeurons)
            ]
        )
        target_neurons: XyloNeurons = sm[0]

        # - How many target synapses per neuron?
        num_target_syns = (
            2
            if len(target_neurons.input_nodes) > len(target_neurons.output_nodes)
            else 1
        )

        # - Find the source neurons
        sm = SetList(
            [
                sm
                for n in w.input_nodes
                for sm in n.source_modules
                if isinstance(sm, XyloNeurons)
            ]
        )
        source_neurons: XyloNeurons = sm[0]

        # - Get source and target HW IDs
        source_ids = source_neurons.hw_ids
        target_ids = target_neurons.hw_ids

        # - Does this go in the recurrent or output weights?
        if isinstance(target_neurons, XyloHiddenNeurons):
            # - Recurrent weights
            these_weights = np.reshape(
                w.weights, (len(source_ids), len(target_ids), num_target_syns)
            )
            these_source_indices = [w_rec_source_ids.index(id) for id in source_ids]
            these_dest_indices = [w_rec_dest_ids.index(id) for id in target_ids]

            # - Assign weights
            w_rec[
                np.ix_(
                    these_source_indices, these_dest_indices, np.arange(num_target_syns)
                )
            ] = these_weights

        elif isinstance(target_neurons, XyloOutputNeurons):
            # - Output weights
            these_source_indices = [w_out_source_ids.index(id) for id in source_ids]
            these_dest_indices = [w_out_dest_ids.index(id) for id in target_ids]

            # - Assign weights
            w_out[np.ix_(these_source_indices, these_dest_indices)] = w.weights

        else:
            raise DRCError(
                f"Unexpected target of weight graph module {w}. Expected XyloHiddenNeurons or XyloOutputNeurons."
            )

    # - If we are not using synapse 2, we need to trim the weights
    if num_hidden_synapses == 1:
        w_in = np.reshape(w_in, (len(input_channels), len(allocated_hidden_neurons)))
        w_rec = np.reshape(
            w_rec, (len(allocated_hidden_neurons), len(allocated_hidden_neurons))
        )

    # --- Extract parameters from nodes ---

    hidden_neurons: SetList[XyloHiddenNeurons] = find_modules_of_subclass(
        graph, XyloHiddenNeurons
    )
    output_neurons: SetList[XyloOutputNeurons] = find_modules_of_subclass(
        graph, XyloOutputNeurons
    )
    num_hidden_neurons = len(allocated_hidden_neurons)
    num_output_neurons = len(allocated_output_neurons)

    dash_mem = np.zeros(num_hidden_neurons, dash_dtype)
    dash_mem_out = np.zeros(num_output_neurons, dash_dtype)
    dash_syn = np.zeros(num_hidden_neurons, dash_dtype)
    dash_syn_2 = np.zeros(num_hidden_neurons, dash_dtype)
    dash_syn_out = np.zeros(num_output_neurons, dash_dtype)
    threshold = np.zeros(num_hidden_neurons, threshold_dtype)
    threshold_out = np.zeros(num_output_neurons, threshold_dtype)
    for n in hidden_neurons:
        these_indices = n.hw_ids
        dash_mem[these_indices] = n.dash_mem

        if len(n.input_nodes) > len(n.output_nodes):
            dash_syn_reshape = np.array(n.dash_syn).reshape((-1, 2))
            for i, index in enumerate(these_indices):
                dash_syn[index] = dash_syn_reshape[i][0]
                dash_syn_2[index] = dash_syn_reshape[i][1]
        else:
            for i, index in enumerate(these_indices):
                dash_syn[index] = n.dash_syn[i]
        threshold[these_indices] = n.threshold

    for n in output_neurons:
        these_indices = [allocated_output_neurons.index(id) for id in n.hw_ids]
        dash_mem_out[these_indices] = n.dash_mem
        for i, index in enumerate(these_indices):
            dash_syn_out[index] = n.dash_syn[i]
        threshold_out[these_indices] = n.threshold

    neurons: SetList[XyloNeurons] = find_modules_of_subclass(graph, XyloNeurons)
    dt = None
    for n in neurons:
        dt = n.dt if dt is None else dt

    # --- Extract aliases from nodes ---

    aliases = find_modules_of_subclass(graph, AliasConnection)

    list_aliases = [[] for _ in range(num_hidden_neurons)]
    for a in aliases:
        # - Find the source neurons
        sm = SetList(
            [
                sm
                for n in a.input_nodes
                for sm in n.source_modules
                if isinstance(sm, XyloNeurons)
            ]
        )
        source_neurons: XyloNeurons = sm[0]

        # - Find the destination neurons
        sm = SetList(
            [
                sm
                for n in a.output_nodes
                for sm in n.source_modules
                if isinstance(sm, XyloNeurons)
            ]
        )
        target_neurons: XyloNeurons = sm[0]

        # - Get the source and target HW IDs
        source_ids = source_neurons.hw_ids
        target_ids = target_neurons.hw_ids

        # - Add to the aliases list
        for (source, target) in zip(source_ids, target_ids):
            list_aliases[source].append(target)

    return {
        "mapped_graph": graph,
        "weights_in": w_in,
        "weights_out": w_out,
        "weights_rec": w_rec,
        "dash_mem": dash_mem,
        "dash_mem_out": dash_mem_out,
        "dash_syn": dash_syn,
        "dash_syn_2": dash_syn_2,
        "dash_syn_out": dash_syn_out,
        "threshold": threshold,
        "threshold_out": threshold_out,
        "weight_shift_in": 0,
        "weight_shift_rec": 0,
        "weight_shift_out": 0,
        "aliases": list_aliases,
        "dt": dt,
    }
