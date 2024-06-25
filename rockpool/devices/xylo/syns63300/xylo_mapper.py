"""
Mapper package for Xylo IMU core

- Create a graph using the :py:meth:`~.graph.GraphModule.as_graph` API
- Call :py:func:`.mapper`

"""

import numpy as np

import copy
import warnings

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
from .xylo_graph_modules import (
    XyloIMUHiddenNeurons,
    XyloIMUOutputNeurons,
    XyloIMUNeurons,
)
from rockpool.devices.xylo.syns61300.xylo_mapper import (
    DRCError,
    DRCWarning,
    check_drc,
    assign_ids_to_module,
    assign_ids_to_class,
    output_nodes_have_neurons_as_source,
    input_to_neurons_is_a_weight,
    first_module_is_a_weight,
    le_16_input_channels,
    all_neurons_have_same_dt,
    output_neurons_cannot_be_recurrent,
    no_consecutive_weights,
    alias_inputs_must_be_neurons,
    alias_output_nodes_must_have_neurons_as_input,
    at_least_two_neuron_layers_needed,
    weight_nodes_have_no_biases,
)

from typing import List, Callable, Set, Optional, Union, Any

__all__ = ["mapper", "DRCError", "DRCWarning"]


def le_16_output_channels(graph: GraphModuleBase) -> None:
    if len(graph.output_nodes) > 16:
        warnings.warn(
            DRCWarning(
                f"Xylo-IMU only supports up to 16 output channels. The network requires {len(graph.output_nodes)} output channels."
            ),
            DRCWarning,
        )


def le_128_input_expansion_neurons(graph: GraphModuleBase) -> None:
    pass


def le_128_output_expansion_neurons(graph: GraphModuleBase) -> None:
    pass


def max_synapses_per_neuron(graph: GraphModuleBase, max_syns: int = 1) -> None:
    # - Neuron modules may only have `max_syns` synapses
    neurons = find_modules_of_subclass(graph, GenericNeurons)

    for n in neurons:
        num_syns = int(len(n.input_nodes) / len(n.output_nodes))

        if num_syns > max_syns:
            raise DRCError(
                f"Neurons may only have {max_syns} synapse(s) per neuron for this device.\nA neuron node {n} has {num_syns} synapses per neuron."
            )


xylo_drc: List[Callable[[GraphModuleBase], None]] = [
    output_nodes_have_neurons_as_source,
    input_to_neurons_is_a_weight,
    first_module_is_a_weight,
    le_16_input_channels,
    le_16_output_channels,
    le_128_input_expansion_neurons,
    le_128_output_expansion_neurons,
    all_neurons_have_same_dt,
    output_neurons_cannot_be_recurrent,
    no_consecutive_weights,
    alias_inputs_must_be_neurons,
    alias_output_nodes_must_have_neurons_as_input,
    at_least_two_neuron_layers_needed,
    weight_nodes_have_no_biases,
    max_synapses_per_neuron,
]
""" List[Callable[[GraphModuleBase], None]]: The collection of design rules for Xylo """


def mapper(
    graph: GraphModuleBase,
    weight_dtype: Union[np.dtype, str] = "float",
    threshold_dtype: Union[np.dtype, str] = "float",
    dash_dtype: Union[np.dtype, str] = "float",
    max_hidden_neurons: int = 496,
    max_output_neurons: int = 16,
) -> dict:
    """
    Map a computational graph onto the Xylo IMU architecture

    This function performs a DRC of the computational graph to ensure it can be mapped onto the Xylo IMU architecture.

    Warnings:
        :py:func:`mapper` operates **in-place** on the graph, and may modify it. If you need the un-mapped graph, you may need to call :py:meth:`.Module.as_graph` again on your :py:class:`.Module`.

    It then allocates neurons and converts the network weights into a specification for Xylo. This specification can be used to create a config object with :py:func:`~rockpool.devices.xylo.syns63300.config_from_specification`.

    Args:
        graph (GraphModuleBase): The graph to map
        weight_dtype (Union[np.dtype, str]): Data type for mapped weight parameters. Default: ``"int8"``
        threshold_dtype (Union[np.dtype, str]): Data type for mapped threshold parameters. Default: ``"int16"``
        dash_dtype (Union[np.dtype, str]): Data type for mapped dash (bitshift time constant) parameters. Default: ``"uint8"``
        max_hidden_neurons (int): Maximum number of available hidden neurons. Default: ``496``, matching Xylo-IMU hardware
        max_output_neurons (int): Maximum number of available output neurons. Default: ``16``, matching Xylo-IMU hardware

    Returns:
        dict: A dictionary of specifications for Xylo IMU, containing the mapped computational graph
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
            XyloIMUOutputNeurons._convert_from(on)
        except Exception as e:
            raise DRCError(f"Error replacing output neuron module {on}.") from e

    # - Replace all other neurons with XyloHiddenNeurons
    nodes, modules = bag_graph(graph)

    for m in modules:
        if isinstance(m, GenericNeurons) and not isinstance(m, XyloIMUOutputNeurons):
            try:
                XyloIMUHiddenNeurons._convert_from(m)
            except Exception as e:
                raise DRCError(f"Error replacing module {m}.") from e

    # --- Assign neurons to HW neurons ---

    # - Get all hidden neurons
    all_hidden_neuron_mods: XyloIMUHiddenNeurons = find_modules_of_subclass(
        graph, XyloIMUHiddenNeurons
    )

    # - Identify input expansion neurons --- these need to be allocated first (lowest neuron IDs)
    # - Find the input weights and their target IENs
    input_weight_mod: LinearWeights = graph.input_nodes[0].sink_modules[0]
    iens_mod: XyloIMUHiddenNeurons = input_weight_mod.output_nodes[0].sink_modules[0]
    num_IENs = len(iens_mod.dash_mem)

    # - Identify output expansion neurons --- these need to be allocated last (highest neuron IDs)
    output_neurons: XyloIMUOutputNeurons = graph.output_nodes[0].source_modules[0]
    list_output_weights_mods: List[LinearWeights] = output_neurons.input_nodes[
        0
    ].source_modules
    list_oens_mods: List[XyloIMUHiddenNeurons] = [
        sm
        for ow in list_output_weights_mods
        for sm in ow.input_nodes[0].source_modules
        if isinstance(sm, XyloIMUHiddenNeurons)
    ]
    num_OENs = sum([len(oen.dash_mem) for oen in list_oens_mods])

    # - Remove IENs and OENs from all hidden neurons
    non_ien_oen_hidden_mods = [
        hn
        for hn in all_hidden_neuron_mods
        if (hn is not iens_mod) and (hn not in list_oens_mods)
    ]

    # - Allocate hidden neurons
    available_hidden_neuron_ids = list(range(max_hidden_neurons))
    try:
        allocated_iens = assign_ids_to_module(iens_mod, available_hidden_neuron_ids)

        allocated_hidden_neurons = [
            assign_ids_to_module(m, available_hidden_neuron_ids)
            for m in non_ien_oen_hidden_mods
        ]
        allocated_hidden_neurons = [
            nid for ids in allocated_hidden_neurons for nid in ids
        ]

        allocated_oens = [
            assign_ids_to_module(m, available_hidden_neuron_ids) for m in list_oens_mods
        ]
        allocated_oens = [nid for ids in allocated_oens for nid in ids]

        all_allocated_hidden_neurons = (
            allocated_iens + allocated_hidden_neurons + allocated_oens
        )

    except Exception as e:
        raise DRCError("Failed to allocate HW resources for hidden neurons.") from e

    # - Enumerate output neurons
    available_output_neuron_ids = list(
        range(max_hidden_neurons, max_hidden_neurons + max_output_neurons)
    )
    try:
        allocated_output_neurons = assign_ids_to_class(
            graph, XyloIMUOutputNeurons, available_output_neuron_ids
        )
    except Exception as e:
        raise DRCError("Failed to allocate HW resources for output neurons.") from e

    # - Enumerate input channels
    input_channels = list(range(len(graph.input_nodes)))

    # # - Extract hidden neurons modules
    # hidden_neurons: SetList[XyloIMUHiddenNeurons] = find_modules_of_subclass(
    #     graph, XyloIMUHiddenNeurons
    # )

    # - Xylo-IMU only provides one synapse per neuron
    num_hidden_synapses = 1

    # --- Map weights and build Xylo weight matrices ---

    # -- Build an input weight matrix
    # - Xylo-IMU only provides one synapse per neuron
    weight_num_synapses = 1

    target_ids = iens_mod.hw_ids
    these_dest_indices = [allocated_iens.index(id) for id in target_ids]

    # - Allocate and assign the input weights
    w_in = np.zeros(
        (len(input_channels), num_IENs, num_hidden_synapses),
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
            len(all_allocated_hidden_neurons),
            len(all_allocated_hidden_neurons),
            num_hidden_synapses,
        ),
        weight_dtype,
    )
    w_rec_source_ids = all_allocated_hidden_neurons
    w_rec_dest_ids = all_allocated_hidden_neurons

    # - Build an output weight matrix
    w_out = np.zeros((num_OENs, len(allocated_output_neurons)), weight_dtype)
    w_out_source_ids = allocated_oens
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
                if isinstance(sm, XyloIMUNeurons)
            ]
        )
        target_neurons: XyloIMUNeurons = sm[0]

        # - Xylo-IMU only provides one synapse per neuron
        num_target_syns = 1

        # - Find the source neurons
        sm = SetList(
            [
                sm
                for n in w.input_nodes
                for sm in n.source_modules
                if isinstance(sm, XyloIMUNeurons)
            ]
        )
        source_neurons: XyloIMUNeurons = sm[0]

        # - Get source and target HW IDs
        source_ids = source_neurons.hw_ids
        target_ids = target_neurons.hw_ids

        # - Does this go in the recurrent or output weights?
        if isinstance(target_neurons, XyloIMUHiddenNeurons):
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

        elif isinstance(target_neurons, XyloIMUOutputNeurons):
            # - Output weights
            these_source_indices = [w_out_source_ids.index(id) for id in source_ids]
            these_dest_indices = [w_out_dest_ids.index(id) for id in target_ids]

            # - Assign weights
            w_out[np.ix_(these_source_indices, these_dest_indices)] = w.weights

        else:
            raise DRCError(
                f"Unexpected target of weight graph module {w}. Expected XyloHiddenNeurons or XyloOutputNeurons."
            )

    # --- Extract parameters from nodes ---

    hidden_neurons: SetList[XyloIMUHiddenNeurons] = find_modules_of_subclass(
        graph, XyloIMUHiddenNeurons
    )
    output_neurons: SetList[XyloIMUOutputNeurons] = find_modules_of_subclass(
        graph, XyloIMUOutputNeurons
    )
    num_hidden_neurons = len(all_allocated_hidden_neurons)
    num_output_neurons = len(allocated_output_neurons)

    dash_mem = np.zeros(num_hidden_neurons, dash_dtype)
    dash_mem_out = np.zeros(num_output_neurons, dash_dtype)
    dash_syn = np.zeros(num_hidden_neurons, dash_dtype)
    dash_syn_2 = np.zeros(num_hidden_neurons, dash_dtype)
    dash_syn_out = np.zeros(num_output_neurons, dash_dtype)
    threshold = np.zeros(num_hidden_neurons, threshold_dtype)
    threshold_out = np.zeros(num_output_neurons, threshold_dtype)
    bias = np.zeros(num_hidden_neurons, weight_dtype)
    bias_out = np.zeros(num_output_neurons, weight_dtype)

    for n in hidden_neurons:
        these_indices = n.hw_ids
        dash_mem[these_indices] = n.dash_mem

        for i, index in enumerate(these_indices):
            dash_syn[index] = n.dash_syn[i]

        threshold[these_indices] = n.threshold
        bias[these_indices] = n.bias

    for n in output_neurons:
        these_indices = [allocated_output_neurons.index(id) for id in n.hw_ids]
        dash_mem_out[these_indices] = n.dash_mem

        for i, index in enumerate(these_indices):
            dash_syn_out[index] = n.dash_syn[i]

        threshold_out[these_indices] = n.threshold
        bias_out[these_indices] = n.bias

    neurons: SetList[XyloIMUNeurons] = find_modules_of_subclass(graph, XyloIMUNeurons)
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
                if isinstance(sm, XyloIMUNeurons)
            ]
        )
        source_neurons: XyloIMUNeurons = sm[0]

        # - Find the destination neurons
        sm = SetList(
            [
                sm
                for n in a.output_nodes
                for sm in n.source_modules
                if isinstance(sm, XyloIMUNeurons)
            ]
        )
        iens_mod: XyloIMUNeurons = sm[0]

        # - Get the source and target HW IDs
        source_ids = source_neurons.hw_ids
        target_ids = iens_mod.hw_ids

        # - Add to the aliases list
        for source, target in zip(source_ids, target_ids):
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
        "bias": bias,
        "bias_out": bias_out,
        "weight_shift_in": 0,
        "weight_shift_rec": 0,
        "weight_shift_out": 0,
        "aliases": list_aliases,
        "dt": dt,
    }
