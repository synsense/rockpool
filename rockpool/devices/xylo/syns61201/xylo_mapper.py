"""
Mapper package for Xylo core v2

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
from .xylo_graph_modules import (
    Xylo2HiddenNeurons,
    Xylo2OutputNeurons,
    Xylo2Neurons,
)
from rockpool.devices.xylo.syns61300.xylo_mapper import (
    xylo_drc,
    DRCError,
    DRCWarning,
    check_drc,
    assign_ids_to_class,
)

from typing import List, Callable, Set, Optional, Union

__all__ = ["mapper", "DRCError", "DRCWarning"]


def mapper(
    graph: GraphModuleBase,
    weight_dtype: Union[np.dtype, str] = "float",
    threshold_dtype: Union[np.dtype, str] = "float",
    dash_dtype: Union[np.dtype, str] = "float",
    max_hidden_neurons: int = 1000,
    max_output_neurons: int = 8,
) -> dict:
    """
    Map a computational graph onto the Xylo v2 (SYNS61201) architecture

    This function performs a DRC of the computational graph to ensure it can be mapped onto the Xylo v2 (SYNS61201) architecture.

    Warnings:
        :py:func:`mapper` operates **in-place** on the graph, and may modify it. If you need the un-mapped graph, you may need to call :py:meth:`.Module.as_graph` again on your :py:class:`.Module`.

    It then allocates neurons and converts the network weights into a specification for Xylo. This specification can be used to create a config object with :py:func:`~rockpool.devices.xylo.syns61201.config_from_specification`.

    Args:
        graph (GraphModuleBase): The graph to map
        weight_dtype (Union[np.dtype, str]): Data type for mapped weight parameters. Default: ``"int8"``
        threshold_dtype (Union[np.dtype, str]): Data type for mapped threshold parameters. Default: ``"int16"``
        dash_dtype (Union[np.dtype, str]): Data type for mapped dash (bitshift time constant) parameters. Default: ``"uint8"``
        max_hidden_neurons (int): Maximum number of available hidden neurons. Default: ``1000``, matching Xylo hardware
        max_output_neurons (int): Maximum number of available output neurons. Default: ``8``, matching Xylo hardware

    Returns:
        dict: A dictionary of specifications for Xylo v2, containing the mapped computational graph
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
            Xylo2OutputNeurons._convert_from(on)
        except Exception as e:
            raise DRCError(f"Error replacing output neuron module {on}.") from e

    # - Replace all other neurons with XyloHiddenNeurons
    nodes, modules = bag_graph(graph)

    for m in modules:
        if isinstance(m, GenericNeurons) and not isinstance(m, Xylo2OutputNeurons):
            try:
                Xylo2HiddenNeurons._convert_from(m)
            except Exception as e:
                raise DRCError(f"Error replacing module {m}.") from e

    # --- Assign neurons to HW neurons ---

    # - Enumerate hidden neurons
    available_hidden_neuron_ids = list(range(max_hidden_neurons))
    try:
        allocated_hidden_neurons = assign_ids_to_class(
            graph, Xylo2HiddenNeurons, available_hidden_neuron_ids
        )
    except Exception as e:
        raise DRCError("Failed to allocate HW resources for hidden neurons.") from e

    # - Enumerate output neurons
    available_output_neuron_ids = list(
        range(max_hidden_neurons, max_hidden_neurons + max_output_neurons)
    )
    try:
        allocated_output_neurons = assign_ids_to_class(
            graph, Xylo2OutputNeurons, available_output_neuron_ids
        )
    except Exception as e:
        raise DRCError("Failed to allocate HW resources for output neurons.") from e

    # - Enumerate input channels
    input_channels = list(range(len(graph.input_nodes)))

    # - How many synapses are we using for hidden neurons?
    hidden_neurons: SetList[Xylo2HiddenNeurons] = find_modules_of_subclass(
        graph, Xylo2HiddenNeurons
    )
    num_hidden_synapses = 1
    for hn in hidden_neurons:
        if len(hn.input_nodes) > len(hn.output_nodes):
            num_hidden_synapses = 2

    # --- Map weights and build Xylo weight matrices ---

    # - Build an input weight matrix
    input_weight_mod: LinearWeights = graph.input_nodes[0].sink_modules[0]
    target_neurons: Xylo2Neurons = input_weight_mod.output_nodes[0].sink_modules[0]
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
                if isinstance(sm, Xylo2Neurons)
            ]
        )
        target_neurons: Xylo2Neurons = sm[0]

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
                if isinstance(sm, Xylo2Neurons)
            ]
        )
        source_neurons: Xylo2Neurons = sm[0]

        # - Get source and target HW IDs
        source_ids = source_neurons.hw_ids
        target_ids = target_neurons.hw_ids

        # - Does this go in the recurrent or output weights?
        if isinstance(target_neurons, Xylo2HiddenNeurons):
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

        elif isinstance(target_neurons, Xylo2OutputNeurons):
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

    hidden_neurons: SetList[Xylo2HiddenNeurons] = find_modules_of_subclass(
        graph, Xylo2HiddenNeurons
    )
    output_neurons: SetList[Xylo2OutputNeurons] = find_modules_of_subclass(
        graph, Xylo2OutputNeurons
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
    bias = np.zeros(num_hidden_neurons, weight_dtype)
    bias_out = np.zeros(num_output_neurons, weight_dtype)

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
        bias[these_indices] = n.bias

    for n in output_neurons:
        these_indices = [allocated_output_neurons.index(id) for id in n.hw_ids]
        dash_mem_out[these_indices] = n.dash_mem

        for i, index in enumerate(these_indices):
            dash_syn_out[index] = n.dash_syn[i]

        threshold_out[these_indices] = n.threshold
        bias_out[these_indices] = n.bias

    neurons: SetList[Xylo2Neurons] = find_modules_of_subclass(graph, Xylo2Neurons)
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
                if isinstance(sm, Xylo2Neurons)
            ]
        )
        source_neurons: Xylo2Neurons = sm[0]

        # - Find the destination neurons
        sm = SetList(
            [
                sm
                for n in a.output_nodes
                for sm in n.source_modules
                if isinstance(sm, Xylo2Neurons)
            ]
        )
        target_neurons: Xylo2Neurons = sm[0]

        # - Get the source and target HW IDs
        source_ids = source_neurons.hw_ids
        target_ids = target_neurons.hw_ids

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
