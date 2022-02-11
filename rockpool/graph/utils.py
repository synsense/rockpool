"""
Utilities for generating and manipulating computational graphs

See Also:
    See :ref:`/advanced/graph_overview.ipynb` for an introduction to computational graphs.

"""


from rockpool.graph.graph_base import (
    GraphModule,
    GraphHolder,
    GraphNode,
    SetList,
    GraphModuleBase,
)

import copy

from typing import Any, Optional, Union, Iterable

__all__ = [
    "connect_modules",
    "bag_graph",
    "find_modules_of_subclass",
    "replace_module",
    "find_recurrent_modules",
]


def connect_modules(
    source: GraphModuleBase,
    dest: GraphModuleBase,
    source_indices: Optional[Iterable[int]] = None,
    dest_indices: Optional[Iterable[int]] = None,
) -> None:
    """
    Connect two :py:class:`.GraphModule` s together

    Connecting two graph modules can only occur if the output and input dimensionality match across the connection. The output :py:class:`.GraphNode` s from the source module will be merged with the input :py:class:`.GraphNodes` of the destination module. The :py:class:`.GraphNode` s of the destination module will then be discarded.

    If ``source`` or ``dest`` are :py:class:`.GraphHolder` s, then the internal subgraphs will be connected, and the :py:class:`.GraphHolder` s may be discarded.

    Examples:
        >>> connect_modules(mod1, mod2)
        # Modules are connected in place, from all output node to all input nodes

        >>> connect_modules(mod1, mod2, range(5))
        # Connect a subset of source output nodes to the destination module
        # Output nodes `mod1.output_nodes[0:5]` are connected to all input nodes `mod2.input_nodes[:]`

        >>> connect_modules(mod1, mod2, None, range(3))
        # All output nodes `mod1.output_nodes[:]` are connected to input nodes `mod2.input_nodes[0:3]`

        >>> connect_modules(mod1, mod2, [0, 2, 4], [1, 2, 5])
        # `mod1` output nodes 0, 2 and 4 are connected to `mod2` input nodes 1, 2, 5

    Args:
        source (GraphModule): The source graph module to connect
        dest (GraphModule): The destination graph module to connect
        source_indices (Optional[Iterable[int]]): The indices of the ``source`` output nodes to connect over. Default: ``None``, use all output nodes
        dest_indices (Optional[Iterable[int]]): The indices of ``dest`` input nodes to connect over. Default: ``None``, use all input nodes
    """
    # - Get indices for source and destination nodes
    source_indices = (
        range(len(source.output_nodes)) if source_indices is None else source_indices
    )
    len_source_nodes = len(source_indices)

    dest_indices = (
        range(len(dest.input_nodes)) if dest_indices is None else dest_indices
    )
    len_dest_nodes = len(dest_indices)

    # - Check channel dimensions
    if len_source_nodes != len_dest_nodes:
        raise ValueError(
            f"Connecting {source.name} and {dest.name}. Number of output nodes {len_source_nodes} does not match number of input nodes {len_dest_nodes}."
        )

    # - Wire up modules over nodes. Keep only the output nodes from the source module.
    for num in range(len_source_nodes):
        # - Get corresponding source and dest nodes index
        source_node_index = source_indices[num]
        dest_node_index = dest_indices[num]

        # - Get corresponding source and dest nodes to merge
        s_o_node = source.output_nodes[source_node_index]
        d_i_node = dest.input_nodes[dest_node_index]

        # - Copy all sinks and sources from dest node into source node
        [s_o_node.add_sink(t) for t in d_i_node.sink_modules]
        [s_o_node.add_source(s) for s in d_i_node.source_modules]

        # - Remove all module links from dest node
        del d_i_node.source_modules[:]
        del d_i_node.sink_modules[:]

    # - Replace input node in all sink objects with the connected output nodes
    #   We need to use a copy to get the correct index; cannot use `[dest.input_nodes[i] for i in dest_index]` since we are modifying `dest.input_nodes`
    dest_nodes = copy.copy(dest.input_nodes)

    for num in range(len_dest_nodes):
        # - Get corresponding source and dest nodes index
        source_node_index = source_indices[num]
        dest_node_index = dest_indices[num]

        # - Get corresponding source and dest nodes
        s_o_node = source.output_nodes[source_node_index]
        d_i_node = dest_nodes[dest_node_index]

        # - For all source modules to this node, replace the node on the output
        for sm in s_o_node.source_modules:
            if d_i_node in sm.output_nodes:
                this_source_node_index = sm.output_nodes.index(d_i_node)
                sm.output_nodes[this_source_node_index] = s_o_node

        # - For all sink modules to this node, replace the node on the input
        for sink in s_o_node.sink_modules:
            if d_i_node in sink.input_nodes:
                this_dest_node_index = sink.input_nodes.index(d_i_node)
                sink.input_nodes[this_dest_node_index] = s_o_node

        # - Replace destination input nodes, if required
        if d_i_node in dest.input_nodes:
            this_dest_node_index = dest.input_nodes.index(d_i_node)
            dest.input_nodes[this_dest_node_index] = s_o_node


def bag_graph(
    graph: GraphModuleBase,
    nodes_bag: Optional[SetList[GraphNode]] = None,
    modules_bag: Optional[SetList[GraphModule]] = None,
) -> (SetList[GraphNode], SetList[GraphModule]):
    """
    Convert a graph into a collection of connection nodes and modules, by traversal

    A graph will be traversed, following all connections. The connection :py:class:`.GraphNode` s and :py:class:`.GraphModule` s will be collected and returned in two collections. Any :py:class:`.GraphHolder` modules will be ignored and discarded.

    Args:
        graph (GraphModuleBase): A graph to analyse
        nodes_bag (SetList): The initial nodes bag. Used in recursive calls. Default: ``None``
        modules_bag (SetList): The initial modules bag. Used in recursive calls. Default: ``None``

    Returns:
        (SetList[GraphNode], SetList[GraphModule]): nodes, modules. `nodes` will be a :py:class:`SetList` containing all the reachable :py:class:`GraphNode` s in `graph`. `modules` will be a :py:class:`SetList` containing all the reachable :py:class:`GraphModule` s in `graph`.
    """
    nodes_bag = SetList() if nodes_bag is None else nodes_bag
    modules_bag = SetList() if modules_bag is None else modules_bag

    # - Have we seen this module before?
    if graph not in modules_bag:
        # - Add this module to the bag
        modules_bag.add(graph)

        # - Add input and output nodes to bag
        [nodes_bag.add(n) for n in graph.input_nodes]
        [nodes_bag.add(n) for n in graph.output_nodes]

        # - Recurse over input nodes
        for n in graph.input_nodes:
            # - Recurse over source modules
            for source in n.source_modules:
                bag_graph(source, nodes_bag, modules_bag)

            for sink in n.sink_modules:
                # - Recurse over sink modules
                bag_graph(sink, nodes_bag, modules_bag)

        # - Recurse over output nodes
        for n in graph.output_nodes:
            for source in n.source_modules:
                # - Recurse over source modules
                bag_graph(source, nodes_bag, modules_bag)

            for sink in n.sink_modules:
                # - Recurse over sink modules
                bag_graph(sink, nodes_bag, modules_bag)

    # - Remove all GraphHolders from the modules bag
    modules_bag_to_return = SetList()
    for mod in modules_bag:
        if not isinstance(mod, GraphHolder):
            modules_bag_to_return.add(mod)

    return nodes_bag, modules_bag_to_return


def find_modules_of_subclass(graph: GraphModuleBase, cls) -> SetList[Any]:
    """
    Search a graph for all :py:class:`.GraphModule` s of a specific class or any subclass

    The search uses `isinstance` to search for ``cls``, so any subclass of ``cls`` will also be found.

    Args:
        graph (GraphModuleBase):
        cls: A class to search for instances of, or instances of any subclass

    Returns:
        SetList[Any]: A collection of objects of the desired class
    """
    _, modules_bag = bag_graph(graph)
    return SetList(m for m in modules_bag if isinstance(m, cls))


def replace_module(target_module: GraphModule, replacement_module: GraphModule) -> None:
    """
    Replace a graph module with a different module

    This function removes a target graph module from a graph, and replaces it with a replacement module. It removes the target module from any connection :py:class:`.GraphNode` s, and wires in the replacement module instead.

    Args:
        target_module (GraphModule): A module inside a graph to replace
        replacement_module (GraphModule): A replacement module to wire into the graph, in place of ``target_module``
    """
    # - Check that the input and output numbers match
    if len(target_module.input_nodes) != len(replacement_module.input_nodes):
        raise ValueError("Number of input nodes do not match")

    if len(target_module.output_nodes) != len(replacement_module.output_nodes):
        raise ValueError("Number of input nodes do not match")

    # - Remove target_module from input and output nodes, replace with replacement_module
    for n in target_module.input_nodes:
        n.remove_sink(target_module)
        n.add_sink(replacement_module)

    for n in target_module.output_nodes:
        n.remove_source(target_module)
        n.add_source(replacement_module)

    # - Include original input and output nodes in replacement_module
    replacement_module.input_nodes = target_module.input_nodes
    replacement_module.output_nodes = target_module.output_nodes


def find_recurrent_modules(graph: GraphModuleBase) -> SetList[GraphModule]:
    """
    Search for graph modules that are connected in a one-module loop

    A "recurrent module" is defined as a graph module that connects with itself via another single graph module. e.g. a module of neurons, connected to a module of weights that itself connects recurrently back from output of the neurons to the input of the neurons.

    Args:
        graph (GraphModuleBase): A graph to search

    Returns:
        SetList[GraphModule]: A collection containing all identified recurrent modules in the graph
    """
    _, modules = bag_graph(graph)

    recurrent_modules = SetList()
    for m in modules:
        # - Get a collection of all source modules
        source_modules = SetList()
        [source_modules.extend(i_node.source_modules) for i_node in m.input_nodes]

        # - Get a collection of all destination modules
        dest_modules = SetList()
        [dest_modules.extend(d_node.sink_modules) for d_node in m.output_nodes]

        # - Detect duplicates between source and destination modules
        if len(set(source_modules).intersection(dest_modules)) > 0:
            recurrent_modules.add(m)

    return recurrent_modules
