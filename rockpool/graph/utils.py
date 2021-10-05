from rockpool.graph.graph_base import GraphModule, GraphHolder, GraphNode, SetList

import copy

__all__ = [
    "connect_modules",
    "bag_graph",
    "find_modules_of_class",
    "replace_module",
    "find_recurrent_modules",
]


def connect_modules(source: GraphModule, dest: GraphModule):
    # - Check channel dimensions
    if len(source.output_nodes) != len(dest.input_nodes):
        raise ValueError(
            f"Connecting {source.name} and {dest.name}. Number of output nodes {len(source.output_nodes)} does not match number of input nodes {len(dest.input_nodes)}."
        )

    # - Wire up modules over nodes. Keep only the output nodes from the source module.
    for node_index in range(len(source.output_nodes)):
        # - Get corresponding source and dest nodes to merge
        s_o_node = source.output_nodes[node_index]
        d_i_node = dest.input_nodes[node_index]

        # - Copy all sinks and sources from dest node into source node
        [s_o_node.add_sink(t) for t in d_i_node.sink_modules]
        [s_o_node.add_source(s) for s in d_i_node.source_modules]

        # - Remove all module links from dest node
        del d_i_node.source_modules[:]
        del d_i_node.sink_modules[:]

    # - Replace input node in all sink objects with the connected output nodes
    dest_nodes = copy.copy(dest.input_nodes)
    for node_index in range(len(source.output_nodes)):
        # - Get corresponding source and dest nodes
        s_o_node = source.output_nodes[node_index]
        d_i_node = dest_nodes[node_index]

        # - For all source modules to this node, replace the node on the output
        for sm in s_o_node.source_modules:
            sm.remove_output(d_i_node)
            sm.add_output(s_o_node)

        # - For all sink modules to this node, replace the node on the input
        for sink in s_o_node.sink_modules:
            sink.remove_input(d_i_node)
            sink.add_input(s_o_node)


def bag_graph(
    graph: GraphModule,
    nodes_bag: SetList[GraphNode] = None,
    modules_bag: SetList[GraphModule] = None,
) -> (SetList[GraphNode], SetList[GraphModule]):
    nodes_bag = SetList() if nodes_bag is None else nodes_bag
    modules_bag = SetList() if modules_bag is None else modules_bag

    # - Have we seen this module before?
    if graph not in modules_bag:
        # - Add this module to the bag
        modules_bag._add_unique(graph)

        # - Add input and output nodes to bag
        [nodes_bag._add_unique(n) for n in graph.input_nodes]
        [nodes_bag._add_unique(n) for n in graph.output_nodes]

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
            modules_bag_to_return._add_unique(mod)

    return nodes_bag, modules_bag_to_return


def find_modules_of_class(graph: GraphModule, cls) -> SetList[GraphModule]:
    _, modules_bag = bag_graph(graph)
    return SetList(m for m in modules_bag if isinstance(m, cls))


def replace_module(target_module: GraphModule, replacement_module: GraphModule):
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


def find_recurrent_modules(graph: GraphModule) -> SetList[GraphModule]:
    _, modules = bag_graph(graph)

    recurrent_modules = SetList()
    for m in modules:
        # - Get a collection of all source modules
        source_modules = SetList()
        [
            source_modules._append_unique(i_node.source_modules)
            for i_node in m.input_nodes
        ]

        # - Get a collection of all destination modules
        dest_modules = SetList()
        [dest_modules._append_unique(d_node.sink_modules) for d_node in m.output_nodes]

        # - Detect duplicates between source and destination modules
        if len(set(source_modules).intersection(dest_modules)) > 0:
            recurrent_modules._add_unique(m)

    return recurrent_modules
