from rockpool.graph.graph_base import GraphModule

from typing import Set

__all__ = ["connect_modules", "bag_graph", "find_modules_of_class", "replace_module"]


def connect_modules(source: GraphModule, dest: GraphModule):
    # - Check channel dimensions
    if len(source.output_nodes) != len(dest.input_nodes):
        raise ValueError(
            f"Connecting {source.name} and {dest.name}. Number of output nodes {len(source.output_nodes)} does not match number of input nodes {len(dest.input_nodes)}."
        )

    # - Wire up modules over nodes
    for o_node, i_node in zip(source.output_nodes, dest.input_nodes):
        [o_node.add_target(t) for t in i_node.sink_modules]
        [i_node.add_source(s) for s in o_node.source_modules]

    # - Replace nodes for output module
    dest.input_nodes = source.output_nodes


def bag_graph(
    graph: GraphModule, nodes_bag: Set = None, modules_bag: Set = None
) -> (Set, Set):
    nodes_bag = set() if nodes_bag is None else nodes_bag
    modules_bag = set() if modules_bag is None else modules_bag

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

    return nodes_bag, modules_bag


def find_modules_of_class(graph: GraphModule, cls) -> Set[GraphModule]:
    _, modules_bag = bag_graph(graph)
    return set(m for m in modules_bag if isinstance(m, cls))


def replace_module(target_module: GraphModule, replacement_module: GraphModule):
    # - Check that the input and output numbers match
    if len(target_module.input_nodes) != len(replacement_module.input_nodes):
        raise ValueError("Number of input nodes do not match")

    if len(target_module.output_nodes) != len(replacement_module.output_nodes):
        raise ValueError("Number of input nodes do not match")

    # - Remove target_module from input and output nodes, replace with replacement_module
    for n in target_module.input_nodes:
        n.sink_modules.remove(target_module)
        n.sink_modules.append(replacement_module)

    for n in target_module.output_nodes:
        n.source_modules.remove(target_module)
        n.source_modules.append(replacement_module)

    # - Include original input and output nodes in replacement_module
    replacement_module.input_nodes = target_module.input_nodes
    replacement_module.output_nodes = target_module.output_nodes


def find_recurrent_modules(graph: GraphModule) -> Set:
    pass
