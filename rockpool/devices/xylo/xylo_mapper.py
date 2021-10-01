from rockpool.graph import (
    GraphModule,
    GraphNode,
    GenericNeurons,
    LinearWeights,
    replace_module,
    bag_graph,
    find_modules_of_class,
)
from rockpool.devices.xylo import XyloHiddenNeurons, XyloOutputNeurons


from typing import List, Callable

__all__ = ["mapper"]


class DRCError(ValueError):
    pass


def output_nodes_have_neurons_as_source(graph: GraphModule):
    # - All output nodes must have a source that is a neuron
    for n in graph.output_nodes:
        for s in n.source_modules:
            if not isinstance(s, GenericNeurons):
                raise DRCError(
                    f"A network output node {n} has a source {s} which is not a neuron"
                )


def input_to_neurons_is_a_weight(graph: GraphModule):
    # - Every neuron module must have weights on the input
    neurons = find_modules_of_class(graph, GenericNeurons)

    for n in neurons:
        for inp in n.input_nodes:
            for sm in inp.source_modules:
                if not isinstance(sm, LinearWeights):
                    raise DRCError(
                        f"A neuron node {n} has a source module {sm} which is not a LinearWeight."
                    )


def first_module_is_a_weight(graph: GraphModule):
    # - The first module after the input must be a set of weights
    for inp in graph.input_nodes:
        for sink in inp.sink_modules:
            if not isinstance(sink, LinearWeights):
                raise DRCError(
                    f"A network input node {inp} has a sink module {sink} which is not a LinearWeight."
                )


xylo_drc = [
    output_nodes_have_neurons_as_source,
    input_to_neurons_is_a_weight,
    first_module_is_a_weight,
]


def check_drc(graph, design_rules: List[Callable[[GraphModule], None]]):
    for dr in design_rules:
        try:
            dr(graph)
        except DRCError as error:
            print(f"Design rule {dr.__module__}.{dr.__name__} triggered an error:")
            raise error


def mapper(graph: GraphModule):
    # - Check design rules
    check_drc(graph, xylo_drc)

    # - Get output spiking layer from output nodes
    output_neurons = set()
    for on in graph.output_nodes:
        for sm in on.source_modules:
            if isinstance(sm, GenericNeurons):
                output_neurons.add(sm)

    # - Replace these output neurons with XyloOutputNeurons
    new_output_neurons = set()
    for on in output_neurons:
        new_output_neurons.add(XyloOutputNeurons._swap(on))

    output_neurons = new_output_neurons

    # - Replace all other neurons with XyloHiddenNeurons
    nodes, modules = bag_graph(graph)

    for m in modules:
        if isinstance(m, GenericNeurons) and not isinstance(m, XyloOutputNeurons):
            XyloHiddenNeurons._swap(m)

    # - Enumerate neurons
    available_hidden_neuron_ids = list(range(1000))
    hidden_neurons = find_modules_of_class(graph, XyloHiddenNeurons)

    for n in hidden_neurons:
        num_needed_ids = len(n.output_nodes)
        n.hw_ids = available_hidden_neuron_ids.pop()

    return graph
