"""
Dynap-SE graph conversion package

The utility functions and class definitions introduced here can be used to transform any LIF network (with certain constrains)
to a Dynapse computational graph

* Non User Facing *
"""

from __future__ import annotations
from typing import List, Tuple, Union

import numpy as np

from rockpool.graph import GraphModule, GraphNode, SetList, GraphHolder, connect_modules
from rockpool.graph.utils import find_recurrent_modules

from rockpool.devices.dynapse.mapping import DynapseNeurons
from rockpool.graph.graph_modules import LIFNeuronWithSynsRealValue, LinearWeights

from .grid import NPGrid
from .state_machine import DFA_Placement

__all__ = ["lifnet_to_dynapsim"]


def lifnet_to_dynapsim(graph: GraphModule) -> Tuple[np.ndarray]:
    """
    lifnet_to_dynapsim applies breath first search in computational graph and installs the weights found in the computational graph
    to one dense weight matrix

    :param graph: the graph head
    :type graph: GraphModule
    :raises ValueError: LIF is at unexpected position! Reshape your network!
    :raises ValueError: LIF Recurrent weights are at unexpected position! Reshape your network!
    :raises ValueError: Linear weights are at unexpected position! Reshape your network!
    :raises TypeError: The graph module is not recognized!
    :raises ValueError: Some modules are not visited!
    :return: w_in, w_rec
        :w_in: extended linear input weights
        :w_rec: merged recurrent weights
    :rtype: Tuple[np.ndarray]
    """

    ### --- Preprocessing --- ###

    # - Get a list of all modules
    modules, rec_modules = find_recurrent_modules(graph)

    # - Compose a grid
    gl = __get_grid_lines(modules)
    ffwd_grid = [NPGrid(row, col) for row, col in zip(gl, gl[1:])]
    rec_grid = [NPGrid(row, col) for row, col in zip(gl, gl)]

    # - Create the weight matrices
    n_in = len(graph.input_nodes)
    n_rec = gl[-1][-1]

    ## - weights
    w_in = np.zeros((n_in, n_rec))
    w_rec = np.zeros((n_rec, n_rec))

    # - Input grid
    in_grid = NPGrid((0, n_in), gl[0])

    # - State control
    layer = -1
    state = DFA_Placement()

    # - Enqueue input nodes
    queue: List[GraphNode] = []
    visited: List[GraphModule] = []
    lif_layers: List[Union[LIFNeuronWithSynsRealValue, DynapseNeurons]] = []
    queue.extend(graph.input_nodes)

    ### --- Stateful BFS --- ###

    while queue:
        # Dequeue a vertex and process
        node = queue.pop(0)
        for sink in node.sink_modules:
            # Enqueue only output nodes
            if sink not in visited:
                visited.append(sink)
                queue.extend(sink.output_nodes)

                # LIF layer found
                if isinstance(sink, (LIFNeuronWithSynsRealValue, DynapseNeurons)):
                    # Check state
                    if not state.lif:
                        raise ValueError(
                            "LIF is at unexpected position! Reshape your network!"
                        )

                    # Store the lif layers in order for later transformation
                    lif_layers.append(sink)

                    # State transition
                    state.next()
                    layer += 1

                # Weight layer found
                elif isinstance(sink, LinearWeights):
                    if sink.biases is not None:
                        raise ValueError("Linear layers should not have biases!")

                    # Recurrent weights
                    if sink in rec_modules:
                        if not state.rec:
                            raise ValueError(
                                "LIF Recurrent weights are at unexpected position! Reshape your network!"
                            )

                        # Place the weights
                        rec_grid[layer].place(w_rec, sink.weights)

                        # State transition
                        state.next(flag_rec=True)

                    # Feed-forward weights
                    else:
                        if not state.linear:
                            raise ValueError(
                                "Linear weights are at unexpected position! Reshape your network!"
                            )

                        # Place the weights
                        if layer >= 0:
                            # post-lift linear layer
                            ffwd_grid[layer].place(w_rec, sink.weights)
                        else:
                            # initial linear layer
                            in_grid.place(w_in, sink.weights)

                            linear_in: LinearWeights = LinearWeights._factory(
                                size_in=w_in.shape[0],
                                size_out=w_in.shape[1],
                                name=sink.name,
                                computational_module=sink.computational_module,
                                weights=w_in,
                            )

                        # State transition
                        state.next(flag_rec=False)

                else:
                    raise TypeError("The graph module is not recognized!")

    # Check if all the layers visited
    if sorted(visited, key=lambda m: id(m)) != sorted(modules, key=lambda m: id(m)):
        raise ValueError("Some modules are not visited!")

    # - Module transormation
    __se = [DynapseNeurons._convert_from(lif) for lif in lif_layers]
    se_layer = DynapseNeurons.merge(__se)

    # - Connecting
    connect_modules(linear_in, se_layer)

    linear_rec = LinearWeights(
        se_layer.output_nodes,
        se_layer.input_nodes,
        name=f"{se_layer.name}_recurrent_{id(se_layer)}",
        computational_module=se_layer.computational_module,
        weights=w_rec,
    )

    return GraphHolder(
        linear_in.input_nodes,
        se_layer.output_nodes,
        f"{graph.name}_transformed_SE_{id(graph)}",
        graph.computational_module,
    )


### --- Private Section --- ###
def __get_grid_lines(modules: List[GraphModule]) -> List[Tuple[int]]:
    """
    __get_grid_lines investigates the list of modules given and finds the grid lines to be used in weight installation

    :param modules: the list of module to investigate
    :type modules: List[GraphModule]
    :raises ValueError: No LIF layers found!
    :return: _description_
    :rtype: List[Tuple[int]]
    """

    # - Get the number of neurons represented at each LIF layer
    len_list = []
    for mod in modules:
        if isinstance(mod, (LIFNeuronWithSynsRealValue, DynapseNeurons)):
            len_list.append(len(mod.output_nodes))

    if not len_list:
        raise ValueError("No LIF layers found!")

    # - Accumulate the number of neurons to get a list of region of interests
    cum = 0
    roi = [cum]
    for temp in len_list:
        cum += temp
        roi.append(cum)

    # - Compose the grid lines
    grid_lines = [(_min, _max) for _min, _max in zip(roi, roi[1:])]
    return grid_lines
