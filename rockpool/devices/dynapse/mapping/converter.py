"""
Dynap-SE graph convert from a LIF network


Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""
from __future__ import annotations
from typing import Dict

from copy import deepcopy

from rockpool.nn.modules.module import ModuleBase
from rockpool.nn.modules import LinearJax
from rockpool.nn.combinators import Sequential
from rockpool.graph import GraphModule

from rockpool.devices.dynapse.simulation.dynapsim import DynapSim

from rockpool.devices.dynapse.mapping.utils import lifnet_to_dynapsim
from rockpool.devices.dynapse.mapping.container import DynapseGraphContainer

__all__ = ["dynapsim_from_graph"]


def dynapsim_from_graph(graph: GraphModule, in_place=False) -> ModuleBase:
    """
    converter converts a lif net graph to sequential dynapsim network combinator

    :param graph: Any graph(constraint) aimed to be simulated by DynapSim
    :type graph: GraphModule
    :return: a sequential combinator possibly encapsulating a ``LinearJax`` layer and a ``DynapSim`` layer, or just a ``DynapSim`` layer in the case that no input weights defined
    :rtype: ModuleBase
    """

    # Convert the graph only if it's absolutely necessary
    try:
        wrapper = DynapseGraphContainer.from_graph_holder(graph)
    except:
        graph = (
            lifnet_to_dynapsim(deepcopy(graph))
            if not in_place
            else lifnet_to_dynapsim(graph)
        )
        wrapper = DynapseGraphContainer.from_graph_holder(graph)

    # Construct the layers
    in_layer = (
        LinearJax.from_graph(wrapper.input) if wrapper.input is not None else None
    )
    dynapsim_layer = DynapSim.from_graph(wrapper.simulator, wrapper.recurrent)

    # The resulting sequential module ! :tada:
    if in_layer is None:
        mod = dynapsim_layer
    else:
        mod = Sequential(in_layer, dynapsim_layer)

    return mod
