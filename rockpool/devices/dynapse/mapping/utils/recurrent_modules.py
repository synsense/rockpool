"""
Dynap-SE graph transformer package

The utility functions and class definitions introduced here can be used to transform any LIF network (with certain constrains)
to a Dynapse computational graph

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

TEMPORARY
09/11/2022
"""
from __future__ import annotations
from typing import List

from rockpool.graph import GraphModule, SetList

__all__ = ["recurrent_modules"]


def recurrent_modules(modules: List[GraphModule]) -> SetList[GraphModule]:
    """
    Search for graph modules that are connected in a one-module loop

    A "recurrent module" is defined as a graph module that connects with itself via another single graph module. e.g. a module of neurons, connected to a module of weights that itself connects recurrently back from output of the neurons to the input of the neurons.

    Args:
        graph (GraphModuleBase): A graph to search

    Returns:
        SetList[GraphModule]: A collection containing all identified recurrent modules in the graph

    [] TODO NOTE : ``rockpool.graph.utils.find_recurrent_modules`` almost the same(one line), just avoid running `bag_graph` twice.
    [] TODO : We can change original implementation (later)
    """

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
