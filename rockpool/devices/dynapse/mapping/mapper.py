"""
Dynap-SE graph graph mapper package

- Create a graph using the :py:meth:`~.graph.GraphModule.as_graph` API
- Call :py:func:`.mapper`

Note : Existing modules are reconstructed considering consistency with Xylo support.


Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""
from __future__ import annotations
from typing import Dict

from copy import deepcopy

from rockpool.graph import GraphModule

from .utils import lifnet_to_dynapsim
from .container import DynapseGraphContainer

__all__ = ["mapper"]


def mapper(graph: GraphModule, in_place=False) -> Dict[str, float]:
    """
    mapper maps a computational graph onto Dynap-SE2 architecture.

    It requires post-processing steps such as
        * weight_quantization (float -> 4-bit)
        * parameter_selection (Isyn -> I_ampa, Igaba, Inmda, Ishunt)
        * parameter_quantization (current value -> coarse-fine : 1e-8 -> (3,75))
        * parameter_clustering (allocates seperate cores for different groups)
        * config (create a samna config object to deploy everything to hardware)

    :param graph: Any graph(constraint) aimed to be deployed to Dynap-SE2
    :type graph: GraphModule
    :return: a specification object which can be used to create a config object
    :rtype: Dict[str, float]
    """

    try:
        wrapper = DynapseGraphContainer.from_graph_holder(graph)
    except:
        graph = (
            lifnet_to_dynapsim(deepcopy(graph))
            if not in_place
            else lifnet_to_dynapsim(graph)
        )
        wrapper = DynapseGraphContainer.from_graph_holder(graph)

    specs = {
        "mapped_graph": graph,
        "weights_in": wrapper.w_in,
        "weights_rec": wrapper.w_rec,
        "Iscale": wrapper.Iscale,
    }

    specs.update(wrapper.current_dict)

    return specs
