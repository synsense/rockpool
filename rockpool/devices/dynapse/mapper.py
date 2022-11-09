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
from typing import Any, Dict, Optional, List, Tuple, Union

from copy import deepcopy
import numpy as np
from dataclasses import dataclass

from rockpool.graph import GraphModule, GraphNode, SetList, GraphHolder, connect_modules
from rockpool.graph.utils import bag_graph

from rockpool.devices.dynapse.default import dlayout
from rockpool.devices.dynapse.graph import DynapseNeurons
from rockpool.graph.graph_modules import LIFNeuronWithSynsRealValue, LinearWeights

__all__ = ["mapper"]


def mapper(
    graph: GraphModule,
) -> Dict[str, float]:
    """
    mapper mapps a computational graph onto Dynap-SE2 architecture

    returns a specification object which can be used to create a config object

    :param graph: _description_
    :type graph: GraphModule
    :return: _description_
    :rtype: Dict[str, float]
    """

    w_in = None
    w_rec = None

    Idc = None
    If_nmda = None
    Igain_ahp = None
    Igain_ampa = None
    Igain_gaba = None
    Igain_nmda = None
    Igain_shunt = None
    Igain_mem = None
    Ipulse_ahp = None
    Ipulse = None
    Iref = None
    Ispkthr = None
    Itau_ahp = None
    Itau_ampa = None
    Itau_gaba = None
    Itau_nmda = None
    Itau_shunt = None
    Itau_mem = None
    Iw_ahp = None
    Iscale = None

    return {
        "mapped_graph": graph,
        "weights_in": w_in,
        "weights_rec": w_rec,
        "Idc": Idc,
        "If_nmda": If_nmda,
        "Igain_ahp": Igain_ahp,
        "Igain_ampa": Igain_ampa,
        "Igain_gaba": Igain_gaba,
        "Igain_nmda": Igain_nmda,
        "Igain_shunt": Igain_shunt,
        "Igain_mem": Igain_mem,
        "Ipulse_ahp": Ipulse_ahp,
        "Ipulse": Ipulse,
        "Iref": Iref,
        "Ispkthr": Ispkthr,
        "Itau_ahp": Itau_ahp,
        "Itau_ampa": Itau_ampa,
        "Itau_gaba": Itau_gaba,
        "Itau_nmda": Itau_nmda,
        "Itau_shunt": Itau_shunt,
        "Itau_mem": Itau_mem,
        "Iw_ahp": Iw_ahp,
        "Iscale": Iscale,
    }
