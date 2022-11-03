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

__all__ = ["mapper", "DRCError", "DRCWarning"]

from typing import Any, Dict, Optional, Union

import numpy as np

from rockpool.graph import GraphModuleBase

from rockpool.devices.dynapse.default import dlayout


class DRCError(ValueError):
    pass


class DRCWarning(Warning, DRCError):
    pass


def mapper(
    graph: GraphModuleBase,
) -> Dict[str, float]:
    """
    mapper mapps a computational graph onto Dynap-SE2 architecture

    returns a specification object which can be used to create a config object

    :param graph: _description_
    :type graph: GraphModuleBase
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
    Iw_base = None

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
        "Iw_base": Iw_base,
    }
