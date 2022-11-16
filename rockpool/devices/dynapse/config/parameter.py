"""
Dynap-SE parameter selection functions 

Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

09/11/2022
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Union
from rockpool.typehints import FloatVector

from copy import deepcopy
import numpy as np
from dataclasses import dataclass

from rockpool.graph import GraphModule, GraphNode, SetList, GraphHolder, connect_modules
from rockpool.graph.utils import bag_graph

from rockpool.devices.dynapse.graph import DynapseNeurons
from rockpool.graph.graph_modules import LIFNeuronWithSynsRealValue, LinearWeights

### --- Utility Functions --- ###
# def parameter_clustering():
#     pass


# def parameter_quantization(
#     Idc: FloatVector,
#     If_nmda: FloatVector,
#     Igain_ahp: FloatVector,
#     Igain_mem: FloatVector,
#     Igain_syn: FloatVector,
#     Ipulse_ahp: FloatVector,
#     Ipulse: FloatVector,
#     Iref: FloatVector,
#     Ispkthr: FloatVector,
#     Itau_ahp: FloatVector,
#     Itau_mem: FloatVector,
#     Itau_syn: FloatVector,
#     Iw_ahp: FloatVector,
#     *args,
#     **kwargs
# ):

# for key, val in kwargs.items():
