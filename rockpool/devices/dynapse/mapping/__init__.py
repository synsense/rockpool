""" 
Dynap-SE2 SNN graph mapping package

Supported Operations:
    * Try to extract a computational graph from any (almost, constraints apply) multi-layer LIF (jax or torch) network and convert it to a `DynapSim` computational graph
    * Try to map the computational graph to hardware configuration by allocating hardware resources
"""

from .graph import *
from .mapper import *
