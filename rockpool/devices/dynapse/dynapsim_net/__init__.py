"""
DynapSim Network getters. Either process the configuration object or a specification dictionary to restore a network.
The resulting network has two layers by default. 
The first layer `nn.modules.LinearJax` contains the input weights (optional, only if input weights present)
The second layer `devices.dynapse.DynapSim` combines all the other layers.
"""
from .from_config import dynapsim_net_from_config
from .from_spec import dynapsim_net_from_spec
