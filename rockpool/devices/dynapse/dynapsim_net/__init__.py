"""
DynapSim Network getters. Either process the configuration object or a specification dictionary to restore a network.
The resulting network has two layers by default. 
The first layer `nn.modules.LinearJax` contains the input weights (optional, only if input weights present)
The second layer `devices.dynapse.DynapSim` combines all the other layers.
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

if backend_available("jax"):
    from .from_config import dynapsim_net_from_config
    from .from_spec import dynapsim_net_from_spec
else:
    dynapsim_net_from_config = missing_backend_shim("dynapsim_net_from_config", "jax")
    dynapsim_net_from_spec = missing_backend_shim("dynapsim_net_from_spec", "jax")
