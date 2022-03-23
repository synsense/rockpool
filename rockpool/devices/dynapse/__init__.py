"""
DynapSE-family device simulations, deployment and HDK support
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

try:
    from .virtual_dynapse import *
except:
    if not backend_available("numpy", "nest"):
        VirtualDynapse = missing_backend_shim("VirtualDynapse", "numpy, nest")
