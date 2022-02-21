"""
Modules using Sinabs and sinabs-slayer as a backend
"""
from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

try:
    from .lif_sinabs import *
except:
    if not backend_available("sinabs"):
        LIFSinabs = missing_backend_shim("LIFSinabs", "sinabs")

try:
    from .lif_slayer import *
except:
    if not backend_available("sinabs-slayer"):
        LIFSlayer = missing_backend_shim("LIFSlayer", "sinabs-slayer")
