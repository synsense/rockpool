"""
Modules using Sinabs and sinabs-exodus as a backend
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
    from .lif_exodus import *
except:
    if not backend_available("sinabs-exodus"):
        LIFSlayer = missing_backend_shim("LIFSlayer", "sinabs-exodus")
        LIFExodus = missing_backend_shim("LIFExodus", "sinabs-exodus")
        ExpSynExodus = missing_backend_shim("ExpSynExodus", "sinabs-exodus")
        LIFMembraneExodus = missing_backend_shim("LIFMembraneExodus", "sinabs-exodus")
