"""
Device interfacing Utilities
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

if backend_available("samna"):
    from .samna import *
    from .utils import *
else:
    DynapseSamna = missing_backend_shim("DynapseSamna", "samna")
    find_dynapse_boards = missing_backend_shim("find_dynapse_boards", "samna")
