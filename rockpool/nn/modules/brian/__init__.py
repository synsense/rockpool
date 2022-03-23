"""Modules using Brian2 as a backend"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

if not backend_available("brian"):
    LIFBrian = missing_backend_shim("LIFBrian", "brian")
else:
    from ...layers.iaf_brian import FFIAFSpkInBrian as LIFBrian
