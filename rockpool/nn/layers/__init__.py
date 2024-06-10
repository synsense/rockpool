"""
Package of converted layers from Rockpool v1
"""

# - Layer base class
from .layer import *

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

# - Brian modules
try:
    from .iaf_brian import *
    from .exp_synapses_brian import *

    import warnings

    warnings.warn(
        "Deprecation: Brian-backed layers are deprecated, and will be removed in future versions.",
        DeprecationWarning,
    )

except:
    if not backend_available("brian"):
        FFIAFBrian = missing_backend_shim("FFIAFBrian", "brian")
        RecIAFBrian = missing_backend_shim("RecIAFBrian", "brian")
        FFIAFBrianBase = missing_backend_shim("FFIAFBrianBase", "brian")
        FFIAFSpkInBrian = missing_backend_shim("FFIAFSpkInBrian", "brian")
        RecIAFBrianBase = missing_backend_shim("RecIAFBrianBase", "brian")
        RecIAFSpkInBrian = missing_backend_shim("RecIAFSpkInBrian", "brian")
        FFExpSynBrian = missing_backend_shim("FFExpSynBrian", "brian")
    else:
        raise

# - Native (numpy / scipy) modules
from .updown import *
from .event_pass import *

# - Numba modules
try:
    from .spike_bt import *
    from .spike_ads import *
except:
    if not backend_available("numba"):
        RecFSSpikeEulerBT = missing_backend_shim("RecFSSpikeEulerBT", "numba")
        RecFSSpikeADS = missing_backend_shim("RecFSSpikeADS", "numba")
    else:
        raise
