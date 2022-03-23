"""Modules using numpy / scipy as a backend"""

try:
    from .exp_syn import *
    from .rate import *
    from .filter_bank import *
    from .instant import *
    from .lif import *
    from .linear import *

except:
    from rockpool.utilities.backend_management import (
        backend_available,
        missing_backend_shim,
    )

    if not backend_available("numpy", "scipy"):
        ExpSyn = missing_backend_shim("ExpSyn", "numpy, scipy")
        Rate = missing_backend_shim("Rate", "numpy, scipy")
        LIF = missing_backend_shim("LIF", "numpy, scipy")
        Instant = missing_backend_shim("Instant", "numpy, scipy")
        Linear = missing_backend_shim("Linear", "numpy, scipy")
        ButterFilter = missing_backend_shim("ButterFilter", "numpy, scipy")
        ButterMelFilter = missing_backend_shim("ButterMelFilter", "numpy, scipy")
