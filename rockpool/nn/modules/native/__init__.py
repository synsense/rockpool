"""Modules using numpy or numba as a backend"""

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

    if not backend_available("numpy", "numba", "scipy"):
        ExpSyn = missing_backend_shim("ExpSyn", "numpy, numba, scipy")
        Rate = missing_backend_shim("Rate", "numpy, numba, scipy")
        LIF = missing_backend_shim("LIF", "numpy, numba, scipy")
        Instant = missing_backend_shim("Instant", "numpy, numba, scipy")
        Linear = missing_backend_shim("Linear", "numpy, numba, scipy")
        ButterFilter = missing_backend_shim("ButterFilter", "numpy, numba, scipy")
        ButterMelFilter = missing_backend_shim("ButterMelFilter", "numpy, numba, scipy")
