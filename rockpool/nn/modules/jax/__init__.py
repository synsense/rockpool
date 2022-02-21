"""Modules using Jax as a backend"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

try:
    from .jax_module import *
    from .exp_syn_jax import *
    from .lif_jax import *
    from .rate_jax import *
    from .softmax_jax import *
    from .jax_lif_ode import *
    from .linear_jax import *
except:
    if not backend_available("jax"):
        JaxModule = missing_backend_shim("JaxModule", "jax")
        RateJax = missing_backend_shim("RateJax", "jax")
        ExpSynJax = missing_backend_shim("ExpSynJax", "jax")
        LIFJax = missing_backend_shim("LIFJax", "jax")
        SoftmaxJax = missing_backend_shim("SoftmaxJax", "jax")
        LogSoftmaxJax = missing_backend_shim("LogSoftmaxJax", "jax")
        LinearJax = missing_backend_shim("LinearJax", "jax")
