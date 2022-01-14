"""
DynapSE-family device simulations, deployment and HDK support
"""


from warnings import warn

try:
    from .adexplif_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .utils import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .dynapse1_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .fpga_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .config import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))
