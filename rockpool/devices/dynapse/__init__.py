"""
DynapSE-family device simulations, deployment and HDK support
"""


from warnings import warn

try:
    from .adexplif_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .biasgen import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .dpi_gui import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .dynapse1_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .figure import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .fpga_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .router import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .simconfig import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .utils import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))
