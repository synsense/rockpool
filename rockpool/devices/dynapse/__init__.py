"""
DynapSE-family device simulations, deployment and HDK support

JAX-backend Dynap-SE1/SE2 simulator support which allows people to 

* Configure their own networks either custom configuration or importing from existing device configuration 
* Run a simulation with any spiking input
* Observe the current changes through time
* Optimize the circuit level bias parameters using standard gradient-based optimization techniques
* Export the simulation configuration in the form of a device configuraiton for deployment

"""
from warnings import warn

# Packages
try:
    from .config import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .infrastructure import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .utils import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

# Modules
try:
    from .adexplif_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .base import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .fpga_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .se1_jax import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

