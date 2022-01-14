"""
DynapSE-family device simulation supporting utilities

Supporting operations:

* Investigating the imported simulation configuration
* Producing informative figures using the simulation module outputs
* Creating random or custom defined spike trains
"""

from warnings import warn

try:
    from .comparison import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .figure import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .spike_input import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))
