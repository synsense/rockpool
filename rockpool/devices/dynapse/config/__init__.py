"""
DynapSE-family simulation configuration package

Supporting operations:

* Creating a custom simulation object
* Importing a simulation configuration from a device configuration
* Exporting to a device configuration from a simulation configuration
"""

from warnings import warn

try:
    from .autoencoder import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .circuits import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .layout import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .simconfig import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .weights import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

