"""
General utilities
"""

try:
    from .property_arrays import *
except (ImportError, ModuleNotFoundError) as err:
    pass

try:
    from .jax_tree_utils import *
except (ImportError, ModuleNotFoundError) as err:
    pass

from .type_handling import *

try:
    from .timedarray_shift import TimedArray
except (ImportError, ModuleNotFoundError) as err:
    pass

from .backend_management import *
