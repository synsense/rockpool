"""
Defines classes for encapsulating and generating networks of layers
"""

import warnings

try:
    from .net_ads import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"{err}")

try:
    from .net_deneve import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"{err}")

try:
    from .wavesense import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"{err}")
