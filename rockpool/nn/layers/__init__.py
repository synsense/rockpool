"""
Package of converted layers from Rockpool v1
"""

# - Layer base class
from .layer import *

# - Brian modules
try:
    from .iaf_brian import *
    from .exp_synapses_brian import *
except (ImportError, ModuleNotFoundError) as err:
    print(f"Could not import package: {err}")

# - Native (numpy / scipy) modules
try:
    from .rate import *
    from .exp_synapses_manual import *
    from .iaf_cl import *
    from .iaf_digital import *
    from .updown import *
    from filter_bank import *
except (ImportError, ModuleNotFoundError) as err:
    print(f"Could not import package: {err}")

# - Numba modules
try:
    from .spike_bt import *
    from .spike_ads import *
except (ImportError, ModuleNotFoundError) as err:
    print(f"Could not import package: {err}")

# - NEST modules
try:
    from iaf_nest import *
    from aeif_nest import *
except (ImportError, ModuleNotFoundError) as err:
    print(f"Could not import package: {err}")
