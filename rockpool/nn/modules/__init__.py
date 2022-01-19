"""
Base package for all Rockpool modules.

Contains :py:class:`.Module` subclasses.

See Also:
    See :ref:`/basics/getting_started.ipynb` for an introductory tutorial.
"""

import warnings

# - Base Module classes
from .module import *
from .timed_module import *

# - Native classes
from .native.linear import *
from .native.instant import *
from .native.filter_bank import *
from .native.exp_syn import *
from .native.rate import *
from .native.lif import *

# - Torch modules
try:
    from .torch.torch_module import *
    from .torch.rate_torch import *
    from .torch.lif_torch import *
    from .torch.lif_bitshift_torch import *
    from .torch.exp_syn_torch import *
    from .torch.lif_neuron_torch import *
    from .torch.linear_torch import *
    from .torch.updown_torch import *

except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"Torch modules not available: {err}")


# - Sinabs modules
try:
    from .sinabs.lif_slayer import *
    from .sinabs.lif_sinabs import *

except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"Sinabs modules not available: {err}")


# - Jax modules
try:
    from .jax.jax_module import *
    from .jax.exp_syn_jax import *
    from .jax.lif_jax import *
    from .jax.rate_jax import *
    from .jax.softmax_jax import *

except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"Jax modules not available: {err}")

# - NEST modules
try:
    from .nest.iaf_nest import FFIAFNest, RecIAFSpkInNest, RecAEIFSpkInNest
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"NEST modules not available: {err}")
