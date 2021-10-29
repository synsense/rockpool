import warnings


# - Base Module classes
from .module import *
from .timed_module import *

# - Native classes
try:
    from .native.linear import *
    from .native.instant import *
    from .native.filter_bank import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"Torch modules not available: {err}")


# - Torch modules
try:
    from .torch.torch_module import *
    from .torch.lif_torch import *
    from .torch.lif_bitshift_torch import *
    from .torch.lowpass import *
    from .torch.exp_syn_torch import *
    from .torch.lif_neuron_torch import *
    from .torch.linear_torch import *
    from .torch.updown_torch import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"Torch modules not available: {err}")

# - Jax modules
try:
    from .jax.jax_module import *
    from .jax.lif_jax import *
    from .jax.rate_jax import *
    from .jax.exp_smooth_jax import *
    from .jax.softmax_jax import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"Jax modules not available: {err}")

# - NEST modules
try:
    from .nest.iaf_nest import FFIAFNest, RecIAFSpkInNest, RecAEIFSpkInNest
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"NEST modules not available: {err}")
