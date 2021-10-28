## __init__.py Smart importer for submodules
import importlib
import warnings
from warnings import warn


try:
    from .module import Module
except:
    warnings.warn('.module modules not available')

try:
    from .torch.torch_module  import TorchModule
    from .torch.lif_torch import LIFTorch
    from .torch.lif_bitshift_torch import LIFBitshiftTorch
    from .torch.lowpass import LowPass
    from .torch.exp_syn_torch import ExpSynTorch
    from .torch.lif_neuron_torch import LIFNeuronTorch
    from .torch.linear_torch import LinearTorch
    from .torch.updown_torch import UpDownTorch
except:
    warnings.warn('Torch modules not available')


try:
    from .jax.jax_module import JaxModule
    from .jax.lif_jax import LIFJax
    from .jax.rate_jax import RateEulerJax
    from .jax.exp_smooth_jax import ExpSmoothJax
    from .jax.softmax_jax import SoftmaxJax, LogSoftmaxJax
    from .native.linear import Linear, LinearJax
    from .native.instant import Instant, InstantJax
except:
    warnings.warn('jax modules not available')

try:
    from .nest.iaf_nest import FFIAFNest, RecIAFSpkInNest, RecAEIFSpkInNest
except:
    warnings.warn('NEST modules not available')

try:
    from .timed_module import TimedModule
    from .timed_module import TimedModuleWrapper
except:
    warnings.warn('.timed_module modules not available')

try:
    from .native.filter_bank import ButterMelFilter
    from .native.filter_bank import ButterFilter
except:
    warnings.warn('.native.filter_bank modules not available')