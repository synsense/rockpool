"""
Xylo-family device simulations, deployment and HDK support

See Also:
    See :ref:`/devices/xylo-overview.ipynb`, :ref:`/devices/torch-training-spiking-for-xylo.ipynb` and :ref:`/devices/analog-frontend-example.ipynb` for documentation of this module.

    Defines the classes :py:class:`XyloSim`, :py:class:`XyloSamna`, :py:class:`AFE`, :py:class:`DivisiveNormalisation`, :py:class:`DivisiveNormalisationNoLFSR`.
"""

# - Import submodules to make them available
import warnings


try:
    from .xylo_sim import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"{err}")

try:
    from .xylo_samna import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"{err}")

try:
    from .analogFrontEnd import *
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"{err}")

try:
    from .xylo_divisive_normalisation import *
    from .xylo_divisive_normalisation import (
        DivisiveNormalisation as DivisiveNormalization,
        DivisiveNormalisationNoLFSR as DivisiveNormalizationNoLFSR,
    )
except Exception as inst:
    warnings.warn(inst.msg)

try:
    from .xylo_graph_modules import *
except Exception as inst:
    warnings.warn(inst.msg)

try:
    from .xylo_mapper import *
except Exception as inst:
    warnings.warn(inst.msg)
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"{err}")
