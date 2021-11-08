"""
Xylo-family device simulations and deployment support

See Also:
    :py:class:`XyloCim`, :py:class:`XyloSamna`, :py:class:`AFE`, :py:class:`DivisiveNormalisation`, :py:class:`DivisiveNormalisationNoLFSR`
"""

# - Import submodules to make them available
import warnings

try:
    from .xylo_cimulator import *
except Exception as e:
    warnings.warn(str(e))

try:
    from .xylo_samna import *
except Exception as e:
    warnings.warn(str(e))

try:
    from .analogFrontEnd import *
except Exception as e:
    warnings.warn(str(e))

try:
    from .xylo_divisive_normalisation import *
    from .xylo_divisive_normalisation import (
        DivisiveNormalisation as DivisiveNormalization,
        DivisiveNormalisationNoLFSR as DivisiveNormalizationNoLFSR,
    )
except Exception as e:
    warnings.warn(str(e))
