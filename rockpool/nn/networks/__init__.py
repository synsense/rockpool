"""
Defines classes for encapsulating and generating networks of layers
"""

import warnings

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

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
except:
    if not backend_available("torch"):
        WaveSenseNet = missing_backend_shim("WaveSenseNet", "torch")
        WaveBlock = missing_backend_shim("WaveBlock", "torch")
    else:
        raise

try:
    from .synnet import *
except (ImportError, ModuleNotFoundError) as err:
    if not backend_available("torch"):
        SynNet = missing_backend_shim("SynNet", "torch")
    else:
        raise
