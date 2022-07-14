"""
Package to sypport the Xylo HW SYNS61300 (Xylo-1 core; "Pollen")

Defines the Rockpool modules :py:class:`.XyloSim`, :py:class:`.XyloSamna`.

Defines the mapper function :py:func:`.mapper`.

Defines the subpackage :py:mod:`.xylo_devkit_utils`.

This package is aliased to :py:mod:`.rockpool.devices.xylo.pollen`
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

try:
    from .xylo_sim import *
except (ImportError, ModuleNotFoundError) as err:
    if not backend_available("xylosim", "samna"):
        XyloSim = missing_backend_shim("XyloSim", "xylosim, samna")
    else:
        raise

from .xylo_mapper import *

try:
    from ..xylo_samna import *
except (ImportError, ModuleNotFoundError) as err:
    if not backend_available("samna"):
        XyloSamna = missing_backend_shim("XyloSamna", "samna")
    else:
        raise


from .. import xylo_devkit_utils
