"""
Package to support the Xylo HW SYNS61300 (Xylo™ core; "Pollen")

Defines the Rockpool modules :py:class:`.XyloSim`, :py:class:`.XyloSamna`.

Defines the mapper function :py:func:`.mapper`.

Defines the subpackage :py:mod:`.xylo_devkit_utils`.

This package is aliased to :py:mod:`.rockpool.devices.xylo.pollen`
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

from .xylo_graph_modules import *

try:
    from .xylo_sim import *
except:
    if not backend_available("xylosim", "samna"):
        XyloSim = missing_backend_shim("XyloSim", "xylosim, samna")
    else:
        raise

try:
    from .xylo_mapper import *
except:
    if not backend_available("xylosim", "samna"):
        XyloSim = missing_backend_shim("mapper", "xylosim, samna")
    else:
        raise

try:
    from .xylo_samna import *
    from .xylo_devkit_utils import *
except:
    if not backend_available("samna"):
        XyloSamna = missing_backend_shim("XyloSamna", "samna")
        config_from_specification = missing_backend_shim(
            "config_from_specification", "samna"
        )
        save_config = missing_backend_shim("save_config", "samna")
        load_config = missing_backend_shim("load_config", "samna")
    else:
        raise
