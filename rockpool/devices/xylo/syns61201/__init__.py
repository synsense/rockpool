"""
Package to support the Xylo HW SYNS61201 (Xylo™Audio 2)

Defines the Rockpool modules :py:class:`.XyloSim`, :py:class:`.AFESim`, :py:class:`.AFESamna`, :py:class:`.DivisiveNormalisation`, :py:class:`.DivisiveNormalisationNoLFSR`.  

Defines the mapper function :py:func:`.mapper`.

Defines the subpackage :py:mod:`.xa2_devkit_utils`.

This package is aliased to :py:mod:`.rockpool.devices.xylo.vA2`
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)


from ..syns65300.afe_sim import *
from .xylo_graph_modules import *

from .afe_sim_empirical import *

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
    from .xa2_devkit_utils import *
    from .xylo_monitor import *
    from .afe_samna import *
    from .power_cycles_model import cycles_model, est_clock_freq
except:
    if not backend_available("samna"):
        AFESamna = missing_backend_shim("AFESamna", "samna")
        XyloSamna = missing_backend_shim("XyloSamna", "samna")
        XyloMonitor = missing_backend_shim("XyloSamna", "samna")
        config_from_specification = missing_backend_shim(
            "config_from_specification", "samna"
        )
        save_config = missing_backend_shim("save_config", "samna")
        load_config = missing_backend_shim("load_config", "samna")
        cycles_model = missing_backend_shim("cycles_model", "samna")
        est_clock_freq = missing_backend_shim("est_clock_freq", "samna")
    else:
        raise

from .xylo_divisive_normalisation import *
