"""
Xylo-family device simulations, deployment and HDK support

See Also:
    See :ref:`/devices/xylo-overview.ipynb`, :ref:`/devices/torch-training-spiking-for-xylo.ipynb` and :ref:`/devices/analog-frontend-example.ipynb` for documentation of this module.

    Defines the classes :py:class:`.XyloSim`, :py:class:`.XyloSamna`, :py:class:`.AFESim`, :py:class:`.AFESamna`, :py:class:`.DivisiveNormalisation`.
    
    Defines the subpackages :py:mod:`.syns61201`, :py:mod:`.syns61300`, :py:mod:`.syns65300`.
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)


try:
    from .syns61300.xylo_sim import *
except (ImportError, ModuleNotFoundError) as err:
    if not backend_available("xylosim", "samna"):
        XyloSim = missing_backend_shim("XyloSim", "xylosim, samna")
    else:
        raise

try:
    from .xylo_samna import *
except (ImportError, ModuleNotFoundError) as err:
    if not backend_available("samna"):
        XyloSamna = missing_backend_shim("XyloSamna", "samna")
        config_from_specification = missing_backend_shim(
            "config_from_specification", "samna"
        )
        save_config = missing_backend_shim("save_config", "samna")
        load_config = missing_backend_shim("load_config", "samna")
    else:
        raise

from .syns65300.afe_sim import *

from .v3.xylo_divisive_normalisation import *
from .v3.xylo_divisive_normalisation import (
    DivisiveNormalisation as DivisiveNormalization,
    DivisiveNormalisationNoLFSR as DivisiveNormalizationNoLFSR,
)

from .xylo_graph_modules import *
from .syns61300.xylo_mapper import *

from .syns61201.afe_samna import *

import rockpool.devices.xylo.syns61300 as pollen
import rockpool.devices.xylo.syns65300 as v1
import rockpool.devices.xylo.syns61201 as v2
