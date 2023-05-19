"""
Package to support the Xylo IMU

Defines the Rockpool modules :py:class:`.XyloIMUMonitor`, :py:class:`.XyloSamna`,

Defines the mapper function :py:func:`.mapper`.

Defines the subpackage :py:mod:`.xylo_imu_devkit_utils`.
"""


from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

from .xylo_mapper import *
from .xylo_graph_modules import *

try:
    from .xylo_samna import *
    from .xylo_imu_devkit_utils import *
    from .imu_monitor import *
except:
    if not backend_available("samna"):
        XyloSamna = missing_backend_shim("XyloSamna", "samna")
        XyloIMUMonitor = missing_backend_shim("XyloIMUMonitor", "samna")
        config_from_specification = missing_backend_shim(
            "config_from_specification", "samna"
        )
        if_config_from_specification = missing_backend_shim(
            "if_config_from_specification", "samna"
        )
        save_config = missing_backend_shim("save_config", "samna")
        load_config = missing_backend_shim("load_config", "samna")
    else:
        raise
