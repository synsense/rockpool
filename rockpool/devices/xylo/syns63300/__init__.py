"""
Package to support the Xylo HW SYNS63300 (Xylo™IMU)

Defines the Rockpool modules :py:class:`~.syns63300.XyloIMUMonitor`, :py:class:`~.syns63300.XyloSamna`, :py:class:`~.syns63300.XyloIMUMonitor`, :py:class:`~.syns63300.IMUIFSim`, :py:class:`~.syns63300.IMUIFSamna`, :py:class:`~.syns63300.IMUData`.

Defines the mapper function :py:func:`~.syns63300.mapper`.
Defines the configuration function :py:func:`~.syns63300.config_from_specification`.

Defines the subpackage :py:mod:`.xylo_imu_devkit_utils`.
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

from .transform import *
from .xylo_mapper import *
from .xylo_graph_modules import *

try:
    from .xylo_sim import *
except:
    if not backend_available("xylosim", "samna"):
        XyloSim = missing_backend_shim("XyloSim", "xylosim, samna")
    else:
        raise

try:
    from .imu_data import *
    from .xylo_samna import *
    from .xylo_imu_devkit_utils import *
    from .imu_monitor import *
    from .imuif_sim import *
    from .imuif_samna import *
    from .power_cycles_model import cycles_model, est_clock_freq
except:
    if not backend_available("samna"):
        XyloSamna = missing_backend_shim("XyloSamna", "samna")
        XyloIMUMonitor = missing_backend_shim("XyloIMUMonitor", "samna")
        IMUIFSim = missing_backend_shim("IMUIFSim", "samna")
        IMUIFSamna = missing_backend_shim("IMUIFSamna", "samna")
        config_from_specification = missing_backend_shim(
            "config_from_specification", "samna"
        )
        save_config = missing_backend_shim("save_config", "samna")
        load_config = missing_backend_shim("load_config", "samna")
        cycles_model = missing_backend_shim("cycles_model", "samna")
        est_clock_freq = missing_backend_shim("est_clock_freq", "samna")
    else:
        raise
