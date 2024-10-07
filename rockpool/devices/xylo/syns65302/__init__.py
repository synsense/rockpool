"""
Package to support the Xylo HW SYNS65302 (Xylo™Audio 3)

Includes simulation, interfacing and deployment modules.

Provides the modules :py:class:`.AFESimExternal`, :py:class:`.AFESimPDM`, :py:class:`~.syns65302.XyloSim`, :py:class:`~.syns65302.XyloSamna`, :py:class:`~.syns65302.XyloMonitor`, and the functions :py:func:`~.syns65302.save_config`, :py:func:`~.syns65302.load_config`, :py:func:`~.syns65302.mapper`, :py:func:`~.syns65302.config_from_specification`.
"""

from .afe.params import *
from .afe_sim import *
from .xylo_mapper import *
from .xylo_samna_pdm import *
from .xylo_graph_modules import *
from ..syns63300 import XyloSim
from ..syns63300.power_cycles_model import *

try:
    from .xylo_samna import *
    from .xa3_devkit_utils import *
    from .xylo_monitor import *
except:
    if not backend_available("samna"):
        AFESamna = missing_backend_shim("AFESamna", "samna")
        XyloSamna = missing_backend_shim("XyloSamna", "samna")
        XyloMonitor = missing_backend_shim("XyloMonitor", "samna")
        config_from_specification = missing_backend_shim(
            "config_from_specification", "samna"
        )
        save_config = missing_backend_shim("save_config", "samna")
        load_config = missing_backend_shim("load_config", "samna")
    else:
        raise
