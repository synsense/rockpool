"""
Modules for pre-processing IMU data, as implemented in Xylo IMU

Defines the modules :py:class:`.IMUIFSim`, :py:class:`.JVSD`, :py:class:`.RotationLookUpTable`, :py:class:`.Quantizer`, :py:class:`.RotationRemoval`, :py:class:`.SampleAndHold`, :py:class:`.SubSpace`.
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

from .filterbank import *
from .jsvd import *
from .lookup import *
from .quantizer import *
from .rectifiers import *
from .rotation_removal import *
from .sample_hold import *
from .spike_encoder import *
from .subspace import *

if backend_available("samna"):
    from .imuif_sim import *
else:
    IMUIFSim = missing_backend_shim("IMUIFSim", "samna")
