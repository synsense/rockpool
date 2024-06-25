"""
IMU-IF submodules, as implemented in Xylo IMU

Implements the modules :py:class:`~.filterbank.FilterBank`, :py:class:`~.filterbank.BandPassFilter`, :py:class:`.RotationRemoval`, :py:class:`.ScaleSpikeEncoder`, :py:class:`.IAFSpikeEncoder`

"""

from .filterbank import *
from .rotation_removal import *
from .spike_encoder import *
