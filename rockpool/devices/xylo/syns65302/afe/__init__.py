"""
Implements a simulation module for the digital audio front-end on Xylo-Audio 3

Provides the modules :py:class:`.Raster`, :py:class:`.MicrophonePDM`, :py:class:`.PolyPhaseFIR`, :py:class:.`PDMADC`, :py:class:`.DivisiveNormalization`, :py:class:`.ChipButterworth`.
"""

from .pdm import PDMADC
from .agc import AGCADC
from .digital_filterbank import ChipButterworth
from .raster import Raster
from .divisive_normalization import DivisiveNormalization
from .params import *
from .external import *
