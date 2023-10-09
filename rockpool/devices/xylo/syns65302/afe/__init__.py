"""
Implements a simulation module for the digital audio front-end on Xylo-Audio 3

Provides the modules :py:class:`.Raster`, :py:class:`.MicrophonePDM`, :py:class:`.PolyPhaseFIR`, :py:class:.`PDM_ADC`, :py:class:`.DivisiveNormalization`, :py:class:`.ChipButterworth`.
"""

from .pdm import PDM_ADC, MicrophonePDM, PolyPhaseFIR, DeltaSigma
from .digital_filterbank import ChipButterworth
from .raster import Raster
from .divisive_normalization import DivisiveNormalization
from .params import *
