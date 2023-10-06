"""
Implements a simulation module for the digital audio front-end on Xylo-Audio 3

Provides the modules :py:class:`.Raster`, :py:class:`.PDM_Microphone`, :py:class:`.PolyPhaseFIR_DecimationFilter`, :py:class:.`PDM_ADC`, :py:class:`.DivisiveNormalization`, :py:class:`.ChipButterworth`.
"""

from .pdm import PDM_ADC, PDM_Microphone, PolyPhaseFIR_DecimationFilter, DeltaSigma
from .digital_filterbank import ChipButterworth
from .raster import Raster
from .divisive_normalization import DivisiveNormalization
