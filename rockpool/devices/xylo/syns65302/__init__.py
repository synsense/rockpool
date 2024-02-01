"""
Package to support the Xylo HW SYNS65302 (Xylo™ Audio 3)

Includes simulation, interfacing and deployment modules.

Provides the modules :py:class:`.AFESimExternal`, :py:class:`.AFESimAGC`, and :py:class:`.AFESimPDM`.
"""
from .afe.params import *
from .afe_sim import *
from .xylo_samna import *
from .xylo_mapper import *

from ..syns63300 import XyloSim
