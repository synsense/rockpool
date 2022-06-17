"""
Package to sypport the Xylo HW SYNS61201 (Xylo-A2)

Defines the Rockpool modules :py:class:`.XyloSim`, :py:class:`.AFESim`, :py:class:`.AFESamna`.

Defines the mapper function :py:func:`.mapper`.

Defines the subpackage :py:mod:`.xa2_devkit_utils`.

This package is aliased to :py:mod:`.rockpool.devices.xylo.v2`
"""

from .xylo_sim import *
from .xylo_mapper import *
from ..syns65300.afe_sim import *
from .afe_samna import *
