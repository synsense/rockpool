"""
Package to sypport the Xylo HW SYNS61300 (Xylo-1 core; "Pollen")

Defines the Rockpool modules :py:class:`.XyloSim`, :py:class:`.XyloSamna`.

Defines the mapper function :py:func:`.mapper`.

Defines the subpackage :py:mod:`.xylo_devkit_utils`.

This package is aliased to :py:mod:`.rockpool.devices.xylo.pollen`
"""

from .xylo_mapper import *
from .xylo_sim import *
from ..xylo_samna import *
from .. import xylo_devkit_utils
