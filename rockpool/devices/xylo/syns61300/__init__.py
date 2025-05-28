"""
Package to support the Simulation of Xylo HW SYNS61300 (Xylo™ core; "Pollen")

Defines the Rockpool modules :py:class:`.XyloSim`

This package is aliased to :py:mod:`.rockpool.devices.xylo.pollen`
"""

try:
    from .xylo_sim import *
except:
    if not backend_available("xylosim", "samna"):
        XyloSim = missing_backend_shim("XyloSim", "xylosim, samna")
    else:
        raise
