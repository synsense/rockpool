"""
Package to support the Simulation of the chips from Xylo™ family (Xylo™ core).

This simulates the basic design of Xylo™ and serves as a base to the chips in the same family.

Defines the Rockpool modules :py:class:`.XyloSim`

"""

try:
    from .xylo_sim import *
except:
    if not backend_available("xylosim", "samna"):
        XyloSim = missing_backend_shim("XyloSim", "xylosim, samna")
    else:
        raise
