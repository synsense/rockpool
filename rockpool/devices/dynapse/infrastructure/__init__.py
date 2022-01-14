"""
DynapSE-family devices infrastructure simulation package. 

The circuits which does not create the dynamical behaivor of the neurons and synapses but 
the ones that make those circuits operate.

Supporting operations:

* Expressing the routing mechanism bruied in the configuation as a connectivity matrix
* Converting the coarse and fine value biases to actual currents in amperes and vice-versa
* Simulating the analog device mismatch effect on the circuit parameters and on the bias currents
"""

from warnings import warn

try:
    from .biasgen import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .mismatch import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

try:
    from .router import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))

