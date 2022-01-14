"""
DynapSE-family device simulations, deployment and HDK support
"""


from warnings import warn

try:
    from .simconfig import *
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))
