"""
DynapSE-family device simulations, deployment and HDK support
"""


from warnings import warn

try:
    from .virtual_dynapse import VirtualDynapse
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))
