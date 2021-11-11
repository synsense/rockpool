"""
Package for simulating and interacting with Dynap™SE hardware
"""

from warnings import warn

try:
    from .virtual_dynapse import VirtualDynapse
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))
