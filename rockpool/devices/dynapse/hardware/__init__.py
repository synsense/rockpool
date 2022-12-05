"""
DynapSE-family hardware configuration package

Supporting operations:

* Creating a custom simulation object
* Importing a simulation configuration from a device configuration
* Exporting to a device configuration from a simulation configuration
"""

from .to_config import config_from_specification
from .interface import DynapseSamna, find_dynapse_boards
from .from_config import dynapsim_from_config
