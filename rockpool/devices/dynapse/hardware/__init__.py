"""
Dynap-SE2 hardware configuration package

Supporting operations:

* Creating a custom simulation object
* Importing a simulation configuration from a device configuration
* Exporting to a device configuration from a simulation configuration
"""

from .config import config_from_specification
from .interface import DynapseSamna, find_dynapse_boards
