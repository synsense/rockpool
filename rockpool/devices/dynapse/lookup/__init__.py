"""
Lookup tables with hard-coded floating point values obtained from
https://hardware.basket.office.synsense.ai/documentation/dynapse2docs/Sections/input_interface_config.html
especially for more accurate simulation of Dynap-SE2 bias generator.

Additionally, encapsulates the FPGA configuration file

* Non User Facing *
"""

from .parameter_name import *
from .paramgen import *
from .scaling import *
from .defaults import *

## Get FPGA configuration file
import os

__dirname__ = os.path.dirname(os.path.abspath(__file__))
SE2_STACK_FPGA_FILEPATH = os.path.join(__dirname__, "bitfiles", "Dynapse2Stack.bit")
