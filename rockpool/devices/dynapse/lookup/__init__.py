"""
Lookup tables obtained from 
https://hardware.basket.office.synsense.ai/documentation/dynapse2docs/Sections/input_interface_config.html
for more accurate simulation of DynapSE-2 bias generator
"""

from .parameter_name import *
from .paramgen import *
from .scaling import *
from .defaults import *

## Get FPGA configuration file
import os

__dirname__ = os.path.dirname(os.path.abspath(__file__))
SE2_STACK_FPGA_FILEPATH = os.path.join(__dirname__, "bitfiles", "Dynapse2Stack.bit")
