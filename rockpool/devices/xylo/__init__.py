"""
Xylo-family device simulations, deployment and HDK support

See Also:
    See :ref:`/devices/xylo-overview.ipynb`, :ref:`/devices/torch-training-spiking-for-xylo.ipynb` and :ref:`/devices/analog-frontend-example.ipynb` for documentation of this module.

    Defines the subpackages :py:mod:`.syns61201`, :py:mod:`.syns61300`, :py:mod:`.syns65300`.

    Use the function :py:func:`.find_xylo_hdks` to connect and identify an HDK, detect the correct version and import the required module for support.

    If you don't have a Xylo HDK, use the module :py:mod:`.syns61300`.

.. list-table:: Xylo versions and support modules
   :widths: 25 25 25
   :header-rows: 1

   * - Chip version
     - Support module
     - Description
   * - SYNS61300
     - :py:mod:`~rockpool.devices.xylo.syns61300`
     - Xylo SNN core only. Two synapses per neuron.
   * - SYNS65300
     - :py:mod:`~rockpool.devices.xylo.syns65300`
     - Xylo Audio v1. Two synapses per neuron. Includes Analog audio front-end core and Xylo SNN core. Includes AFE simulation modules.
   * - SYNS61201
     - :py:mod:`~rockpool.devices.xylo.61201`
     - Xylo Audio v2. Two synapses per neuron. Includes Analog audio front-end core and Xylo SNN core. Includes divisive normalisation simulation modules. Includes AFE simulation modules.
"""

from .helper import find_xylo_hdks

# from rockpool.utilities.backend_management import (
#     backend_available,
#     missing_backend_shim,
# )


# try:
#     from .syns61300.xylo_sim import *
# except (ImportError, ModuleNotFoundError) as err:
#     if not backend_available("xylosim", "samna"):
#         XyloSim = missing_backend_shim("XyloSim", "xylosim, samna")
#     else:
#         raise

# try:
#     from .syns61300.xylo_samna import *
# except (ImportError, ModuleNotFoundError) as err:
#     if not backend_available("samna"):
#         XyloSamna = missing_backend_shim("XyloSamna", "samna")
#         config_from_specification = missing_backend_shim(
#             "config_from_specification", "samna"
#         )
#         save_config = missing_backend_shim("save_config", "samna")
#         load_config = missing_backend_shim("load_config", "samna")
#     else:
#         raise

# from .syns65300.afe_sim import *

# from .syns61201.afe_samna import *
# from .syns61201.xylo_divisive_normalisation import *

# from .syns61300.xylo_graph_modules import *
# from .syns61300.xylo_mapper import *


import rockpool.devices.xylo.syns61300 as pollen
import rockpool.devices.xylo.syns65300 as vA1
import rockpool.devices.xylo.syns61201 as vA2
