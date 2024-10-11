"""
Xylo-family device simulations, deployment and HDK support

See Also:
    See :ref:`/devices/xylo-overview.ipynb`, :ref:`/devices/torch-training-spiking-for-xylo.ipynb` and :ref:`/devices/analog-frontend-example.ipynb` for documentation of this module.

    Defines the subpackages :py:mod:`.syns61201`, :py:mod:`.syns61300`, :py:mod:`.syns65300`, :py:mod:`.syns63300`, :py:mod:`.syns65302`

    Use the function :py:func:`~.devices.xylo.find_xylo_hdks` to connect and identify an HDK, detect the correct version and import the required module for support.

    If you don't have a Xylo HDK, use the module :py:mod:`.syns61300` to access Xylo support.

.. list-table:: Xylo versions and support modules
   :widths: 25 25 25
   :header-rows: 1

   * - Chip version
     - Support module
     - Description
   * - SYNS61300
     - :py:mod:`~rockpool.devices.xylo.syns61300`
     - Xylo™ SNN core only. Two synapses per neuron.
   * - SYNS65300
     - :py:mod:`~rockpool.devices.xylo.syns65300`
     - Xylo™Audio 1. Two synapses per neuron. Includes Analog audio front-end core and Xylo SNN core. Includes AFE simulation modules.
   * - SYNS61201
     - :py:mod:`~rockpool.devices.xylo.syns61201`
     - Xylo™Audio 2. Two synapses per neuron. Includes Analog audio front-end core and Xylo SNN core. Includes divisive normalisation simulation modules. Includes AFE simulation modules.
   * - SYNS65302
     - :py:mod:`~rockpool.devices.xylo.syns65302`
     - Xylo™Audio 3. Two synapses per neuron. 992 hidden neurons, 32 output neurons. Includes AFE with PDM microphone input.
   * - SYNS63300
     - :py:mod:`~rockpool.devices.xylo.syns63300`
     - Xylo™IMU. One synapse per neuron. 496 hidden neurons, 16 output neurons. Includes direct IMU sensor interface, and IMU IF simulation.
"""

from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

import rockpool.devices.xylo.syns61300 as pollen
import rockpool.devices.xylo.syns65300 as vA1
import rockpool.devices.xylo.syns61201 as vA2
import rockpool.devices.xylo.syns65302 as vA3
import rockpool.devices.xylo.syns63300 as imu

try:
    from .helper import *
except:
    if not backend_available("samna"):
        find_xylo_hdks = missing_backend_shim("find_xylo_hdks", "samna")
        check_firmware_versions = missing_backend_shim(
            "check_firmware_versions", "samna"
        )
    else:
        raise
