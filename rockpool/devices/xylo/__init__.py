"""
Xylo-family device simulations, deployment and HDK support

See Also:
    See :ref:`/devices/xylo-overview.ipynb`, :ref:`/devices/torch-training-spiking-for-xylo.ipynb` and :ref:`/devices/analog-frontend-example.ipynb` for documentation of this module.

    Defines the subpackages :py:mod:`.syns61201`, :py:mod:`.syns61300`, :py:mod:`.syns65300`, :py:mod:`.imu`.

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
     - :py:mod:`~rockpool.devices.xylo.syns61201`
     - Xylo Audio v2. Two synapses per neuron. Includes Analog audio front-end core and Xylo SNN core. Includes divisive normalisation simulation modules. Includes AFE simulation modules.
   * - Xylo IMU
     - :py:mod:`~rockpool.devices.xylo.imu`
     - Xylo IMU. One synapse per neuron. 496 hidden neurons, 16 output neurons. Includes direct IMU sensor interface.
"""

from .helper import find_xylo_hdks

import rockpool.devices.xylo.syns61300 as pollen
import rockpool.devices.xylo.syns65300 as vA1
import rockpool.devices.xylo.syns61201 as vA2
