"""
Implments :py:class:`.IMUData` to read raw IMU data for Xylo IMU
"""

from typing import Optional, Tuple
import samna
import time
import math
import numpy as np


from rockpool.nn.modules.module import Module
from . import xylo_imu_devkit_utils as hdkutils
from .xylo_imu_devkit_utils import XyloIMUHDK


__all__ = ["IMUData"]


class IMUData(Module):
    """
    Interface to the IMU sensor on a Xylo IMU HDK

    This module uses ``samna`` to interface to the IMU hardware on a Xylo IMU HDK. It permits recording from the IMU sensor.

    To record from the module, use the :py:meth:`~.IMUData.evolve` method. You need to pass this method an empty matrix, with the desired number of time-steps. The time-step ``dt`` is specified at module instantiation.

    .. Warning::

        :py:class:`.IMUData` needs the FPGA to have access to the MC3632 IMU sensor on the Xylo HDK to operate correctly.
        If the MC3632 sensor is already connected directly to Xylo, for example if :py:class:`.XyloIMUMonitor` or :py:class:`.IMUIFSamna` are being used, then initialising the :py:class:`.IMUData` module will fail with an error.
        You must disconnect the IMU sensor from Xylo, or reset the HDK, to proceed.
        You can delete the already-connected module to reset the HDK.
    """

    def __init__(
        self,
        device: XyloIMUHDK,
        sample_frequency: int = 200,
        *args,
        **kwargs,
    ):
        """
        Instantiate an IMUData Module, via a samna backend

        Args:
            device (XyloIMUHDK): A connected XyloIMUHDK device.
            frequency (int): The frequency to read data from IMU sensor, in Hz. Default: `200`
        """
        super().__init__(shape=(0, 3), spiking_input=False, spiking_output=False)

        # - Check device validation
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo IMU HDK device.")

        # - Store the device
        self._device: XyloIMUHDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Store the dt
        sample_frequency = int(sample_frequency)
        self.dt = 1.0 / sample_frequency

        # - Register buffers to read and write events
        # - Set the frequency and config the IMU sensor to ready for data reading
        (
            self._read_buffer,
            self._write_buffer,
            self._accel_buffer,
            self._mc,
            self._accel_graph,
        ) = hdkutils.initialise_imu_sensor(device, sample_frequency)

    def evolve(
        self,
        input_data,
        timeout: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Use the IMU sensor to record live IMU data and return

        Args:
            input_data (np.ndarray): An array ``[T, 3]``, specifying the number of time-steps to record.

        Returns:
            (np.ndarray, dict, dict) output_events, {}, {}
        """
        # - Get the shape of the output data
        data, _ = self._auto_batch(input_data)
        Nb, Nt, Nc = data.shape

        # - Check batch size
        if Nb > 1:
            raise ValueError(
                f"Batched data are not supported by IMUData. Got batched input data with shape {[Nb, Nt, Nc]}."
            )

        # - Determine a read timeout
        timeout = 2 * Nt * self.dt if timeout is None else timeout

        # - Clear the read buffer to ensure no previous events influence
        self._accel_buffer.get_events()

        # - Read Nt Acceleration events
        events = self._accel_buffer.get_n_events(Nt, int(timeout * 1000))
        if len(events) < Nt:
            raise TimeoutError(
                f"IMUSensor: Read timeout of {timeout} sec. Expected {Nt} events, read {len(events)}."
            )

        # - Decode acceleration events
        out = []
        for e in events:
            x = e.x * 4 / math.pow(2, 14)
            y = e.y * 4 / math.pow(2, 14)
            z = e.z * 4 / math.pow(2, 14)
            out.append([x, y, z])

        return np.array(out), {}, {}
