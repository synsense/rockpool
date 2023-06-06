from typing import Any, Optional, Tuple, Union
import samna
import time
import math
import numpy as np


from rockpool.nn.modules.module import Module
from . import xylo_imu_devkit_utils as hdkutils
from .xylo_imu_devkit_utils import XyloIMUHDK, IMUSensorHDK


__all__ = ["XyloIMUData"]


class XyloIMUData(Module):
    """
    Interface to the IMU sensor on a Xylo-Imu HDK

    This module uses ``samna`` to interface to the IMU hardware on a Xylo-IMU HDK. It permits recording from the IMU sensor.

    To record from the module, use the :py:meth:`~.XyloImuData.evolve` method. You need to pass this method an empty matrix, with the desired number of time-steps. The time-step ``dt`` is specified at module instantiation.
    """

    def __init__(
        self,
        device: XyloIMUHDK,
        frequency: float = 200.0,
        *args,
        **kwargs,
    ):
        """
        Instantiate a XyloIMUData Module, via a samna backend

        Args:
            device (XyloIMUHDK): A connected XyloIMUHDK device.
            frequency (float): The frequency to read data from IMU sensor. Default: 200.0
        """

        # - Check device validation
        if device is None:
            raise ValueError("`device` must be a valid, opened Xylo IMU HDK device.")

        # - Store the device
        self._device: XyloIMUHDK = device
        """ `.XyloHDK`: The Xylo HDK used by this module """

        # - Register buffers to read and write events
        self._read_buffer, self._write_buffer, mc = hdkutils.initialise_imu_sensor(
            device
        )

        # - Store the IMU sensor
        self._mc = mc

        # - Store the dt
        self.dt = 1 / frequency

        # - Calculate the time interval and config the IMU sensor to ready for data reading
        ti = int(1 / frequency * 1e8)
        hdkutils.config_imu_sensor(self._mc, ti)

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
        Nt, Nc = input_data.shape

        if Nc != 3:
            raise ValueError(
                f"The specified data should have 3 channels! Recording data with shape [{Nt, Nc}]."
            )

        out = []
        count = 0

        # - Determine a read timeout
        timeout = 2 * Nt * self.dt if timeout is None else timeout
        t_start = time.time()
        t_timeout = t_start + timeout

        # - Clear the read buffer to ensure no previous events influence
        self._read_buffer.get_events()

        while count < int(Nt):
            evts = self._read_buffer.get_events()
            for e in evts:
                if isinstance(e, samna.events.Acceleration):
                    count += 1
                    x = e.x * 4 / math.pow(2, 14)
                    y = e.y * 4 / math.pow(2, 14)
                    z = e.z * 4 / math.pow(2, 14)
                    output = [x, y, z]
                    out.append(output)

                # - Check for read timeout
                if time.time() > t_timeout:
                    raise TimeoutError(f"IMUSensor: Read timeout of {timeout} sec.")

        out = np.array(out).T

        return out, {}, {}
