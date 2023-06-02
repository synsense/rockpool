"""
This module implements the rectifier unit in IMU preprocessing.
The rectifier is applied to the IMU data after rotation removal either directly or after it is passed through the IMU filterbank.
"""
import numpy as np
from rockpool.devices.xylo.imu.preprocessing.utils import type_check


class FullWaveRectifier:
    """Full-wave rectification to the bipolar input signals"""

    @type_check
    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        """
        Args:
            sig_in (np.ndarray): input signal.
        """

        return np.abs(sig_in)

    def __call__(self, *args, **kwargs):
        """
        this module is the same as evolve and is implemented for further convenience.
        """
        return self.evolve(*args, **kwargs)


class HalfWaveRectifier:
    """Half-wave rectification to the bipolar input signals"""

    @type_check
    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        """

        Args:
            sig_in (np.ndarray): input signal.
        """

        sig_out = np.copy(sig_in)
        sig_out[sig_out < 0] = 0

        return sig_out
