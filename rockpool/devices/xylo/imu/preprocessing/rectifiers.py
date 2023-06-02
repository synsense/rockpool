# -----------------------------------------------------------
# This module implements the rectifier unit in IMU preprocessing.
# The rectifier is applied to the IMU data after rotation removal either directly or after
# it is passed through the IMU filterbank.
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 31.08.2022
# -----------------------------------------------------------
import numpy as np
from imu_preprocessing.util.type_decorator import type_check


class FullWaveRectifier:
    @type_check
    def evolve(self, sig_in: np.ndarray):
        """This modules applies full-wave rectification to the input bipolar signal.

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
    @type_check
    def evolve(self, sig_in: np.ndarray):
        """This modules applies half-wave rectification to the input bipolar signal.

        Args:
            sig_in (np.ndarray): input signal.
        """

        sig_out = np.copy(sig_in)
        sig_out[sig_out < 0] = 0

        return sig_out

    def __call__(self, *args, **kwargs):
        """
        this module is the same as evolve and is implemented for further convenience.
        """
        return self.evolve(*args, **kwargs)
