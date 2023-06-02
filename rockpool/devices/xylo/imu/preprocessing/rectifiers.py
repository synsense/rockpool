"""
This module implements the rectifier unit in IMU preprocessing.
The rectifier is applied to the IMU data after rotation removal either directly or after it is passed through the IMU filterbank.
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.modules.module import Module


class FullWaveRectifier(Module):
    """Full-wave rectification to the bipolar input signals"""

    def __init__(self, shape: Optional[Union[Tuple, int]] = (48, 48)) -> None:
        """Object constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to (48, 48).
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Recitfy the input signal

        Args:
            input_data (np.ndarray): batched input data recorded from IMU sensor. It should be in integer format. (BxTx3)
            record (bool, optional): If True, the intermediate results are recorded and returned. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                the covariance matrix of the input data (BxTx3x3)
                empty dictionary
                empty dictionary
        """
        return np.abs(input_data), {}, {}


class HalfWaveRectifier(Module):
    """Half-wave rectification to the bipolar input signals"""

    def __init__(self, shape: Optional[Union[Tuple, int]] = (48, 48)) -> None:
        """Object constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to (48, 48).
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Recitfy the input signal

        Args:
            input_data (np.ndarray): batched input data recorded from IMU sensor. It should be in integer format. (BxTx3)
            record (bool, optional): If True, the intermediate results are recorded and returned. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                the covariance matrix of the input data (BxTx3x3)
                empty dictionary
                empty dictionary
        """

        sig_out = np.copy(input_data)
        sig_out[input_data < 0] = 0

        return sig_out, {}, {}
