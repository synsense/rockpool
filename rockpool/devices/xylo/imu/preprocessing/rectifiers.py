"""
This module implements the rectifier unit in IMU preprocessing.
The rectifier is applied to the IMU data after rotation removal either directly or after it is passed through the IMU filterbank.
"""
from typing import Optional, Tuple, Union

import numpy as np

from rockpool.nn.modules import Instant

__all__ = ["FullWaveRectifier", "HalfWaveRectifier"]


class FullWaveRectifier(Instant):
    """Full-wave rectification to the bipolar input signals"""

    def __init__(self, shape: Optional[Union[Tuple, int]] = (48, 48)) -> None:
        super().__init__(
            shape=shape, spiking_input=False, spiking_output=False, function=np.abs
        )


class HalfWaveRectifier(Instant):
    """Full-wave rectification to the bipolar input signals"""

    def __init__(self, shape: Optional[Union[Tuple, int]] = (48, 48)) -> None:
        super().__init__(
            shape=shape,
            spiking_input=False,
            spiking_output=False,
            function=lambda x: np.maximum(x, 0),
        )
