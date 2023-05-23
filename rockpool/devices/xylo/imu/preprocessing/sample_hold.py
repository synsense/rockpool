"""
This module implement a simple sample-and-hold module.
This is not typically implemented like this in the hardware.
However we can still use this for comparison.
"""
from typing import Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

__all__ = ["SampleAndHold"]


class SampleAndHold(Module):
    """Samples and holds the signal in time dimension (BxTxC)"""

    def __init__(
        self,
        sampling_period: int,
        shape: Optional[Union[Tuple, int]] = (3, 3),
    ) -> None:
        """Object Constructor

        Args:
            sampling_period (int): Sampling period that the signal is sampled and held
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        self.sampling_period = SimulationParameter(sampling_period, shape=(1,))
        """Sampling period that the signal is sampled and held"""

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Operate always along the time axes.

        Args:
            input_data (np.ndarray): input signal of shape BxTxC where T is the time-dimension along which sample-and-hold is done. (BxTxC)
            record (bool, optional): record flag to match with the other rockpool modules. Practically useless. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict, Dict]:
                out_data: the python-object quantized version of the input signal.
                state_dict: empty dictionary.
                record_dict: empty dictionary.
        """
        # BxTxC
        input_data, _ = self._auto_batch(input_data)
        __B, __T, __C = input_data.shape

        # Generate the output data
        out_data = np.zeros_like(input_data)
        num_periods = int(np.ceil(__T / self.sampling_period))

        for period in range(num_periods):
            start_idx = int(period * self.sampling_period)

            end_idx = int((period + 1) * self.sampling_period)
            end_idx = end_idx if end_idx <= __T else __T

            # copy and repeat the signal along the time dimension
            out_data[:, start_idx:end_idx, :] = np.full_like(
                out_data[:, start_idx:end_idx, :], input_data[:, start_idx, :]
            )

        return out_data, {}, {}

    def __str__(self) -> str:
        string = "Sample-and-Hold maodule:\n" + f"period: {self.sampling_period}"
        return string
