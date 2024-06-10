"""
This module implement a simple sample-and-hold module.
This is not typically implemented like this in the hardware.
However we can still use this for comparison.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.syns63300.imuif.utils import type_check
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

__all__ = ["SampleAndHold"]


class SampleAndHold(Module):
    """
    Samples and holds a signal in the time dimension (BxTxC)
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (3, 3),
        sampling_period: int = 10,
    ) -> None:
        """Object Constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to ``(3, 3)``.
            sampling_period (int): Sampling period that the signal is sampled and held. Defaults to ``10``.
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        self.sampling_period = SimulationParameter(
            sampling_period, shape=(1,), cast_fn=int
        )
        """(int) Sampling period that the signal is sampled and held"""

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Operate always along the time axes

        Args:
            input_data (np.ndarray): input signal of shape ``BxTxC`` where T is the time-dimension along which sample-and-hold is done. ``(B, T, C)``
            record (bool, optional): Unused.

        Returns:
            Tuple[np.ndarray, Dict, Dict]:
                out_data: the python-object quantized version of the input signal.
                state_dict: empty dictionary.
                record_dict: empty dictionary.
        """
        # BxTxC
        input_data, _ = self._auto_batch(input_data)
        input_data = np.array(input_data, dtype=np.int64)
        __B, __T, __C = input_data.shape

        # Generate the output data
        out_data = np.zeros_like(input_data)
        num_periods = int(np.ceil(__T / self.sampling_period))

        for period in range(num_periods):
            start_idx = period * self.sampling_period

            end_idx = (period + 1) * self.sampling_period
            end_idx = end_idx if end_idx <= __T else __T

            # copy and repeat the signal along the time dimension
            out_data[:, start_idx:end_idx, :] = np.repeat(
                input_data[:, start_idx, np.newaxis, :], end_idx - start_idx, axis=1
            )

        out_data = np.array(out_data, dtype=object)
        return out_data, {}, {}
