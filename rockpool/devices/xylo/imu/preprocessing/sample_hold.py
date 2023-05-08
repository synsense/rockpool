"""
This module implement a simple sample-and-hold module.
This is not typically implemented like this in the hardware.
However we can still use this for comparison.
"""
from typing import Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.modules.module import Module

__all__ = ["SampleAndHold"]


class SampleAndHold(Module):
    """Samples and holds the signal in its last index (supposed to be the time-index)

    Parameters:
        sampling_period (int): sampling period.
    """

    def __init__(
        self,
        sampling_period: int,
        shape: Optional[Union[Tuple, int]] = None,
    ) -> None:
        super().__init__(shape, spiking_input=False, spiking_output=False)
        self.sampling_period = sampling_period

    @type_check
    def evolve(
        self, sig_in: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Assume that the last index is the time-dimension, so the sample-and-hold is done always along the last axes.
        Use swapaxes to change the axes if needed before calling the function.

        Args:
            sig_in (np.ndarray): input signal of shape (*,*,...,*, T) where T is the time-dimension along which sample-and-hold is done.
            record (bool, optional): record flag to match with the other rockpool modules. Practically useless. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict, Dict]:
                data: the python-object quantized version of the input signal.
                state_dict: empty dictionary.
                record_dict: empty dictionary.
        """
        # sample and hold the last component
        sig_in_shape = sig_in.shape

        # additional care with 1-dim array
        if len(sig_in_shape) == 1:
            raise ValueError("The input signal should be at least 2-dim!")

        sig_out = np.zeros_like(sig_in)

        num_periods = int(np.ceil(sig_in.shape[-1] / self.sampling_period))

        for period in range(num_periods):
            start_idx = period * self.sampling_period

            end_idx = (period + 1) * self.sampling_period
            end_idx = end_idx if end_idx <= sig_in.shape[-1] else sig_in.shape[-1]

            # copy and repeat the signal along the last dimension
            sig_out[..., start_idx:end_idx] = np.expand_dims(sig_in[..., start_idx], -1)

        # return to the original shape
        sig_out = sig_out.reshape(sig_in_shape)

        return sig_out, {}, {}

    def __str__(self) -> str:
        string = "Sample-and-Hold maodule:\n" + f"period: {self.sampling_period}"
        return string
