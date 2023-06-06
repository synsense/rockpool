"""
This module implements the spike encoding for the signal coming out of filters or IMU sensor directly.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

NUM_OUT_BITS = 4
"""number of bits devoted to storing the output spike encoding"""

__all__ = ["ScaleSpikeEncoder", "IAFSpikeEncoder"]


class ScaleSpikeEncoder(Module):
    """
    Encode spikes as follows

    (i)     Apply full-wave rectification to the input signal
    (ii)    Down-scale the input signal by right-bit-shift by `num_scale_bit` (e.g. multiplying with 1/2^num_scale_bits)
    (iii)   Truncate the output so that it can fit within `num_out_bits`
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (15, 15),
        num_scale_bits: int = 10,
    ) -> None:
        """
        Object constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to (15, 15).
            num_scale_bits (int): number of right-bit-shifts needed for down-scaling the input signal. Defaults to 10.
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=True)

        if num_scale_bits < 0:
            raise ValueError(
                f"num_scale_bits should be a non-negative integer. Got {num_scale_bits}"
            )
        if num_scale_bits > 15:
            raise ValueError(
                f"num_scale_bits should be less than or equal to 15. Got {num_scale_bits}"
            )

        self.num_scale_bits = SimulationParameter(
            num_scale_bits, shape=(1,), cast_fn=int
        )
        """number of right-bit-shifts needed for down-scaling the input signal"""

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Processes the input signal and return the encoded spikes

        Args:
            input_data (np.ndarray): filtered data recorded from IMU sensor. It should be in integer format. (BxTx48)
            record (bool, optional): If True, the intermediate results are recorded and returned. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                Encoded spikes (BxTx48)
                empty dictionary
                empty dictionary
        """
        input_data, _ = self._auto_batch(input_data)
        input_data = np.array(input_data, dtype=np.int64).astype(object)

        # Full-wave rectification
        output_data = np.abs(input_data)

        # scale the signal
        output_data = output_data >> self.num_scale_bits

        # truncate the signal
        threshold = (1 << NUM_OUT_BITS) - 1
        output_data[output_data > threshold] = threshold

        return output_data, {}, {}


class IAFSpikeEncoder(Module):
    """
    Synchronous integrate and fire spike encoder

    More specifically, denoting a specific filter output by $$y(t)$$,
    this module computes $$s(t) = \\sum_{n=0}^t |y(n)|$$ and as soon as it passes the firing threshold $$\\theta$$.
    Say at a time $$t=t_0$$, it produces a spike at $$t=t_0$$ and reduces the counter by $$\theta$$ and keeps accumulating the input signal
    $$s(t) = s(t_0) - \\theta + \\sum_{n=t_0+1}^t |y(n)|$$ until the next threshold crossing and spike generation happens.
    """

    def __init__(
        self, shape: Optional[Union[Tuple, int]] = (15, 15), threshold: int = 1000
    ) -> None:
        """
        Object constructor

        Args:
            shape (Optional[Union[Tuple, int]], optional): the shape of the input signal. Defaults to (15, 15).
            threshold (int): the threshold of the IAF neuron (quantized). Default to 1000
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=True)

        self.threshold = SimulationParameter(threshold, shape=(1,), cast_fn=int)
        """the threshold of the IAF neuron (quantized)"""

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Processes the input signal and return the encoded spikes

        Args:
            input_data (np.ndarray): filtered data recorded from IMU sensor. It should be in integer format. (BxTx48)
            record (bool, optional): If True, the intermediate results are recorded and returned. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                Encoded spikes (BxTx48)
                empty dictionary
                empty dictionary
        """
        input_data, _ = self._auto_batch(input_data)
        __B, __T, __C = input_data.shape
        input_data = np.array(input_data, dtype=np.int64).astype(object)

        # Full-wave rectification
        output_data = np.abs(input_data)

        # compute the cumsum along the time axis
        output_data = np.cumsum(output_data, axis=1)

        # compute the number of spikes produced so far
        num_spikes = output_data // self.threshold

        # add a zero column to make sure that the dimensions match
        num_spikes = np.hstack([np.zeros((__B, 1, __C), dtype=object), num_spikes])

        # compute the spikes along the time axis
        spikes = np.diff(num_spikes, axis=1)

        # if there are more than one spikes, truncate it to 1
        np.clip(spikes, 0, 1, out=spikes)

        return spikes, {}, {}
