"""
This module implements the spike encoding for the signal coming out of filters or IMU sensor directly.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.syns63300.imuif.utils import (
    type_check,
    unsigned_bit_range_check,
)
from rockpool.devices.xylo.syns63300.imuif.params import NUM_BITS_SPIKE
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter


__all__ = ["ScaleSpikeEncoder", "IAFSpikeEncoder"]


class ScaleSpikeEncoder(Module):
    """
    Encode spikes as follows

    (i)     Apply full-wave rectification to the input signal
    (ii)    Down-scale the input signal by right-bit-shift by `num_scale_bit` (e.g. multiplying with 1/2^num_scale_bits)
    (iii)   Truncate the output so that it can fit within `NUM_BITS_SPIKE`
    """

    def __init__(
        self,
        shape: Union[Tuple, int] = (15, 15),
        num_scale_bits: Union[List[int], int] = 5,
    ) -> None:
        """
        Object constructor

        Args:
            shape (Union[Tuple, int]): The number of input and output channels. Defaults to (15, 15).
            num_scale_bits (Union[List[int], int]): number of right-bit-shifts needed for down-scaling the input signal. Defaults to ``5`` for each neuron.
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=True)

        num_scale_bits = (
            [num_scale_bits] * self.size_out
            if isinstance(num_scale_bits, int)
            else num_scale_bits
        )

        [unsigned_bit_range_check(__s, n_bits=5) for __s in num_scale_bits]

        self.num_scale_bits = SimulationParameter(
            num_scale_bits,
            shape=(self.size_out,),
        )
        """(int) Number of right-bit-shifts needed for down-scaling the input signal"""

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Processes the input signal and return the encoded spikes

        Args:
            input_data (np.ndarray): filtered data recorded from IMU sensor. It should be in integer format. (BxTx15)
            record (bool): Unused.

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
        for ch, __scale in enumerate(self.num_scale_bits):
            output_data[:, :, ch] = output_data[:, :, ch] >> __scale

        # truncate the signal
        threshold = (1 << NUM_BITS_SPIKE) - 1
        output_data[output_data > threshold] = threshold

        return output_data, {}, {}


class IAFSpikeEncoder(Module):
    """
    Synchronous integrate and fire spike encoder

    More specifically, denoting a specific filter output by $$y(t)$$,
    this module computes $$s(t) = \\sum_{n=0}^t |y(n)|$$ and as soon as it passes the firing threshold $$\\theta$$, say at time $$t=t_0$$, it produces a spike at $$t=t_0$$ and reduces the counter by $$\theta$$ and keeps accumulating the input signal:
    $$s(t) = s(t_0) - \\theta + \\sum_{n=t_0+1}^t |y(n)|$$ until the next threshold crossing and spike generation occurs.
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (15, 15),
        threshold: Union[List[int], int] = 1024,
    ) -> None:
        """
        Instantiate an IAF spike encoder

        Args:
            shape (Optional[Union[Tuple, int]]): the input and output shape of the module. Defaults to ``(15, 15)``.
            threshold (Union[int, List[int]]): the thresholds of the IAF neurons (quantized unsigned integer). Provide a single integer to use the same threshold for each neuron, or a list of thresholds with one for each neuron. Default: ``1024`` for each neuron.
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=True)

        threshold = (
            [threshold] * self.size_out if isinstance(threshold, int) else threshold
        )
        [unsigned_bit_range_check(th, n_bits=31) for th in threshold]
        self.threshold = SimulationParameter(threshold, shape=(self.size_out,))
        """ (np.ndarray) The unsigned integer thresholds of the IAF neurons"""

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Processes the input signal and return the encoded spikes

        Args:
            input_data (np.ndarray): filtered data recorded from IMU sensor. It should be in integer format. (BxTx15)
            record (bool, optional): Unused.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                Encoded spikes (BxTx15)
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
        for ch, __th in enumerate(self.threshold):
            output_data[:, :, ch] = output_data[:, :, ch] // __th

        # add a zero column to make sure that the dimensions match
        num_spikes = np.hstack([np.zeros((__B, 1, __C), dtype=object), output_data])

        # compute the spikes along the time axis
        spikes = np.diff(num_spikes, axis=1)

        # if there are more than one spike per dt, truncate it to 1
        np.clip(spikes, 0, 1, out=spikes)

        return spikes, {}, {}
