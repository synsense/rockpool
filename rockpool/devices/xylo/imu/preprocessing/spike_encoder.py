"""
This module implements the spike encoding for the signal coming out of filters or IMU sensor directly.
"""

import numpy as np
from rockpool.devices.xylo.imu.preprocessing.utils import type_check


class ScaleSpikeEncoder:
    """
    Encode spikes as follows

    (i)     Apply full-wave rectification to the input signal
    (ii)    Down-scale the input signal by right-bit-shift by `num_scale_bit` (e.g. multiplying with 1/2^num_scale_bits)
    (iii)   Truncate the output so that it can fit within `num_out_bits`
    """

    def __init__(self, num_scale_bits: int, num_out_bits: int) -> None:
        """
        Object constructor

        Args:
            num_scale_bits (int): number of right-bit-shifts needed for down-scaling the input signal.
            num_out_bits (int): number of bits devoted to storing the output spike encoding.
        """
        self.num_scale_bits = num_scale_bits
        self.num_out_bits = num_out_bits

    @type_check
    def evolve(self, sig_in: np.ndarray):
        # compute the absolute value of the input signal
        sig_in = np.abs(sig_in)

        # scale the signal
        sig_in = sig_in >> self.num_scale_bits

        # truncate the signal
        threshold = (1 << self.num_out_bits) - 1

        sig_in[sig_in > threshold] = threshold

        return sig_in


class IAFSpikeEncoder:
    """
    Synchronous integrate and fire spike encoder

    More specifically, denoting a specific filter output by $$y(t)$$,
    this module computes $$s(t) = \\sum_{n=0}^t |y(n)|$$ and as soon as it passes the firing threshold $$\\theta$$.
    Say at a time $$t=t_0$$, it produces a spike at $$t=t_0$$ and reduces the counter by $$\theta$$ and keeps accumulating the input signal
    $$s(t) = s(t_0) - \\theta + \\sum_{n=t_0+1}^t |y(n)|$$ until the next threshold crossing and spike generation happens.
    """

    def __init__(self, iaf_threshold: int) -> None:
        """
        Object constructor

        Args:
            iaf_threshold (int): the threshold of the IAF neuron (quantized)
        """
        self.iaf_threshold = iaf_threshold

    @type_check
    def evolve(self, sig_in: np.ndarray):
        # check the number of channels
        if len(sig_in.shape) == 1:
            sig_in = sig_in.reshape(1, -1)

        # compute the absolute value of the signal
        sig_in = np.abs(sig_in)

        # compute the cumsum along the time axis
        sig_in = np.cumsum(sig_in, axis=1)

        # compute the number of spikes produced so far
        num_spikes = sig_in // self.iaf_threshold

        # add a zero column to make sure that the dimensions match
        num_spikes = np.hstack(
            [np.zeros((num_spikes.shape[0], 1), dtype=object), num_spikes]
        )

        # compute the spikes
        spikes = np.diff(num_spikes, axis=1)

        # if there are more than one spikes, truncate it to 1
        spikes[spikes > 1] = 1

        return spikes
