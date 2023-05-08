# -----------------------------------------------------------
# This module implement a simple sample-and-hold module.
# This is not typically implemented like this in the hardware.
# However we can still use this for comparison.
#
#
# (C) Karla Burelo, Saeid Haghighatshoar
# email: {karla.burelo, saeid.haghighatshoar}@synsense.ai
#
# last update: 28.08.2022
# -----------------------------------------------------------
import numpy as np
from imu_preprocessing.util.type_decorator import type_check


class SampleAndHold:
    def __init__(self, sampling_period: int):
        """this module samples and holds the signal in its last index (supposed to be the time-index)

        Args:
            sampling_period (int): sampling period.
        """
        self.sampling_period = sampling_period

    @type_check
    def evolve(self, sig_in: np.ndarray):
        """this modules samples and holds the input signal in its last index.
        We assume that the last index is the time-dimension, so the sample-and-hold is done always along the last axes.
        Use swapaxes to change the axes if needed before calling the function.

        Args:
            sig_in (np.ndarray): input signal of shape (*,*,...,*, T) where T is the time-dimension along which sample-and-hold is done.
        """
        # sample and hold the last component
        sig_in_shape = sig_in.shape

        # additional care with 1-dim array
        if len(sig_in_shape) == 1:
            sig_in = sig_in.reshape(1, -1)

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

        return sig_out

    # add the call version for further convenience
    def __call__(self, *args, **kwargs):
        """
        This function simply calls the `evolve()` function and is added for further convenience.
        """
        return self.evolve(*args, **kwargs)

    def __str__(self):
        string = "Sample-and-Hold maodule:\n" + f"period: {self.sampling_period}"
        return string
