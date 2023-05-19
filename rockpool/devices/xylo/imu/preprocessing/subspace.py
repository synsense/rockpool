"""
Take the input data from IMU sensor and compute the covariance matrix that is then fed into JSVD module.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.modules.module import Module

__all__ = ["SubSpace"]


class SubSpace(Module):
    def __init__(
        self,
        num_bits_in: int,
        num_bits_highprec_filter: int,
        num_bits_multiplier: int,
        num_avg_bitshift: int,
        shape: Optional[Union[Tuple, int]] = 3,
    ) -> None:
        """Data averaging and covariance estimation for the input data.

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of channels. Defaults to 3.
            num_bits_in (int): number of bits in the input data. We assume a sign magnitude format.
            num_bits_highprec_filter (int) : number of bits devoted to computing the high-precision filter (to avoid dead-zone effect)
            num_bits_multiplier (int): number of bits devoted to computing [x(t) x(t)^T]_{ij}. If less then needed, the LSB values are removed.
            num_avg_bitshift (int): number of bitshifts used in the low-pass filter implementation.
                The effective window length of the low-pass filter will be `2**num_avg_bitshift`
        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)
        self.num_bits_in = num_bits_in
        self.num_bits_highprec_filter = num_bits_highprec_filter
        self.num_bits_multiplier = num_bits_multiplier
        self.num_avg_bitshift = num_avg_bitshift

    @type_check
    def evolve(
        self, sig_in: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Processes the input signal sample-by-sample and estimate the subspace.

        Args:
            sig_in (np.ndarray): 3 x T input data received from IMU sensor. It should be in integer format.
            record (bool, optional): If True, the intermediate results are recorded and returned. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]: the covariance matrix of the input data, empty dictionary, empty dictionary
        """

        # check the validity of the data
        if np.max(np.abs(sig_in)) >= 2 ** (self.num_bits_in - 1):
            raise ValueError(
                f"some elements of the input data are beyond the range of {self.num_bits_in} bits!"
            )

        if sig_in.ndim != 2:
            raise ValueError(
                "The input IMU signal should be a 2D signal of dimension 3 x T."
            )

        dim, T = sig_in.shape

        if dim != 3:
            raise ValueError(
                "IMU signal should have three input channels corresponding to x-, y-, z-component of acceleration!"
            )

        # check that the values are indeed integers
        if np.linalg.norm(np.floor(sig_in) - sig_in) > 0:
            raise TypeError(
                "All the components of the input signal should be integers! Make sure that the data is quantized properly!"
            )

        # -- bit size calculation
        # maximimum number of bits that can be used for storing the result of multiplication x(t) * x(t)^T
        max_bits_mult_output = 2 * self.num_bits_in - 1

        # number of bitshifts needed in implementing the high-precision filter
        mult_right_bit_shift = max_bits_mult_output - self.num_bits_multiplier

        # initialize the covariance matrix and covariance matrix with larger precision
        C_highprec = np.zeros((3, 3), dtype=np.int64).astype(object)
        C_list = []

        for sig_val in sig_in.T:
            # compute the multiplication [x(t) x(t)^T]_ij
            xx_trans = np.outer(sig_val, sig_val)

            xx_trans = (
                (xx_trans >> mult_right_bit_shift)
                if mult_right_bit_shift > 0
                else (xx_trans << -mult_right_bit_shift)
            )

            # do the high-precision filter computation
            C_highprec = C_highprec - (C_highprec >> self.num_avg_bitshift) + xx_trans

            # note that due to the specific shape of the low-pass filter used for averaging the input signal,
            # the output of the low-pass filter will be always less than the input max value
            if np.max(np.abs(C_highprec)) >= 2 ** (self.num_bits_highprec_filter - 1):
                raise OverflowError(
                    "Overflow or underflow in the high-precision filter!"
                )

            # apply right-bit-shift to go to the output
            C_regular = C_highprec >> self.num_avg_bitshift

            # add the values of the list
            C_list.append(np.copy(C_regular))

        # convert into numpy arrays: T x 3 x 3
        C_list = np.array(C_list, dtype=object)

        return C_list, {}, {}

    def __str__(self):
        string = (
            "Subspace tracking module for estimating the 3 x 3 covariance matrix of the input 3 x T data:\n"
            + f"number of bits of input signal: {self.num_bits_in}\n"
            + f"number of right bit-shifts used in low-pass filter implementation: {self.num_avg_bitshift}\n"
            + f"averaging window size: {2**self.num_avg_bitshift} samples\n"
            + f"number of bits used for implementing the multiplication module: {self.num_bits_multiplier}\n"
            + f"number of bits used for computing the high-precision filter (to avoid dead-zone in low-pass filter): {self.num_bits_highprec_filter}"
        )

        return string
