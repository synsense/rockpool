"""
Take the input data from IMU sensor and compute the covariance matrix that is then fed into JSVD module.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.xylo.syns63300.imuif.params import (
    NUM_BITS,
    NUM_BITS_HIGHPREC_FILTER_BASE,
    NUM_BITS_MULTIPLIER,
)
from rockpool.devices.xylo.syns63300.imuif.utils import (
    type_check,
    unsigned_bit_range_check,
)
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter, State

__all__ = ["SubSpace"]


class SubSpace(Module):
    """
    Averaging and covariance estimation for 3D IMU signals
    BxTx3 -> BxTx(3x3)
    """

    def __init__(
        self, shape: Optional[Union[Tuple, int]] = (3, 9), num_avg_bitshift: int = 4
    ) -> None:
        """Instantiate a `SubSpace` module

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to ``(3, 9)``.
            num_avg_bitshift (int): number of bitshifts used in the low-pass filter implementation. Defaults to ``4``.
                The effective window length of the low-pass filter will be ``2 ** num_avg_bitshift``
        """
        if shape[1] != shape[0] ** 2:
            raise ValueError(
                f"The output size should be {shape[0] ** 2} to compute the covariance matrix!"
            )

        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        unsigned_bit_range_check(num_avg_bitshift, n_bits=5)

        self.num_avg_bitshift = SimulationParameter(
            num_avg_bitshift, shape=(1,), cast_fn=int
        )

        """number of bitshifts used in the low-pass filter implementation."""
        self.num_bits_highprec_filter = SimulationParameter(
            NUM_BITS_HIGHPREC_FILTER_BASE + num_avg_bitshift, shape=(1,), cast_fn=int
        )
        """number of bits devoted to computing the high-precision filter (to avoid dead-zone effect)"""

        self.C_highprec = State(
            data=np.zeros((self.size_in, self.size_in), dtype=np.int64).astype(object),
            shape=(self.size_in, self.size_in),
            permit_reshape=False,
        )
        """Covariance matrix with high precision"""

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Processes the input signal sample-by-sample and estimate the subspace.

        Args:
            input_data (np.ndarray): batched input data recorded from IMU sensor. It should be in integer format. (BxTx3)
            record (bool, optional): If True, the intermediate results are recorded and returned. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                the covariance matrix of the input data (BxTx3x3)
                empty dictionary
                empty dictionary
        """

        # check the validity of the data
        unsigned_bit_range_check(np.max(np.abs(input_data)), n_bits=NUM_BITS - 1)

        # check that the values are indeed integers
        if np.linalg.norm(np.floor(input_data) - input_data) > 0:
            raise TypeError(
                "All the components of the input signal should be integers! Make sure that the data is quantized properly!"
            )

        # -- Batch processing
        input_data, _ = self._auto_batch(input_data)
        __B, __T, __C = input_data.shape
        input_data = np.array(input_data, dtype=np.int64)

        # -- bit size calculation
        # maximimum number of bits that can be used for storing the result of multiplication x(t) * x(t)^T
        max_bits_mult_output = 2 * NUM_BITS - 1

        # number of bitshifts needed in implementing the high-precision filter
        mult_right_bit_shift = max_bits_mult_output - NUM_BITS_MULTIPLIER

        C_batched = []

        for sig_in in input_data:
            C_list = []
            for sig_val in sig_in:
                # compute the multiplication [x(t) x(t)^T]_ij
                xx_trans = np.outer(sig_val, sig_val)

                xx_trans = (
                    (xx_trans >> mult_right_bit_shift)
                    if mult_right_bit_shift > 0
                    else (xx_trans << -mult_right_bit_shift)
                )

                # do the high-precision filter computation
                self.C_highprec = (
                    self.C_highprec
                    - (self.C_highprec >> self.num_avg_bitshift)
                    + xx_trans
                )

                # note that due to the specific shape of the low-pass filter used for averaging the input signal,
                # the output of the low-pass filter will be always less than the input max value
                if np.max(np.abs(self.C_highprec)) >= 2 ** (
                    self.num_bits_highprec_filter - 1
                ):
                    raise OverflowError(
                        "Overflow or underflow in the high-precision filter!"
                    )

                # apply right-bit-shift to go to the output
                C_regular = self.C_highprec >> self.num_avg_bitshift

                # add the values of the list
                C_list.append(np.copy(C_regular))

            C_list = np.array(C_list, dtype=object)
            C_batched.append(C_list)

        # convert into numpy arrays: B x T x 3 x 3
        C_batched = np.array(C_batched, dtype=object)

        # flatten the channel dimension
        C_batched = C_batched.reshape(__B, __T, self.size_out)
        return C_batched, {}, {}
