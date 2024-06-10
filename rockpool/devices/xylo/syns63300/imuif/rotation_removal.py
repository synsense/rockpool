"""
Rotation-Removal module for removing the rotation from the IMU input signal.
"""

from typing import Any, Dict, Tuple, Optional, Union

import numpy as np

from rockpool.devices.xylo.syns63300.imuif.rotation import JSVD, SampleAndHold, SubSpace
from rockpool.devices.xylo.syns63300.imuif.utils import (
    type_check,
    unsigned_bit_range_check,
)
from rockpool.devices.xylo.syns63300.imuif.params import NUM_BITS, NUM_BITS_ROTATION
from rockpool.nn.combinators import Sequential
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter

__all__ = ["RotationRemoval"]


class RotationRemoval(Module):
    """
    A Rockpool module simulating the rotation estimation and removal block in the Xylo IMU interface

    1. Takes the T x 3 input data received from an IMU sensor
    2. Computes the 3 x 3 sample covariance using subspace estimation module
    3. Applies a sample-and-hold module to compute SVD only at specific periods
    4. Computes the SVD of the resulting covariance matrix to find the rotation matrix
    5. Applies the rotation matrix to the input data to compute the rotation-removed version of the input data

    The resulting signal is then forwarded to the filterbank module.
    In this version, we are using `object` rather than `np.int64` so that our simulation works for arbitrary number of quantization bit size for the parameters.
    """

    def __init__(
        self,
        shape: Optional[Union[Tuple, int]] = (3, 3),
        num_avg_bitshift: int = 4,
        sampling_period: int = 10,
    ) -> None:
        """Instantiate a `RotationRemoval` object

        Args:
            shape (Optional[Union[Tuple, int]], optional): The number of input and output channels. Defaults to ``(3, 3)``.
            num_avg_bitshift (int): number of bitshifts used in the low-pass filter implementation. Default to ``4``. The effective window length of the low-pass filter will be ``2 ** num_avg_bitshift``
            sampling_period (int): Sampling period that the signal is sampled and held, in number of timesteps. Defaults to ``10``.

        """
        super().__init__(shape=shape, spiking_input=False, spiking_output=False)

        unsigned_bit_range_check(num_avg_bitshift, n_bits=5)
        unsigned_bit_range_check(sampling_period, n_bits=11)

        self.sub_estimate = Sequential(
            SubSpace(
                shape=(self.size_in, self.size_in**2),
                num_avg_bitshift=num_avg_bitshift,
            ),
            SampleAndHold(
                sampling_period=sampling_period,
                shape=(self.size_in**2, self.size_in**2),
            ),
        )

        self.num_avg_bitshift = SimulationParameter(
            num_avg_bitshift, shape=(1,), cast_fn=int
        )
        """number of bitshifts used in the low-pass filter implementation"""

        self.sampling_period = SimulationParameter(
            sampling_period, shape=(1,), cast_fn=int
        )
        """sampling period that the signal is sampled and held"""

        self.jsvd = JSVD()

    @type_check
    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Take the BxTx3 raw analog IMU signal and processes it to produce the BxTx3 rotation-removed signal.

        Args:
            input_data (np.ndarray): the input signal (BxTx3)
            record (bool, optional): record flag to match with the other rockpool modules. Practically useless. Defaults to False.

        Raises:
            ValueError: if the dimensions do not match.

        Returns:
            np.ndarray: Output signal after rotation removal (BxTx3)
            Dict[str, Any]: empty dictionary
            Dict[str, Any]: empty dictionary
        """

        # Input handling (BxTx3)
        input_data, _ = self._auto_batch(input_data)
        input_data = np.array(input_data, dtype=np.int64).astype(object)
        __B, __T, __C = input_data.shape

        # compute the covariances using subspace estimation: do not save the high-precision ones
        # B x T x 3 x 3
        batch_cov_SH, _, _ = self.sub_estimate(input_data)
        batch_cov_SH = batch_cov_SH.reshape((__B, __T, __C, __C))

        # feed the computed covariance matrices into a JSVD module and compute the rotation and diagonal matrix
        covariance_old = -np.ones((3, 3), dtype=object)
        rotation_old = np.eye(3).astype(np.int64).astype(object)

        data_out = []

        # loop over the batch
        for cov_SH, signal in zip(batch_cov_SH, input_data):
            signal_out = []
            # loop over the time dimension
            for cov_new, sample in zip(cov_SH, signal):
                # check if the covariance matrix is repeated
                if np.linalg.norm(covariance_old - cov_new) == 0:
                    # output signal sample after rotation removal
                    sample_out = self.rotate(sample, rotation_old.T)
                    signal_out.append(sample_out)

                # if not, compute the JSVD
                else:
                    rotation_new, diagonal_new = self.jsvd(cov_new)

                    # correct the sign of rotation to keep the consistency with the previous rotations
                    # no need to change the diagonal matrix
                    sign_new_old = (
                        np.sign(np.diag(rotation_new.T @ rotation_old))
                        .astype(np.int8)
                        .astype(object)
                    )
                    rotation_new = rotation_new @ np.diag(sign_new_old)

                    # output signal sample after rotation removal
                    sample_out = self.rotate(sample, rotation_new.T)
                    signal_out.append(sample_out)

                    # update the covariance matrix
                    covariance_old = cov_new
                    rotation_old = rotation_new

            data_out.append(signal_out)

        data_out = np.array(data_out, dtype=object)

        return data_out, {}, {}

    # utility modules
    @type_check
    def rotate(self, sample: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """Rotate a 1 timestep IMU signal. The number of bitshifts needed to fit the multiplication into the buffer!

        NOTE: the amplitude amplification due to multiplication with a rotation matrix is already taken into account by right-bit-shift of 1

        Args:
            sample (np.ndarray): one timestep signal (3,).
            rotation_matrix (np.ndarray): 3 x 3 rotation matrix.

        Returns:
            np.ndarray: Rotation removed sample.
        """

        signal_out = []

        for row in rotation_matrix:
            buffer = 0
            for rot, val in zip(row, sample):
                update = (rot * val) >> NUM_BITS_ROTATION

                if abs(update) >= 2 ** (NUM_BITS - 1):
                    raise ValueError(
                        f"The update value {update} encountered in rotation-input signal multiplication is beyond the range [-{2**(NUM_BITS-1)}, +{2**(NUM_BITS-1)}]!"
                    )

                buffer += update

                if abs(buffer) >= 2 ** (NUM_BITS - 1):
                    raise ValueError(
                        f"The beffer value {buffer} encountered in rotation-input signal multiplication is beyond the range [-{2**(NUM_BITS-1)}, +{2**(NUM_BITS-1)}]!"
                    )

            # add this component
            signal_out.append(buffer)

        signal_out = np.asarray(signal_out, dtype=object)
        return signal_out
