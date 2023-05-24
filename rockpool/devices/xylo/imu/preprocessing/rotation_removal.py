"""
Rotation-Removal module:
1. Takes the T x 3 input data received from an IMU sensor,
2. Computes the 3 x 3 sample covariance using subspace estimation module,
3. Applies a sample-and-hold module to compute SVD only at specific periods
4. Computes the SVD of the resulting covariance matrix to find the rotation matrix,
5. Applies the rotation matrix to the input data to compute the rotation-removed version of the input data

The resulting signal is then forwarded to the filterbank module. 
In this version, we are using `object` rather than `np.int64` so that our simulation works for arbitrary number of quantization bit size for the parameters.
"""
from typing import Any, Dict, Tuple

import numpy as np

from rockpool.devices.xylo.imu.preprocessing import JSVD, SampleAndHold, SubSpace
from rockpool.devices.xylo.imu.preprocessing.utils import type_check
from rockpool.nn.combinators import Sequential

__all__ = ["RotationRemoval"]

class RotationRemoval:
    def __init__(
        self, subspace: SubSpace, sampler: SampleAndHold, jsvd: JSVD, num_bits_out: int
    ) -> None:
        """Object constructor.

        Args:
            subspace (SubSpace): subspace estimation/tracking module.
            sampler (SampleAndHold): sample-and-hold module which allows to update the covariance matrix with a given period.
            jsvd (JSVD): JSVD mdoule for computing the SVD of the input covariance matrix and recovering the underlying rotation.
            num_bits_out (int): number of bits in the final signal (obtained after rotation removal).

        """
        self.subspace = subspace
        self.sampler = sampler
        self.jsvd = jsvd
        self.num_bits_out = num_bits_out

    @type_check
    def evolve(
        self, sig_in: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """this modules takes the 3 x T input signal and processes it to produce the 3 x T rotation-removed signal.

        Args:
            sig_in (np.ndarray): the input signal.

        Raises:
            ValueError: if the dimensions do not match.

        Returns:
            np.ndarray: the 3 x T signal after rotation removal.
        """
        if sig_in.shape[-1] != 3:
            raise ValueError("th input signal should have 3 input channels!")

        # compute the covarinces using subspace estimation: do not save the high-precision ones
        # T x 3 x 3
        __B, __T, __C = sig_in.shape
        mod = Sequential(self.subspace, self.sampler)
        covariance_list_SH, _, _ = mod(sig_in)
        covariance_list_SH = covariance_list_SH.reshape((__B, __T, __C, __C))

        # feed the computed covariance matrices into a JSVD module and compute the rotation and diagonal matrix
        covariance_old = -np.ones((3, 3), dtype=object)
        rotation_old = np.eye(3).astype(np.int64).astype(object)
        diagonal_old = np.eye(3).astype(np.int64).astype(object)

        rotation_list = []
        diagonal_list = []
        sig_out = []

        rotation_list_batch = []
        diagonal_list_batch = []
        sig_out_batch = []

        for covariance_list_SH_batch, sig_in_batch in zip(covariance_list_SH, sig_in):
            for covariance_new, sig_in_sample in zip(
                covariance_list_SH_batch, sig_in_batch
            ):
                # check if the covariance matrix is repeated
                if np.linalg.norm(covariance_old - covariance_new) == 0:
                    # do not do any computation: just copy-paste the old values
                    rotation_list.append(np.copy(rotation_old))
                    diagonal_list.append(np.copy(diagonal_old))

                    # output signal sample after rotation removal
                    sig_out_sample = self.rotate(rotation_old.T, sig_in_sample)

                    sig_out.append(np.copy(sig_out_sample))

                    continue

                # compute the JSVD
                # return R_list, C_list, R_last_sorted, C_last_sorted
                rotation_new, diagonal_new = self.jsvd(covariance_new)

                # correct the sign of rotation to keep the consistency with the previous rotations
                # no need to change the diagonal matrix
                sign_new_old = (
                    np.sign(np.diag(rotation_new.T @ rotation_old))
                    .astype(np.int8)
                    .astype(object)
                )
                rotation_new = rotation_new @ np.diag(sign_new_old)

                # add them to the list
                rotation_list.append(np.copy(rotation_new))
                diagonal_list.append(np.copy(diagonal_new))

                # output signal sample after rotation removal
                sig_out_sample = self.rotate(rotation_new.T, sig_in_sample)

                sig_out.append(np.copy(sig_out_sample))

                # update the covariance matrix
                covariance_old = covariance_new
                rotation_old = rotation_new
                diagonal_old = diagonal_new

            # convert into array and return
            rotation_list = np.asarray(rotation_list, dtype=object)
            diagonal_list = np.asarray(diagonal_list, dtype=object)
            sig_out = np.asarray(sig_out, dtype=object).T

        rotation_list_batch.append(rotation_list)
        diagonal_list_batch.append(diagonal_list)
        sig_out_batch.append(sig_out)

        rotation_list_batch = np.asarray(rotation_list_batch, dtype=object)
        diagonal_list_batch = np.asarray(diagonal_list_batch, dtype=object)
        sig_out_batch = np.asarray(sig_out_batch, dtype=object)

        # return  3 x T output signal and T x 3 x 3 rotation/diagonal matrices
        return sig_out_batch, rotation_list_batch, diagonal_list_batch

    # add the call version for further convenience
    def __call__(self, *args, **kwargs):
        """
        This function simply calls the `evolve()` function and is added for further convenience.
        """
        return self.evolve(*args, **kwargs)

    # utility modules
    @type_check
    def rotate(self, rotation_matrix: np.ndarray, sig_sample: np.ndarray) -> np.ndarray:
        """this module takes a rotation matrix and also a 3 x 1 signal sample and rotates it.

        Args:
            rotation_matrix (np.ndarray): 3 x 3 input rotation matrix.
            sig_sample (np.ndarray): 3 x 1 input signal sample.

        Returns:
            np.ndarray: 3 x 1 input signal after being multiplied with transpose of rotation matrix (rotation removal).
        """
        # number of bits used in rotation removal
        num_bits_rotation = self.jsvd.num_bits_rotation

        # number of bits used in the input signal
        num_bits_in = self.subspace.num_bits_in

        # number of bitshifts needed to fit the multiplication into the buffer
        # NOTE: the amplitude amplification due to multiplication with a rotation matrix
        # is already taken into account by right-bit-shift of 1
        num_right_bit_shifts = num_bits_rotation + num_bits_in - self.num_bits_out

        sig_out = []

        for row_vec in rotation_matrix:
            buffer = 0

            for val_rotation, val_sig_in in zip(row_vec, sig_sample):
                update = (val_rotation * val_sig_in) >> num_right_bit_shifts

                if abs(update) >= 2 ** (self.num_bits_out - 1):
                    raise ValueError(
                        f"The update value {update} encountered in rotation-input signal multiplication is beyond the range [-{2**(self.num_bits_out-1)}, +{2**(self.num_bits_out-1)}]!"
                    )

                buffer += update

                if abs(buffer) >= 2 ** (self.num_bits_out - 1):
                    raise ValueError(
                        f"The beffer value {buffer} encountered in rotation-input signal multiplication is beyond the range [-{2**(self.num_bits_out-1)}, +{2**(self.num_bits_out-1)}]!"
                    )

            # add this component
            sig_out.append(buffer)

        sig_out = np.asarray(sig_out, dtype=object)

        return sig_out
