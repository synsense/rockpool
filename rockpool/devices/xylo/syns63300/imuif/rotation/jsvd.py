"""
This module takes the 3 x 3 covariance matrix from the IMU subspace estimation module and
computes the 3 x 3 rotation matrix and 3 x 3 diagonal matrix.

    (i)     Compute R.T @ C @ R in one-shot rather than doing it in two steps with 2 matrix multiplication.
    (ii)    Apply infinite-bit approximation, which is valid when number of angles in lookup table is large enough.
    (iii)   This yields a higher precision in implementation of JSVD.
"""

import warnings
from typing import List, Tuple

import numpy as np

from rockpool.devices.xylo.syns63300.imuif.utils import (
    type_check,
    unsigned_bit_range_check,
    bucket_decorator,
)

from rockpool.devices.xylo.syns63300.imuif.params import (
    COV_EXTRA_BIT,
    ROT_EXTRA_BIT,
    NUM_BITS_COVARIANCE,
    NUM_BITS_ROTATION,
    NROUND,
    NUM_ANGLES,
    NUM_BITS,
)

EPS = 10e-30

__all__ = ["JSVD", "RotationLookUpTable"]


class RotationLookUpTable:
    """A lookup table for JSVD algorithm.
    The range of angles for lookup tables are [0, 45] and they are quantized into `num_angles` angle bins.
    The data for each angle is quantized/truncated into `num_bits` bits.
    """

    def __init__(self) -> None:
        """
        To make sure that the lookup table has a good precision we do the following:
            (i)     we always work with angles in the range [0, 45] degrees.
            (ii)    since the tan(2 theta) ranges in [0, infinity] to avoid loss of precision,
                        - we use tan(2 theta) for theta in [0, 22.5]
                        - we use cot(2 theta) for theta in [22.5, 45].
            (iii)   the lookup table consists of
                        - the tan2-cot2 value in the [0, 22.5] and [22.5,45]
                        - sin and cos in the whole range [0, 45]
                        - 1/sin(2 theta) in the range [22.5, 45]
                        - 1/cos(2 theta) in the range [0, 22.5]

            (iv)       the row of the lookup table used depends on how abs(2b) compare with abs(a-c) in the 2D sub-matrix
                       |a  b|
                       |b  c|
        """

        # in the new method, we are able to explore the whole range of angles in [0, 45]
        max_angle_degree = 45
        max_angle_radian = max_angle_degree / 180 * np.pi

        self.angles_radian = np.linspace(0, max_angle_radian, NUM_ANGLES)
        self.angles_degree = self.angles_radian / np.pi * 180

        # compute the sin and cos in the lookup table
        self.sin_vals = np.sin(self.angles_radian)
        self.cos_vals = np.cos(self.angles_radian)

        # compute the index of all angles less than 22.5 degrees
        ind_low_angle = self.angles_degree <= max_angle_degree / 2

        # compute the tan(2 theta) and cot(2 theta) in the table
        # set them equal to NAN at the positions when they should not be used.
        self.tan2_vals = np.tan(2 * self.angles_radian)
        self.tan2_vals[ind_low_angle == False] = np.nan

        self.cot2_vals = 1 / np.tan(2 * self.angles_radian)
        self.cot2_vals[ind_low_angle == True] = np.nan

        # compute the 1/sin(2 theta) and 1/cos(2 theta)
        # set them equal to NAN at the positions when they are not needed.

        self.inv_2cos2_vals = 1 / (2 * np.cos(2 * self.angles_radian) + EPS)
        self.inv_2cos2_vals[ind_low_angle == False] = np.nan

        self.inv_2sin2_vals = 1 / (2 * np.sin(2 * self.angles_radian) + EPS)
        self.inv_2sin2_vals[ind_low_angle == True] = np.nan

        # compute the quantized values:
        #   (i)     switch to python-version with infinite number of bits
        #   (ii)    add EPS to make sure that all the values are smaller than 1 and can fit in num_bits after quantization
        # NOTE: all the values are positive in the lookup table

        self.tan2_vals_quantized = np.asarray(
            [
                int(el / (1 + EPS)) if not np.isnan(el) else np.nan
                for el in (self.tan2_vals * 2**NUM_BITS)
            ],
            dtype=object,
        )
        self.cot2_vals_quantized = np.asarray(
            [
                int(el / (1 + EPS)) if not np.isnan(el) else np.nan
                for el in (self.cot2_vals * 2**NUM_BITS)
            ],
            dtype=object,
        )
        self.sin_vals_quantized = np.asarray(
            [
                int(el / (1 + EPS)) if not np.isnan(el) else np.nan
                for el in ((2**NUM_BITS - 1) * self.sin_vals)
            ],
            dtype=object,
        )
        self.cos_vals_quantized = np.asarray(
            [
                int(el / (1 + EPS)) if not np.isnan(el) else np.nan
                for el in ((2**NUM_BITS - 1) * self.cos_vals)
            ],
            dtype=object,
        )

        self.inv_2cos2_vals_quantized = np.asarray(
            [
                int(el) if not np.isnan(el) else np.nan
                for el in (2**NUM_BITS * self.inv_2cos2_vals)
            ],
            dtype=object,
        )
        self.inv_2sin2_vals_quantized = np.asarray(
            [
                int(el) if not np.isnan(el) else np.nan
                for el in (2**NUM_BITS * self.inv_2sin2_vals)
            ],
            dtype=object,
        )

    @bucket_decorator
    def find_angle(self, a: int, b: int, c: int) -> tuple:
        """this module computes the best angle in lookup table that matches the given `a`, `b`, `c` parameters.
        For further details see the paper-dropbox document:
        https://paper.dropbox.com/doc/Hardware-implementation-of-3-x-3-SVD-for-IMU-preprocessing--Bnj3EbtGBF9Th1GUqYAFXn_3Ag-g16myO9A46nqYVLmFkNgQ

        Args:
            a (int): `a` parameter of the 2 x 2 covariance matrix.
            b (int): `b` parameter of the 2 x 2 covariance matrix.
            c (int): `c` parameter of the 2 x 2 covariance matrix.

        Raises:
            ValueError: is the values are not chosen properly.

        Returns:
            Tuple: row-index, angle-deg, angle-rad, sin-val, cos-val, inv_2sin2_vals, inv_2cos2_vals, sin_val_quant, cos_val_quant, inv_2sin2_val_quant, inv_2cos2_val_quant in the lookup table.
        """
        for param in [a, b, c]:
            if abs(param - int(param)) > 0:
                raise ValueError(
                    "Lookup table only accepts integer `a, b, c` entries! It seems quantization is not taken into account!"
                )

        if a < 0 or b < 0 or c < 0:
            raise ValueError(
                "all the `a, b, c` parameters used to fetch a row of lookup table should be positive!"
            )
        # first check marginal cases where the angle can be easily found
        if b == 0:
            row_index = 0

        elif a == c:
            row_index = NUM_ANGLES - 1

        else:
            # here we follow one of these two methods
            if abs(2 * b) <= abs(a - c):
                # in this case `0<= tan(2 theta) <= 1` and angle is in the range [0, 22.5] degrees.
                # NOTE: here we should use the `tan(2 theta)` values to find the corresponding row of the lookup table.
                # we simply do the following:
                #   (i)     we count the number of rows with  tan(2 theta) * abs(a-c) <= abs(2*b),
                #   (ii)    we do this to find the corresponding row of the lookup table.
                row_index = np.sum(
                    [
                        (
                            el * abs(a - c) <= abs(2 * b) * 2**NUM_BITS
                            if not np.isnan(el)
                            else False
                        )
                        for el in self.tan2_vals_quantized
                    ]
                )

                # this check is not needed in hardware and is done here due to precision issues in python
                if np.isnan(self.inv_2cos2_vals_quantized[row_index]):
                    row_index -= 1

            else:
                # in this case `0<= cot(2 theta) <= 1` and angle is in the range [22.5, 45] degrees.
                # NOTE: here we should use the `cot(2 theta)` values to find the corresponding row of the lookup table.
                # we simply do the following:
                #   (i)     we count the number of rows with  tan(2 theta) * abs(a-c) <= abs(2*b),
                #   (ii)    we do this to find the corresponding row of the lookup table.
                #   (iii)   since the cot(2 theta) values start later in the lookup table we need shift the row-index
                row_index = np.sum(
                    [
                        (
                            el * abs(2 * b) >= abs(a - c) * 2**NUM_BITS
                            if not np.isnan(el)
                            else True
                        )
                        for el in self.cot2_vals_quantized
                    ]
                )

                if row_index == NUM_ANGLES:
                    row_index -= 1

        return (
            row_index,
            self.angles_degree[row_index],
            self.angles_radian[row_index],
            self.sin_vals[row_index],
            self.cos_vals[row_index],
            self.inv_2sin2_vals[row_index],
            self.inv_2cos2_vals[row_index],
            self.tan2_vals[row_index],
            self.cot2_vals[row_index],
            self.sin_vals_quantized[row_index],
            self.cos_vals_quantized[row_index],
            self.inv_2sin2_vals_quantized[row_index],
            self.inv_2cos2_vals_quantized[row_index],
            self.tan2_vals_quantized[row_index],
            self.cot2_vals_quantized[row_index],
        )

    def print_table(self, format: str = "dec", report: bool = True) -> str:
        """Print the lookup table in a given format

        Args:
            format (str, optional): 'bin' or 'hex'. Defaults to "dec".
            report (bool, optional): print the table or not. Defaults to True.

        Raises:
            ValueError: When the format is not supported.

        Returns:
            str: The lookup table as a string in the given format.
        """
        format = format.lower()

        if format not in ["bin", "hex", "dec"]:
            raise ValueError(
                "only binary ('bin'), hexadecimal ('hex'), and decimal ('dec') are supported!"
            )

        string = (
            "th-deg\ttan2-Q\tcot2-Q\t\tsin-Q\t\tcos-Q\t\tinv_2sin2-Q\tinv_2cos2-Q\n"
        )
        string += ("-" * 90) + "\n"

        for (
            theta_deg,
            tan2_val_quant,
            cot2_val_quant,
            sin_val_quant,
            cos_val_quant,
            inv_2sin2_val_quant,
            inv_2cos2_val_quant,
        ) in zip(
            self.angles_degree,
            self.tan2_vals_quantized,
            self.cot2_vals_quantized,
            self.sin_vals_quantized,
            self.cos_vals_quantized,
            self.inv_2sin2_vals_quantized,
            self.inv_2cos2_vals_quantized,
        ):
            if format == "bin":
                string += (
                    f"{theta_deg:0.3f}\t{bin(tan2_val_quant) if not np.isnan(tan2_val_quant) else np.nan}\t{bin(cot2_val_quant) if not np.isnan(cot2_val_quant) else np.nan}\t\t"
                    + f"{bin(sin_val_quant)}\t\t{bin(cos_val_quant)}\t\t{bin(inv_2sin2_val_quant) if not np.isnan(inv_2sin2_val_quant) else np.nan}\t\t{bin(inv_2cos2_val_quant) if not np.isnan(inv_2cos2_val_quant) else np.nan}\n"
                )
            elif format == "hex":
                string += (
                    f"{theta_deg:0.3f}\t{hex(tan2_val_quant) if not np.isnan(tan2_val_quant) else np.nan}\t{hex(cot2_val_quant) if not np.isnan(cot2_val_quant) else np.nan}\t\t"
                    + f"{hex(sin_val_quant)}\t\t{hex(cos_val_quant)}\t\t{hex(inv_2sin2_val_quant) if not np.isnan(inv_2sin2_val_quant) else np.nan}\t\t{hex(inv_2cos2_val_quant) if not np.isnan(inv_2cos2_val_quant) else np.nan}\n"
                )
            elif format == "dec":
                string += (
                    f"{theta_deg:0.3f}\t{tan2_val_quant if not np.isnan(tan2_val_quant) else np.nan}\t{cot2_val_quant if not np.isnan(cot2_val_quant) else np.nan}\t\t"
                    + f"{sin_val_quant}\t\t{cos_val_quant}\t\t{inv_2sin2_val_quant if not np.isnan(inv_2sin2_val_quant) else np.nan}\t\t{inv_2cos2_val_quant if not np.isnan(inv_2cos2_val_quant) else np.nan}\n"
                )
            else:
                raise ValueError(
                    "only binary, hexadecimal, and decimal formats are supported!"
                )

        if report:
            print(string)

        return string

    def __str__(self) -> str:
        """
        return the decimal format of the lookup table.
        """
        return self.print_table(
            format="dec",
            report=False,
        )


class JSVD:
    """
    Runs Jaccobi SVD algorithm in FPGA precision

    Note: This is the 2nd version of the algorithm and uses joint matrix multiplication in order to reduce the
    number of multiplication operations
    """

    def __init__(self) -> None:
        """Object constructor."""

        self.lookuptable = RotationLookUpTable()
        """lookup table used for computation"""

    @type_check
    def __call__(
        self, C_in: np.ndarray, record: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """Run Jaccobi-SVD and return all the intermediate states.

        Args:
            C_in (np.ndarray): 3 x 3 covariance matrix.

        Raises:
            ValueError: The input covariance matrix should be 3 x 3.
            ValueError: The input covariance matrix does not seem to be PSD! This may cause numerical issues in computation!
            ValueError: The input covariance matrix does not seem to be symmetric! This may cause numerical issues in computation!
            ValueError: The input covariance matrix does not fit the number of bits specified for the computation!
            ValueError: Over- or under-flow happened in updating the almost-diagonal matrix D!
            ValueError: Negative value in the diagonal matrix D!

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                R_last_sorted (np.ndarray): the last rotation matrix after sorting.
                C_last_sorted (np.ndarray): the last covariance matrix after sorting.
        """
        # check the dimensions and apply other sanity checks
        row, col = C_in.shape

        if row != 3 or col != 3:
            raise ValueError("the input covariance matrix should be 3 x 3.")

        if np.any(np.linalg.eigvals(C_in.astype(np.float64)) < 0.0):
            warnings.warn(
                "The input covariance matrix does not seem to be PSD! This may cause numerical issues in computation!"
            )

        if (
            np.linalg.norm(C_in - C_in.T) / np.linalg.norm(C_in.astype(np.float64))
            > 0.000001
        ):
            raise ValueError(
                "The input covariance matrix does not seem to be symmetric! This may cause issues in computation!"
            )

        unsigned_bit_range_check(np.max(np.abs(C_in)), NUM_BITS_COVARIANCE - 1)

        # estimated covariance matrices
        C_list = [C_in]

        # estimated rotation matrices
        R_list = [(2 ** (NUM_BITS_ROTATION - 1) - 1) * np.eye(3, dtype=object)]

        R = np.copy(R_list[0])
        D = np.copy(C_list[0])

        niter = 0

        for n_round in range(NROUND):
            # number of rounds the algorithm should run

            # NOTE: check the early termination condition
            # if all the off-diagonal elements in D are already zero, it is diagonal and in SVD format.
            # this happens when D if of the following form:
            # | *  0  0 |
            # | 0  *  0 |
            # | 0  0  * |
            #
            # we check this by making sure that all the off-diagonal terms are zero
            if np.sum(np.abs(D - np.diag(np.diag(D)))) == 0:
                # print(f"JSVD: early termination n_round: {n_round}, dim: {dim} (out of [3, 2, 1])!")
                break

            # otherwise continue by iterating over dimensions ....
            for dim in range(0, 3)[-1::-1]:
                # dim starts as 2 -> 1 -> 0
                # update the number of iteration
                niter += 1

                # the dimension that is removed in computation: 2 (0,1 are used) -> 1 (0,2 are used) -> 0 (1,2 are used)
                # we do this by using the following `selection matrix`
                selection = self.selection_matrix(dim=dim)

                # `selection matrix` specifies which 2 x 2 sub-matrix should be used
                #
                # 3 possible cases depending on dim in {0, 1, 2}
                #
                # dim = 0: |*  *  *| --> |a  b| -> selection matrix = |0  0  0|
                #          |*  a  b|     |b  c|                       |0  1  1|
                #          |*  b  c|                                  |0  1  1|
                #
                # dim = 1: |a  *  b| --> |a  b| -> selection matrix = |1  0  1|
                #          |*  *  *|     |b  c|                       |0  0  0|
                #          |b  *  c|                                  |1  0  1|
                #
                # dim = 2: |a  b  *| --> |a  b| -> selection matrix = |1  1  0|
                #          |b  c  *|     |b  c|                       |1  1  0|
                #          |*  *  *|                                  |0  0  0|

                # choose the sub-matrix
                sub_matrix_D = np.copy(D[selection == 1]).reshape(2, 2)

                # extract the three values in the 2 x 2 sub-matrix
                # NOTE: the sub-matrix is symmetric.
                a, b, c = sub_matrix_D[0, 0], sub_matrix_D[0, 1], sub_matrix_D[1, 1]

                # NOTE: an additional early termination here
                # if b==0, the 2 x 2 sub-matrix is already diagonal so no need to apply JSVD to diagonalize it.
                # skip JUST this dim: other dims may still need computation.
                if b == 0:
                    # do not run this iteration ...
                    # NOTE: but we store the previous state here for later comparison
                    C_list.append(np.copy(C_list[-1]))
                    R_list.append(np.copy(R_list[-1]))
                    continue

                # extract the entries from lookup table ...
                # since the lookup table contains only positive values:
                #   (i)  save the sign to be used later
                #   (ii) take the sign into account in the final computation
                sign_tan2 = +1 if b * (a - c) >= 0 else -1

                if b * (a - c) > 0:
                    sign_tan2 = +1
                elif b * (a - c) == 0:
                    sign_tan2 = int(np.sign(b))
                else:
                    sign_tan2 = -1

                # fetch the sin and cos values from the lookup table
                # apply absolute values and use just positive values for fetching the row of
                # lookup table since sign is already taken into account
                (
                    row_index,
                    angle_deg,
                    angle_rad,
                    sin_val,
                    cos_val,
                    inv_2sin2_val,
                    inv_2cos2_val,
                    tan2_val,
                    cot2_val,
                    sin_val_quant,
                    cos_val_quant,
                    inv_2sin2_val_quant,
                    inv_2cos2_val_quant,
                    tan2_val_quant,
                    cot2_val_quant,
                ) = self.lookuptable.find_angle(abs(a), abs(b), abs(c))

                ################################# update C ###################################

                # choose which one is better (A):(a-c)/( 2cos(2 theta) ) or (B):2b/( 2sin(2 theta) )
                # this depends on which range of angles we are
                #       [0, 22.5] -> (A) is better and gives better precision
                #       [22.5, 45] -> (B) is better and gives better precision
                rotation_correction_small_angles = (a - c) * inv_2cos2_val_quant
                rotation_correction_large_angles = (
                    2 * b * inv_2sin2_val_quant * sign_tan2
                )

                # choose when to apply method (A) or method (B)
                if abs(2 * b) > abs(a - c):
                    # LARGE ANGLES: tan(2 theta) > 1.0

                    # NOTE: sign correction is needed here since `SIN(2 theta)` appears here and `SIN(2 theta)` has the same sign as `TAN(2 theta)``.
                    a_new = (
                        ((a + c) << NUM_BITS) // 2 + rotation_correction_large_angles
                    ) >> NUM_BITS
                    c_new = (
                        ((a + c) << NUM_BITS) // 2 - rotation_correction_large_angles
                    ) >> NUM_BITS

                    if abs(a_new) >= 2 ** (
                        NUM_BITS_COVARIANCE - 1 + COV_EXTRA_BIT
                    ) or abs(c_new) >= 2 ** (NUM_BITS_COVARIANCE - 1 + COV_EXTRA_BIT):
                        raise ValueError(
                            "over- or under-flow happened in updating the almost-diagonal matrix D!"
                        )

                else:
                    # SMALL ANGLES: tan(2 theta) <= 1.0

                    # NOTE: sign modification is NOT needed here since we have `COS(2 theta)` and it is always positive even for negative angles.
                    a_new = (
                        ((a + c) << NUM_BITS) // 2 + rotation_correction_small_angles
                    ) >> NUM_BITS
                    c_new = (
                        ((a + c) << NUM_BITS) // 2 - rotation_correction_small_angles
                    ) >> NUM_BITS

                    if abs(a_new) >= 2 ** (
                        NUM_BITS_COVARIANCE - 1 + COV_EXTRA_BIT
                    ) or abs(c_new) >= 2 ** (NUM_BITS_COVARIANCE - 1 + COV_EXTRA_BIT):
                        raise ValueError(
                            "over- or under-flow happened in updating the almost-diagonal matrix D!"
                        )

                # check the sign and make sure there is no problem on the diagonal elements
                # NOTE: the diagonal elements should be kept always positive
                # so set the negative values to 0
                if a_new <= 0 or c_new <= 0:
                    warnings.warn(
                        "negative diagonal elements encountered during covariance update!"
                    )

                a_new = a_new if a_new > 0 else 0
                c_new = c_new if c_new > 0 else 0

                # copy the computed and updated 2 x 2 covariance matrix
                sub_matrix_D_updated = np.array([[a_new, 0], [0, c_new]], dtype=object)

                # update the column vector and (dim, dim)-diagonal element in the not-selected part of C
                # NOTE: as an example
                # if dim=1  we select the 2 x 2 sub-matrix containing the elements (a, b, c) -> after the update we obtain (a_new, 0, c_new)
                # we need to also update the 2 x 1 vector containing (*, x) and also the diagonal element `+` at location (1,1) in the matrix
                # |a  *  b|
                # |*  +  x|
                # |b  x  c|

                if dim == 0:
                    sub_vector_D = D[[1, 2], 0]
                elif dim == 1:
                    sub_vector_D = D[[0, 2], 1]
                else:
                    # dim=2
                    sub_vector_D = D[[0, 1], 2]

                sub_vector_D_updated = np.zeros(2, dtype=object)
                sub_vector_D_updated[0] = (
                    cos_val_quant * sub_vector_D[0]
                    + sin_val_quant * sign_tan2 * sub_vector_D[1]
                ) >> NUM_BITS
                sub_vector_D_updated[1] = (
                    -sin_val_quant * sign_tan2 * sub_vector_D[0]
                    + cos_val_quant * sub_vector_D[1]
                ) >> NUM_BITS

                if np.abs(sub_vector_D[0]) >= 2 ** (
                    NUM_BITS_COVARIANCE - 1 + COV_EXTRA_BIT
                ) or np.abs(sub_vector_D[1]) >= 2 ** (
                    NUM_BITS_COVARIANCE - 1 + COV_EXTRA_BIT
                ):
                    raise ValueError(
                        "over- or under-flow happened in updating the diagonal matrix D!"
                    )

                ## - Uncomment in debug mode - ##
                # Double-check the result with the direct matrix multiplication

                # U, _, _ = np.linalg.svd(sub_matrix_D.astype(np.float64))
                # rotation = np.eye(3)
                # rotation[selection == 1] = U.ravel()
                # D_rotated = (rotation.T @ D @ rotation).astype(np.int64)

                # copy the updated values in the C matrix
                # 2 x 2 sub-matrix
                D[selection == 1] = sub_matrix_D_updated.ravel()

                # 2 x 1 upper subvector, diagonal (dim, dim), and 1 x 2 lower subvector
                if dim == 0:
                    D[1, 0] = sub_vector_D_updated[0]
                    D[0, 1] = D[1, 0]
                    D[2, 0] = sub_vector_D_updated[1]
                    D[0, 2] = D[2, 0]

                    D[0, 0] = D[0, 0]

                elif dim == 1:
                    D[0, 1] = sub_vector_D_updated[0]
                    D[1, 0] = D[0, 1]
                    D[2, 1] = sub_vector_D_updated[1]
                    D[1, 2] = D[2, 1]

                    D[1, 1] = D[1, 1]
                else:
                    # dim=2
                    D[0, 2] = sub_vector_D_updated[0]
                    D[2, 0] = D[0, 2]
                    D[1, 2] = sub_vector_D_updated[1]
                    D[2, 1] = D[1, 2]

                    D[2, 2] = D[2, 2]

                ################################# update R ###################################

                # compute the 2D rotation matrix after being embedded in 3D
                # NOTE: this can be done in hardware by the same method we applied for computing/updating D
                # here we use the fact the resulting 3D matrix has 0's so we use matrix multiplication instead ...
                R2 = np.array(
                    [
                        [cos_val_quant, -sign_tan2 * sin_val_quant],
                        [sign_tan2 * sin_val_quant, cos_val_quant],
                    ],
                    dtype=object,
                )
                R2_embed_in_3d = np.zeros((3, 3), dtype=object)

                R2_embed_in_3d[selection == 1] = R2.ravel()
                # old version
                # R2_embed_in_3d[dim,dim] = (2**NUM_BITS)-1
                # new version: to reduce the number of multiplications
                R2_embed_in_3d[dim, dim] = 2**NUM_BITS

                # NOTE: example for dim=1
                # if the computed rotation matrix using lookup table was |cos_val_quant  -sin_val_quant|
                #                                                        |sin_val_quant   cos_val_quant|
                #
                # and if dim=1 we embed it like
                #
                # |cos_val_quant              0            -sin_val_quant|
                # |0                  2**num_bits_lut                   0|
                # |sin_val_quant              0             cos_val_quant|
                #

                R_out = np.zeros((3, 3), dtype=object)

                for i in range(3):
                    for j in range(3):
                        update = 0
                        for k in range(3):
                            # update = (R[i,k] * R2_embed[k,j]) >> num_lookuptable_bits
                            update += R[i, k] * R2_embed_in_3d[k, j]

                            if abs(update) >= 2 ** (
                                NUM_BITS_ROTATION + NUM_BITS - 1 + ROT_EXTRA_BIT
                            ):
                                raise ValueError(
                                    "an over- or under-flow happened in rotation update!"
                                )

                        R_out[i, j] = update >> NUM_BITS

                        if abs(R_out[i, j]) >= 2 ** (
                            NUM_BITS_ROTATION - 1 + ROT_EXTRA_BIT
                        ):
                            raise ValueError(
                                "an over- or under-flow happened in rotation update!"
                            )

                # store the rotation matrix after update
                R = R_out

                # save the matrices
                C_list.append(np.copy(D))
                R_list.append(np.copy(R))

        # fill the list of returned value/stored states in case of early termination
        max_iter = NROUND * 3  # because  of 3 dimensions

        for i in range(max_iter - niter):
            C_list.append(np.copy(C_list[-1]))
            R_list.append(np.copy(R_list[-1]))

        # in the final step order the components such that the diagonal elements are
        C_last = C_list[-1]
        R_last = R_list[-1]

        R_last_sorted, C_last_sorted = self.sort_singular_values(
            R_last=R_last, C_last=C_last
        )

        # return the computed matrices and the final sorted ones

        return R_last_sorted, C_last_sorted

    # utility functions
    def selection_matrix(self, dim: int) -> np.ndarray:
        """this function returns a 3 x 3 0-1 valued matrix corresponding to the elements that need to be selected.

        Args:
            dim (int): the index of the element that needs to be removed. It is in the range: 0, 1, 2.

            Example:
                if dim = 0 the matrix will be as follows to remove the first row and column

                |0 0 0|
                |0 1 1|
                |0 1 1|


                if dim = 1 the matrix will be as follows to remove the second row and column

                |1 0 1|
                |0 0 0|
                |1 0 1|

        Returns:
            np.ndarray: the 3 x 3 selection matrix corresponding to the elements that need to be kept.
        """

        if dim < 0 or dim >= 3:
            raise ValueError(
                "the dropped index can be in the range {0, 1, 2} since we have 3 x 3 matrices."
            )

        selection = np.ones(3, dtype=np.int8)
        selection[dim] = 0

        selection = np.einsum("i, j -> ij", selection, selection)

        return selection

    @type_check
    def sort_singular_values(
        self, R_last: np.ndarray, C_last: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """this module takes the almost-diagonal covariance matrix and ist corresponding rotation and sorts it according to diagonal elements.

        Args:
            R_last (np.ndarray): the rotation matrix corresponding to the last covariance matrix.
            C_last (np.ndarray): the last covariance matrix (almost diagonal one).

        Returns:
            Tuple(np.ndarray, np.ndarray): the sorted rotation and covariance matrix (almost diagonal).
        """

        # take the diagonal elements and sort them in a descending order (SVD convention)
        order = np.argsort(np.diag(C_last))[-1::-1]

        # modify the rotation matrix
        R_modified = np.copy(R_last)
        R_modified = R_modified[:, order]

        # modify covariance matrix
        C_modified = np.copy(C_last)
        C_modified = C_modified[order, :]
        C_modified = C_modified[:, order]

        return R_modified, C_modified

    # functions for evaluating the performance of JSVD
    @type_check
    def distance_subspace_metric(self, C_list: List[np.ndarray]) -> np.ndarray:
        """This module computes how the input covariance matrices close to diagonal.
        We use the subspace metric.

        Args:
            C_list (List[np.ndarray]): a list containing the PSD 3 x 3 matrices

        Returns:
            np.ndarray: the subspace metric for each matrix in the list.
        """

        subspace_metric = []

        for C in C_list:
            U, _, _ = np.linalg.svd(C.astype(np.float64))

            # compute abs value
            U = np.abs(U)

            # fix the possible permutation
            max_val_index = np.argmax(U, axis=1)
            permutation = np.zeros((3, 3))
            for i in range(3):
                permutation[i, max_val_index[i]] = 1

            U = U @ permutation.T

            subspace_metric.append(np.linalg.norm(U - np.eye(U.shape[0])))

        subspace_metric = np.asarray(subspace_metric)

        return subspace_metric

    @type_check
    def distance_offdiagonal_metric(self, C_list: List[np.ndarray]) -> np.ndarray:
        """This module computes how the input covariance matrices close to diagonal.
        We use the norm of the off-diagonal elements as a distance metric.

        Args:
            C_list (List[np.ndarray]): a list containing the PSD 3 x 3 matrices

        Returns:
            np.ndarray: the off-diagonal metric for each matrix in the list.
        """

        subspace_metric = []

        for C in C_list:
            metric = np.linalg.norm(C - np.diag(np.diag(C))) / np.linalg.norm(C)
            subspace_metric.append(metric)

        subspace_metric = np.asarray(subspace_metric)

        return subspace_metric
