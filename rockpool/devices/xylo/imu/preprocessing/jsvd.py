"""
This module takes the 3 x 3 covariance matrix from the IMU subspace estimation module and
computes the 3 x 3 rotation matrix and 3 x 3 diagonal matrix.

    (i)     Compute R.T @ C @ R in one-shot rather than doing it in two steps with 2 matrix multiplication.
    (ii)    Apply infinite-bit approximation, which is valid when number of angles in lookup table is large enough.
    (iii)   This yields a higher precision in implementation of JSVD.
"""
import warnings
from functools import wraps
from typing import List, Tuple

import numpy as np

from rockpool.devices.xylo.imu.preprocessing.lookup import RotationLookUpTable
from rockpool.nn.modules.module import Module

__all__ = ["JSVD"]


def type_check(func):
    """Type-check decorator for IMU python simulation to make sure that all the input data are of type `python.object`.
    This assures that the hardware and software will behave the same for all register sizes.

    Args:
        func (Callable): the function to be decorated.
    """

    def verify(input: List[np.ndarray]) -> None:
        if isinstance(input, list):
            if len(input) != 0:
                for el in input:
                    type_check(el)

        elif isinstance(input, np.ndarray):
            if input.dtype != object or type(input.ravel()[0]) != int:
                raise TypeError(
                    f"The elements of the following variable are not of type `python.object` integer. This may cause mismatch between hardware and python implementation."
                    + f"problem with the following variable:\n{input}\n"
                    + f"To solve the problem make sure that all the arrays have `dtype=object`. You can use `Quantizer` class in `quantizer.py` module."
                )

    @wraps(func)
    def inner_func(*args, **kwargs):
        for arg in args:
            verify(arg)

        for key in kwargs:
            verify(kwargs[key])

        return func(*args, **kwargs)

    return inner_func


class JSVD(Module):
    def __init__(
        self,
        lookuptable: RotationLookUpTable,
        num_bits_covariance: int,
        num_bits_rotation: int,
        nround: int = 4,
    ) -> None:
        """Runs Jaccobi SVD algorithm in FPGA precision.
        this is the 2nd version of the algorithm and used joint matrix multiplication in order to reduce the
        number of multiplication operations.

        Args:
            lookuptable (RotationLookUpTable): lookup table used for computation.
            num_bits_covariance (int): number of bits used for the covariance matrix.
            num_bits_rotation (int): number of bits devoted for implementing rotation matrix.
            nround (int): number of round rotation computation and update is done over all 3 axes/dims.
        """

        self.lookuptable = lookuptable
        self.num_bits_covariance = num_bits_covariance
        self.num_bits_rotation = num_bits_rotation
        self.nround = nround

    @type_check
    def evolve(
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
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                R_list (List[np.ndarray]): list of rotation matrices.
                C_list (List[np.ndarray]): list of covariance matrices.
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

        if np.max(np.abs(C_in)) >= 2 ** (self.num_bits_covariance - 1):
            raise ValueError(
                f"The input covariance matrix does not fit in the {self.num_bits_covariance} bits assigned to it! If needed, apply quantization to the covariance to truncate it!"
            )

        # --- initialization and history of computation
        COV_EXTRA_BIT = 2  # because the components of the covariance can enlarger by a factor 3 (at most), thus, an additional register size of 2
        ROT_EXTRA_BIT = 1  # rotation can expand at most by a factor during the multiplication, thus, an additional register size of 1

        # estimated covariance matrices
        C_list = [C_in]

        # estimated rotation matrices
        R_list = [(2 ** (self.num_bits_rotation - 1) - 1) * np.eye(3, dtype=object)]

        R = np.copy(R_list[0])
        D = np.copy(C_list[0])

        niter = 0

        for n_round in range(self.nround):
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
                        ((a + c) << self.lookuptable.num_bits) // 2
                        + rotation_correction_large_angles
                    ) >> self.lookuptable.num_bits
                    c_new = (
                        ((a + c) << self.lookuptable.num_bits) // 2
                        - rotation_correction_large_angles
                    ) >> self.lookuptable.num_bits

                    if abs(a_new) >= 2 ** (
                        self.num_bits_covariance - 1 + COV_EXTRA_BIT
                    ) or abs(c_new) >= 2 ** (
                        self.num_bits_covariance - 1 + COV_EXTRA_BIT
                    ):
                        raise ValueError(
                            "over- or under-flow happened in updating the almost-diagonal matrix D!"
                        )

                else:
                    # SMALL ANGLES: tan(2 theta) <= 1.0

                    # NOTE: sign modification is NOT needed here since we have `COS(2 theta)` and it is always positive even for negative angles.
                    a_new = (
                        ((a + c) << self.lookuptable.num_bits) // 2
                        + rotation_correction_small_angles
                    ) >> self.lookuptable.num_bits
                    c_new = (
                        ((a + c) << self.lookuptable.num_bits) // 2
                        - rotation_correction_small_angles
                    ) >> self.lookuptable.num_bits

                    if abs(a_new) >= 2 ** (
                        self.num_bits_covariance - 1 + COV_EXTRA_BIT
                    ) or abs(c_new) >= 2 ** (
                        self.num_bits_covariance - 1 + COV_EXTRA_BIT
                    ):
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
                ) >> self.lookuptable.num_bits
                sub_vector_D_updated[1] = (
                    -sin_val_quant * sign_tan2 * sub_vector_D[0]
                    + cos_val_quant * sub_vector_D[1]
                ) >> self.lookuptable.num_bits

                if np.abs(sub_vector_D[0]) >= 2 ** (
                    self.num_bits_covariance - 1 + COV_EXTRA_BIT
                ) or np.abs(sub_vector_D[1]) >= 2 ** (
                    self.num_bits_covariance - 1 + COV_EXTRA_BIT
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
                # R2_embed_in_3d[dim,dim] = (2**self.lookuptable.num_bits)-1
                # new version: to reduce the number of multiplications
                R2_embed_in_3d[dim, dim] = 2**self.lookuptable.num_bits

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
                                self.num_bits_rotation
                                + self.lookuptable.num_bits
                                - 1
                                + ROT_EXTRA_BIT
                            ):
                                raise ValueError(
                                    "an over- or under-flow happened in rotation update!"
                                )

                        R_out[i, j] = update >> self.lookuptable.num_bits

                        if abs(R_out[i, j]) >= 2 ** (
                            self.num_bits_rotation - 1 + ROT_EXTRA_BIT
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
        max_iter = self.nround * 3  # because  of 3 dimensions

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

        if record == True:
            record_dict = {"C_list": C_list, "R_list": R_list}
        else:
            record_dict = {}

        state_dict = {}
        out = (R_last_sorted, C_last_sorted)

        return out, state_dict, record_dict

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

    def __str__(self) -> str:
        string = (
            "JSVD module for computing the rotation in IMU dataset:"
            + f"number of bits used for covariance computation: {self.num_bits_covariance}\n"
            + f"number of bits used for rotation computation and storage: {self.num_bits_rotation}\n\n"
            + f"rotation lookuptable used for angle estimation:\n{self.lookuptable}"
        )
        return string


if __name__ == "__main__":
    from quantizer import Quantizer

    num_angles = 64
    num_bits = 16
    lut = RotationLookUpTable(num_angles=num_angles, num_bits=num_bits)

    # create the JSVD module
    jsvd = JSVD(
        lookuptable=lut,
        num_bits_covariance=num_bits + 10,
        num_bits_rotation=num_bits + 10,
    )

    # create the covariance matrix
    C = np.random.rand(3, 3)
    C = C @ C.T

    # create a quantizer module
    quantizer = Quantizer(None, scale=0.999 / np.max(np.abs(C)), num_bits=num_bits)
    C, _, _ = quantizer(C)

    # run the JSVD module
    R_list, C_list, R_last, C_last = jsvd.evolve(C)

    # print the results
    print(f"the input covariance matrix:\n{C}\n")
    print(f"the final covariance matrix:\n{C_last}\n")
    print(f"the final rotation matrix:\n{R_last}\n")
    print(f"the list of covariance matrices:\n{C_list}\n")
    print(f"the list of rotation matrices:\n{R_list}\n")
