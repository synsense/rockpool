# -----------------------------------------------------------
# This module takes the 3 x 3 covariance matrix from the IMU subspace estimation module and
# computes the 3 x 3 rotation matrix and 3 x 3 diagonal matrix.
#
# Version 2:
#       (i)     We implement the idea proposed by @Huaqiu Zhang to compute
#       R.T @ C @ R in one-shot rather than doing it in two steps
#       with 2 matrix multiplication.
#       (ii)    We apply infinite-bit approximation, which is valid when number of angles in lookup table is large enough.
#       (iii)   This yields a higher precision in implementation of JSVD.
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 12.09.2022
# -----------------------------------------------------------

# required packages
from asyncio import selector_events
from multiprocessing.sharedctypes import Value
import numpy as np
import warnings
from imu_preprocessing.util.bucket_decorator import bucket_decorator

from imu_preprocessing.util.type_decorator import type_check


class RotationLookUpTable2:
    def __init__(self, num_angles, num_bits):
        """this module builds a lookup table for JSVD algorithm.
        The range of angles for lookup tables are [0, 45] and they are quantized into `num_angles` angle bins.
        The data for each angle is quantized/trancated into `num_bits` bits.

        Args:
            num_angles (int): number of angles in lookup table.
            num_bits (int): number of bits used for quantizing the lookup table.
        """
        self.num_angles = num_angles
        self.num_bits = num_bits

        self._compute_lookup_table()

    def _compute_lookup_table(self):
        """
        this module biulds the lookup table.
        To make sure that the lookup table has a good precision we do the following:
            (i)     we always work with angles in the range [0, 45] degrees.
            (ii)    since the tan(2 theta) ranges in [0, infty] to avoid loss of precision,
                        - we use tan(2 theta) for theta in [0, 22.5]
                        - we use cot(2 theta) for theta in [22.5, 45].
            (iii)   the lookup table consists of
                        - the tan2-cot2 value in the [0, 22.5] and [22.5,45]
                        - sin and cos in the whole range [0, 45]
                        - 1/sin(2 theta) in the range [22.5, 45]
                        - 1/cos(2 theta) in the range [0, 22.5]

            (iv)       the row of the lookup table used depends on how abs(2b) comparse with abs(a-c) in the 2D submatrix
                       |a  b|
                       |b  c|

        """

        # in the new method, we are able to explore the whole range of angles in [0, 45]
        max_angle_degree = 45
        max_angle_radian = max_angle_degree / 180 * np.pi

        self.angles_radian = np.linspace(0, max_angle_radian, self.num_angles)
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
        EPS = 10e-30

        self.inv_2cos2_vals = 1 / (2 * np.cos(2 * self.angles_radian) + EPS)
        self.inv_2cos2_vals[ind_low_angle == False] = np.nan

        self.inv_2sin2_vals = 1 / (2 * np.sin(2 * self.angles_radian) + EPS)
        self.inv_2sin2_vals[ind_low_angle == True] = np.nan

        # compute the quanized values:
        #   (i)     switch to python-version with inifinite number of bits
        #   (ii)    add EPS to make sure that all the values are smaller than 1 and can fit in num_bits after quantization
        # NOTE: all the values are positive in the lookup table
        self.tan2_vals_quantized = np.asarray(
            [
                int(el / (1 + EPS)) if not np.isnan(el) else np.nan
                for el in (self.tan2_vals * 2**self.num_bits)
            ],
            dtype=object,
        )
        self.cot2_vals_quantized = np.asarray(
            [
                int(el / (1 + EPS)) if not np.isnan(el) else np.nan
                for el in (self.cot2_vals * 2**self.num_bits)
            ],
            dtype=object,
        )
        self.sin_vals_quantized = np.asarray(
            [
                int(el / (1 + EPS)) if not np.isnan(el) else np.nan
                for el in ((2**self.num_bits - 1) * self.sin_vals)
            ],
            dtype=object,
        )
        self.cos_vals_quantized = np.asarray(
            [
                int(el / (1 + EPS)) if not np.isnan(el) else np.nan
                for el in ((2**self.num_bits - 1) * self.cos_vals)
            ],
            dtype=object,
        )

        self.inv_2cos2_vals_quantized = np.asarray(
            [
                int(el) if not np.isnan(el) else np.nan
                for el in (2**self.num_bits * self.inv_2cos2_vals)
            ],
            dtype=object,
        )
        self.inv_2sin2_vals_quantized = np.asarray(
            [
                int(el) if not np.isnan(el) else np.nan
                for el in (2**self.num_bits * self.inv_2sin2_vals)
            ],
            dtype=object,
        )

    @bucket_decorator
    def find_angle(self, a, b, c):
        """this module computes the best angle in lookup table that matches the given `a`, `b`, `c` parametets.
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
            row_index = self.num_angles - 1

        else:
            # here we follow one of these two metheds
            if abs(2 * b) <= abs(a - c):
                # in this case `0<= tan(2 theta) <= 1` and angle is in the range [0, 22.5] degrees.
                # NOTE: here we should use the `tan(2 theta)` values to find the corresponding row of the lookup table.
                # we simply do the following:
                #   (i)     we count the number of rows with  tan(2 theta) * abs(a-c) <= abs(2*b),
                #   (ii)    we do this to find the corresponding row of the lookup table.
                row_index = np.sum(
                    [
                        el * abs(a - c) <= abs(2 * b) * 2**self.num_bits
                        if not np.isnan(el)
                        else False
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
                        el * abs(2 * b) >= abs(a - c) * 2**self.num_bits
                        if not np.isnan(el)
                        else True
                        for el in self.cot2_vals_quantized
                    ]
                )

                if row_index == self.num_angles:
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

    def print_table(self, format: str = "dec", report=True):
        """this module prints the lookup table in a given format

        Args:
            format (str): 'bin' or 'hex'
            report (bool): print the table.

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

    def __str__(self):
        """
        return the decimal format of the lookuptable.
        """
        return self.print_table(
            format="dec",
            report=False,
        )
