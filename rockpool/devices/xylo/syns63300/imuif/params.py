"""
Hard-set parameters for the Xylo IMU HDK
"""

__all__ = [
    "B_WORST_CASE",
    "CLOCK_RATE",
    "COV_EXTRA_BIT",
    "FILTER_ORDER",
    "N_CHANNEL",
    "NROUND",
    "NUM_ANGLES",
    "NUM_BITS",
    "NUM_BITS_COVARIANCE",
    "NUM_BITS_HIGHPREC_FILTER_BASE",
    "NUM_BITS_MULTIPLIER",
    "NUM_BITS_ROTATION",
    "NUM_BITS_SPIKE",
    "ROT_EXTRA_BIT",
]

B_WORST_CASE = 9
"""Number of additional bits devoted to storing filter taps such that no over- and under-flow can happen"""

CLOCK_RATE = 200
"""Fixed computation step rate of 200Hz for Xylo IMU"""

COV_EXTRA_BIT = 2
"""The components of the covariance can enlarger by a factor 3 (at most), thus, an additional register size of 2"""

DEFAULT_FILTER_BANDS = [
    (1.0, 2.0, 200.0),
    (2.0, 4.0, 200.0),
    (4.0, 8.0, 200.0),
    (8.0, 16.0, 200.0),
    (16.0, 32.0, 200.0),
    (1.0, 2.0, 200.0),
    (2.0, 4.0, 200.0),
    (4.0, 8.0, 200.0),
    (8.0, 16.0, 200.0),
    (16.0, 32.0, 200.0),
    (1.0, 2.0, 200.0),
    (2.0, 4.0, 200.0),
    (4.0, 8.0, 200.0),
    (8.0, 16.0, 200.0),
    (16.0, 32.0, 200.0),
]
"""Default passbands for the IMUIF Filters"""

FILTER_ORDER = 1
"""HARD_CODED: Filter order of the Xylo-IMU filters"""

N_CHANNEL = 15
"""Fixed number of channels employed"""

NROUND = 4
"""number of round rotation computation and update is done over all 3 axes/dims"""

NUM_ANGLES = 64
"""number of angles in lookup table"""

NUM_BITS = 16
"""number of bits in the continuous signal(input, obtained after rotation removal). We assume a sign magnitude format."""

NUM_BITS_COVARIANCE = 31
"""number of bits used for the covariance matrix"""

NUM_BITS_HIGHPREC_FILTER_BASE = 31
"""number of bits devoted to computing the high-precision filter (to avoid dead-zone effect). NOTE: This is the base value. The actual value is computed as `NUM_BITS_HIGHPREC_FILTER_BASE + num_avg_bitshift`"""

NUM_BITS_MULTIPLIER = 31
"""number of bits devoted to computing [x(t) x(t)^T]_{ij}. If less then needed, the LSB values are removed"""

NUM_BITS_ROTATION = 32
"""number of bits devoted for implementing rotation matrix"""

NUM_BITS_SPIKE = 4
"""number of bits devoted to storing the output spike encoding"""

ROT_EXTRA_BIT = 1
"""Rotation can expand at most by a factor during the multiplication, thus, an additional register size of 1"""
