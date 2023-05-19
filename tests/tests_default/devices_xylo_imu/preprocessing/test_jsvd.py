"""Test JSVD computation module"""


def test_import():
    from rockpool.devices.xylo.imu.preprocessing.jsvd import JSVD


def test_type_check():
    import numpy as np
    import pytest
    from numpy.testing import assert_array_equal

    from rockpool.devices.xylo.imu.preprocessing.utils import type_check

    # define a class
    @type_check
    def myfun(var_in):
        return var_in

    # two input arrays
    arr_pass = np.array([1, 2, 3], dtype=object)
    arr_fail = np.array([1.2, 3.4, 5.6, 10], dtype=np.float64)

    with pytest.raises(ValueError):
        myfun(arr_fail)

    assert_array_equal(myfun(arr_pass), arr_pass)


def test_JSVD_low_rank_gravity():
    import numpy as np
    from numpy.testing import assert_almost_equal

    from rockpool.devices.xylo.imu.preprocessing import JSVD, Quantizer

    # - Init JSVD module
    jsvd = JSVD(
        num_angles=64,
        num_bits_lookup=16,
        num_bits_covariance=32,
        num_bits_rotation=32,
        nround=4,
    )

    num_test = 10
    np.random.seed(2023)

    deviation_list = []
    for _ in range(num_test):
        # - Create a random rotation matrix
        T = 100
        num_ch = 3
        gravity = 10
        random_rotation, _, _ = np.linalg.svd(np.random.randn(num_ch, num_ch))

        # several variation of data
        x_in = np.random.randn(num_ch, T)
        x_in[0, :] += gravity
        x_in = random_rotation @ x_in
        U, D, V = np.linalg.svd(x_in)
        D[0] = 0

        x_in_low_rank = U @ np.diag(D) @ V[:num_ch, :]

        cov_in = 1 / T * (x_in_low_rank @ x_in_low_rank.T)

        # quantize C
        Q = Quantizer(
            shape=num_ch,
            scale=0.999 / np.max(np.abs(cov_in)),
            num_bits=jsvd.num_bits_covariance,
        )
        cov_quant, _, _ = Q(cov_in)

        # compute the covariance matrix via SVD
        U, _, _ = np.linalg.svd(cov_quant[0].astype(np.float64))

        # compute the covariance matrix using JSVD
        R_last, C_last = jsvd(cov_quant[0])

        # compute the correlation
        corr = R_last.T @ U
        corr = corr / np.max(np.abs(corr))

        max_index = np.argmax(np.abs(corr), axis=1)
        permutation = np.zeros((3, 3))
        for i in range(3):
            permutation[i, max_index[i]] = 1

        # compute the correlation after permutation removal
        corr_ordered = corr @ permutation.T

        # correct the signs as well
        corr_ordered_signcorrected = corr_ordered @ np.diag(
            np.sign(np.diag(corr_ordered))
        )

        assert_almost_equal(corr_ordered_signcorrected, np.eye(3), decimal=1)
