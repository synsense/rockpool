"""Test JSVD computation module"""


def test_JSVD_low_rank_gravity():
    from rockpool.devices.xylo.imu.preprocessing import (
        RotationLookUpTable,
        JSVD,
        Quantizer,
    )
    import numpy as np

    # - Init JSVD module
    jsvd = JSVD(
        lookuptable=RotationLookUpTable(num_angles=64, num_bits=16),
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
            scale=0.999 / np.max(np.abs(cov_in)), num_bits=jsvd.num_bits_covariance
        )
        cov_quant, _, _ = Q(cov_in)

        # compute the covariance matrix via SVD
        U, _, _ = np.linalg.svd(cov_quant.astype(np.float64))

        # compute the covariance matrix using JSVD
        (R_last, C_last), _, _ = jsvd.evolve(cov_quant)

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

        deviation = np.linalg.norm(
            corr_ordered_signcorrected - np.eye(3)
        ) / np.linalg.norm(np.eye(3))

        # Test
        max_deviation = 0.05
        assert (
            deviation < max_deviation
        ), "JSVD failed to estimate the covariance matrix correctly"
        deviation_list.append(deviation)

    print(deviation_list)


if __name__ == "__main__":
    test_JSVD_low_rank_gravity()
