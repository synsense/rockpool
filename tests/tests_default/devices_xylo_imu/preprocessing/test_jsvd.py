"""Test JSVD computation module"""


def test_JSVD_correct_angle():
    from rockpool.devices.xylo.imu.preprocessing import (
        RotationLookUpTable,
        JSVD,
        Quantizer,
    )
    import matplotlib.pyplot as plt
    import numpy as np

    # --- create a suitable lookup table
    num_angles = 64
    num_bits = 16

    lut = RotationLookUpTable(num_angles=num_angles, num_bits=num_bits)

    # JSVD module
    num_bits_covariance = 32
    num_bits_rotation = 32
    nround = 4
    jsvd = JSVD(
        lookuptable=lut,
        num_bits_covariance=num_bits_covariance,
        num_bits_rotation=num_bits_rotation,
        nround=nround,
    )

    # test setting
    num_tests = 10
    max_deviation = 0.05

    num_pass = 0
    deviation_list = []
    for test in range(num_tests):
        print(
            f"\ncorrelation test between JSVD and SVD: num-angles:{num_angles}, num-bits:{num_bits}, num-bits-covariance:{num_bits_covariance}, num-bits-rotation:{num_bits_rotation}, TEST #{test}\n"
        )

        T = 100
        gravity = 10
        random_rotation, _, _ = np.linalg.svd(np.random.randn(3, 3))

        # several variation of data
        data_types = ["low-rank", "gravity-rotated", "noise-gravity-rotated"]

        data_type = data_types[-1]

        if data_type == "gravity-rotated":
            x = np.zeros((3, T))
            x[0, :] += gravity
            x = random_rotation @ x

        elif data_type == "noise-gravity-rotated":
            x = np.random.randn(3, T)
            x[0, :] += gravity
            x = random_rotation @ x
            # x=np.tile(x[:,0].reshape(-1,1), T)

        elif data_type == "low-rank":
            x = np.random.randn(3, T)
            x[0, :] += gravity
            x = random_rotation @ x
            U, D, V = np.linalg.svd(x)
            D[0] = 0

            x_low_rank = U @ np.diag(D) @ V[:3, :]
            x = x_low_rank

        else:
            assert 1 == 0, "a data type is needed for investigation!"

        C = 1 / T * (x @ x.T)

        # quantize C
        num_bits_covariance = 32
        quantizer = Quantizer(
            shape=None,
            scale=0.999 / np.max(np.abs(C)),
            num_bits=num_bits_covariance,
        )

        C_q, _, _ = quantizer(C)

        # compute the covariance matrix via SVD
        U, _, _ = np.linalg.svd(C_q.astype(np.float64))

        # compute the covariance matrix using JSVD2
        (R_last, C_last), _, _ = jsvd.evolve(C_q)

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

        deviation_list.append(deviation)

        print("*" * 100)
        if deviation < max_deviation:
            print(
                f"test #{test} passed => relative deviation: {deviation} below the permitted threshold: {max_deviation}!"
            )
            num_pass += 1
        else:
            print(
                f"test #{test} did not pass => relative deviation: {deviation} above the permitted threshold: {max_deviation}!"
            )
        print("*" * 100)

    # compute the last data sample after rotation removal
    R_last_normalized = R_last / np.linalg.norm(R_last[:]) * np.sqrt(3)
    x_rotation_removed = R_last_normalized.T @ x

    U, D, _ = np.linalg.svd(x)
    x_rotation_removed_exact = U.T @ x

    # result of analysis
    print("\n\n")
    print("*" * 100)
    print("*" * 100)
    print(
        f"{num_pass} out of {num_tests} were passed successfully with maximum relative deviation {max_deviation}!"
    )
    print("*" * 100)
    print("*" * 100)

    # plot the results
    plt.figure(figsize=(12, 12))
    plt.subplot(211)
    plt.hist(deviation_list, 100)
    plt.xlabel("relative deviation in JSVD")
    plt.ylabel("histogram")
    plt.grid(True)
    plt.title("statistical analysis of the behavior of JSVD")

    plt.subplot(212)
    plt.semilogy(np.sort(deviation_list), np.linspace(1, 0, len(deviation_list)))
    plt.xlabel("relative deviation")
    plt.ylabel("CCDF")
    plt.ylim([0.001, 1])
    plt.plot([max_deviation, max_deviation], [0.001, 1], "r", linewidth=2)
    plt.title(f"{num_pass} / {num_tests} tests below the max-deviation:{max_deviation}")

    plt.grid(True)

    plt.show(block=False)

    # plot the last data sample
    plt.figure(figsize=(12, 12))
    plt.subplot(311)
    plt.plot(x.T)
    plt.grid(True)
    plt.ylabel("channels")
    plt.title("before rotation removal")

    plt.subplot(312)
    plt.plot(x_rotation_removed.T)
    plt.grid(True)
    plt.ylabel("channels")
    plt.title("after JSVD rotation removal")

    plt.subplot(313)
    plt.plot(x_rotation_removed_exact.T)
    plt.grid(True)
    plt.xlabel("sample index")
    plt.ylabel("channels")
    plt.title("after EXACT rotation removal")

    plt.show()


if __name__ == "__main__":
    test_JSVD_correct_angle()
    print("end of test for JSVD mdoule!")
