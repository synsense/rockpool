def test_SubSpace():
    # required packages
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import lfilter

    from imu_preprocessing.subspace_tracking import SubSpace
    from imu_preprocessing.quantizer import Quantizer

    fs = 200  # sampling rate of IMU sensors

    # a sinusoid signal
    f0 = 30
    amp = 2.0
    T = 1000

    phases = np.random.rand(3) * 2 * np.pi
    sig_in = amp * np.sin(
        (2 * np.pi * f0 / fs * np.arange(T)).reshape(1, -1) + phases.reshape(3, 1)
    )
    sig_in[0, :] += 10 * amp

    # quantize the input signal: assume that the amplitude fills all the bits
    # normalize the signal amplitude
    Q = Quantizer()
    num_bits_in = 16

    sig_in_quant = Q.quantize(
        sig_in=sig_in,
        scale=0.999 / (np.max(np.abs(sig_in))),
        num_bits=num_bits_in,
    )

    # how much time averaging we would like to have
    avg_window_duration = 50e-3  # in milli-sec
    avg_window_len = int(avg_window_duration * fs)

    # closest bitshift size for averaging filter implementation
    avg_bitshift = int(np.log2(avg_window_len) + 1)

    # number of bits devoted to storing the multiplication
    num_bits_multiplier = 2 * num_bits_in

    # bits devoted to the implementation of high-prec filter
    bits_highprec_filter = num_bits_multiplier + avg_bitshift

    # subspace module
    subspace = SubSpace(
        num_bits_in=num_bits_in,
        num_bits_multiplier=num_bits_multiplier,
        num_bits_highprec_filter=bits_highprec_filter,
        num_avg_bitshift=avg_bitshift,
    )

    # compute the output covariance matrix estimate
    C_list, C_highprec_list = subspace(sig_in_quant)

    # flatten the matrices into 3 x 3 --> 9 to see the components
    C_list_comp = C_list.reshape(-1, 9, 1).squeeze()
    C_highprec_list_comp = C_highprec_list.reshape(-1, 9, 1).squeeze()

    # apply normalization
    C_list_comp_norm = C_list_comp / np.max(np.abs(C_list_comp))
    C_highprec_list_comp_norm = C_highprec_list_comp / np.max(
        np.abs(C_highprec_list_comp)
    )

    # compute the true covariance matrix without any quantization and flatten it
    filt_b = [1 / (2**avg_bitshift)]
    filt_a = [1, -(1 - 1 / (2**avg_bitshift))]

    true_covariance_comp = lfilter(
        filt_b,
        filt_a,
        np.asarray(
            [np.outer(sig_sample, sig_sample).ravel() for sig_sample in sig_in.T]
        ),
        axis=0,
    )

    # normalize
    true_covariance_comp_norm = true_covariance_comp / np.max(
        np.abs(true_covariance_comp)
    )

    # which index to plot: 1 -> (0,0), 2 -> (0,1), ..., 9 -> (2,2)
    index_to_plot = 0

    # -- print the results

    # high-precision version
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(611)
    plt.plot(C_highprec_list_comp_norm, "blue")
    plt.ylabel("high-prec covaraince")
    plt.grid(True)
    plt.title(
        f"avg bit-shift:{avg_bitshift}, avg window:{2**avg_bitshift}, bits-in:{num_bits_in}, "
        + f"bits-mult:{num_bits_multiplier}, bits-high-prec:{bits_highprec_filter}, bits-out:{bits_highprec_filter-avg_bitshift}"
    )

    plt.subplot(612, sharex=ax)
    plt.plot(C_highprec_list_comp_norm, "blue")
    plt.plot(true_covariance_comp_norm, "gray")
    plt.ylabel("normalized high-prec")
    plt.legend(["quantized (blue) and inf-bit (gray)"])
    plt.grid(True)

    plt.subplot(613, sharex=ax)
    plt.plot(C_highprec_list_comp_norm[:, index_to_plot], "blue")
    plt.plot(true_covariance_comp_norm[:, index_to_plot], "gray")
    plt.ylabel("normalized high-prec")
    plt.legend(
        [
            f"quantized (index={index_to_plot}/9)",
            f"original unquantized (index={index_to_plot}/9)",
        ]
    )
    plt.grid(True)

    # low-precision version
    plt.subplot(614, sharex=ax)
    plt.plot(C_list_comp_norm, "blue")
    plt.ylabel("low-prec covariance")
    plt.grid(True)

    plt.subplot(615, sharex=ax)
    plt.plot(C_list_comp_norm, "blue")
    plt.plot(true_covariance_comp_norm, "gray")
    plt.ylabel("normalized low-prec")
    plt.legend(["quantized (blue) and inf-bit (gray)"])
    plt.grid(True)

    plt.subplot(616, sharex=ax)
    plt.plot(C_list_comp_norm[:, index_to_plot], "blue")
    plt.plot(true_covariance_comp_norm[:, index_to_plot], "gray")
    plt.xlabel("time")
    plt.ylabel("normalized low-prec")
    plt.legend(
        [
            f"quantized (index={index_to_plot}/9)",
            f"original unquantized (index={index_to_plot}/9)",
        ]
    )
    plt.grid(True)

    plt.show()


def test_SubSpace2():
    # required packages
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import lfilter

    from imu_preprocessing.subspace_tracking import SubSpace
    from imu_preprocessing.quantizer import Quantizer

    """test for subspace module."""
    # sampling rate of IMU sensors
    fs = 200

    # a sinusoid signal
    f0 = 30
    amp = 2.0
    T = 1000

    phases = np.random.rand(3) * 2 * np.pi
    sig_in = amp * np.sin(
        (2 * np.pi * f0 / fs * np.arange(T)).reshape(1, -1) + phases.reshape(3, 1)
    )
    sig_in[0, :] += 10 * amp

    # quantize the input signal: assume that the amplitude fills all the bits
    # normalize the signal amplitude
    Q = Quantizer()
    num_bits_in = 16

    sig_in_quant = Q.quantize(
        sig_in=sig_in,
        scale=0.999 / (np.max(np.abs(sig_in))),
        num_bits=num_bits_in,
    )

    # bits devoted to the multipliers:
    #       - typically `2 x bits_in + 1` to have a full precision multiplier.
    #       - but smaller values are also possible at the cost of some loss in precision.
    bits_multiplier = 8

    # how much time averaging we would like to have
    avg_window_duration = 50e-3  # in milli-sec
    avg_window_len = int(avg_window_duration * fs)

    # closest bitshift size for averaging filter implementation
    avg_bitshift = int(np.log2(avg_window_len) + 1)

    # bits devoted to the implementation of high-prec filter
    bits_highprec_filter = bits_multiplier + avg_bitshift

    # subspace module
    subspace = SubSpace(
        num_bits_in=num_bits_in,
        num_bits_multiplier=bits_multiplier,
        num_bits_highprec_filter=bits_highprec_filter,
        num_avg_bitshift=avg_bitshift,
    )

    # compute the output covariance matrix estimate
    C_list, C_highprec_list = subspace(sig_in_quant)

    # flatten the matrices into 3 x 3 --> 9 to see the components
    C_list_comp = C_list.reshape(-1, 9, 1).squeeze()
    C_highprec_list_comp = C_highprec_list.reshape(-1, 9, 1).squeeze()

    # apply normalization
    C_list_comp_norm = C_list_comp / np.max(np.abs(C_list_comp))
    C_highprec_list_comp_norm = C_highprec_list_comp / np.max(
        np.abs(C_highprec_list_comp)
    )

    # compute the true covariance matrix without any quantization and flatten it
    filt_b = [1 / (2**avg_bitshift)]
    filt_a = [1, -(1 - 1 / (2**avg_bitshift))]

    true_covariance_comp = lfilter(
        filt_b,
        filt_a,
        np.asarray(
            [np.outer(sig_sample, sig_sample).ravel() for sig_sample in sig_in.T]
        ),
        axis=0,
    )

    # normalize
    true_covariance_comp_norm = true_covariance_comp / np.max(
        np.abs(true_covariance_comp)
    )

    # which index to plot: 1 -> (0,0), 2 -> (0,1), ..., 9 -> (2,2)
    index_to_plot = 0

    # -- print the results

    # high-precision version
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(611)
    plt.plot(C_highprec_list_comp_norm, "blue")
    plt.ylabel("high-prec covaraince")
    plt.grid(True)
    plt.title(
        f"covariance estimation for averaging window size = {2**avg_bitshift} samples: blue: qunatized, gray: unquantized"
    )

    plt.subplot(612, sharex=ax)
    plt.plot(C_highprec_list_comp_norm, "blue")
    plt.plot(true_covariance_comp_norm, "gray")
    plt.ylabel("normalized high-prec")
    plt.grid(True)

    plt.subplot(613, sharex=ax)
    plt.plot(C_highprec_list_comp_norm[:, index_to_plot], "blue")
    plt.plot(true_covariance_comp_norm[:, index_to_plot], "gray")
    plt.ylabel("normalized high-prec")
    plt.legend(
        [
            f"quantized (index={index_to_plot}/9)",
            f"original unquantized (index={index_to_plot}/9)",
        ]
    )
    plt.grid(True)

    # low-precision version
    plt.subplot(614, sharex=ax)
    plt.plot(C_list_comp_norm, "blue")
    plt.ylabel("low-prec covariance")
    plt.grid(True)

    plt.subplot(615, sharex=ax)
    plt.plot(C_list_comp_norm, "blue")
    plt.plot(true_covariance_comp_norm, "gray")
    plt.ylabel("normalized low-prec")
    plt.grid(True)

    plt.subplot(616, sharex=ax)
    plt.plot(C_list_comp_norm[:, index_to_plot], "blue")
    plt.plot(true_covariance_comp_norm[:, index_to_plot], "gray")
    plt.xlabel("time")
    plt.ylabel("normalized low-prec")
    plt.legend(
        [
            f"quantized (index={index_to_plot}/9)",
            f"original unquantized (index={index_to_plot}/9)",
        ]
    )
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    test_SubSpace()
    print("end of simulation of subspace module!")
