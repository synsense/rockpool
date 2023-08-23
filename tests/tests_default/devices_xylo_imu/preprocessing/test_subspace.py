def test_import():
    from rockpool.devices.xylo.imu.imuif_submod import SubSpace

    assert SubSpace is not None


def test_subspace():
    # required packages
    import numpy as np
    from numpy.testing import assert_allclose
    from scipy.signal import lfilter

    from rockpool.devices.xylo.imu.imuif_submod import Quantizer, SubSpace
    from rockpool.nn.combinators import Sequential

    #  - Generate the input signal
    r_sampling = 200
    phase = 30
    amp = 1.0
    T = 1000
    gravity = 10

    phases = np.random.rand(3) * 2 * np.pi
    base = np.sin(2 * np.pi * phase / r_sampling * np.arange(T))
    signal = np.zeros((T, 3))
    for i in range(3):
        signal[:, i] = amp * base + phases[i]
    signal[0, :] += gravity

    # - Construct the network
    num_bits_in = 16
    avg_window_duration = 50e-3  # in milli-sec

    ## How much time averaging we would like to have
    avg_window_len = int(avg_window_duration * r_sampling)
    avg_bitshift = int(np.log2(avg_window_len) + 1)

    # - Subspace module

    subspace = Sequential(
        Quantizer(
            shape=(3,),
            scale=0.999 / (np.max(np.abs(signal))),
            num_bits=num_bits_in,
        ),
        SubSpace(num_avg_bitshift=avg_bitshift),
    )
    C_list, _, _ = subspace(signal)

    # flatten the matrices into 3 x 3 --> 9 to see the components
    C_flat = C_list.reshape(-1, 9, 1).squeeze()
    C_flat_norm = C_flat / np.max(np.abs(C_flat))

    # compute the ground truth covariance matrix without any quantization and flatten it
    C_ground = lfilter(
        b=[1 / (2**avg_bitshift)],
        a=[1, -(1 - 1 / (2**avg_bitshift))],
        x=np.asarray(
            [np.outer(sig_sample, sig_sample).ravel() for sig_sample in signal]
        ),
        axis=0,
    )

    C_ground_norm = C_ground / np.max(np.abs(C_ground))

    assert_allclose(
        np.array(C_ground_norm, dtype=float),
        np.array(C_flat_norm, dtype=float),
        atol=1e-4,
    )
