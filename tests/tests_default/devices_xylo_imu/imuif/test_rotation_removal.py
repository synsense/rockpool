def test_import():
    from rockpool.devices.xylo.imu.imuif_submod import RotationRemoval

    assert RotationRemoval is not None


def test_rotation_removal():
    """
    IMPORTANT NOTE: NEED TO BE WORKED ON AND FIXED.
    CHECK ISSUE #252
    """
    from rockpool.devices.xylo.imu.imuif_submod import RotationRemoval
    from rockpool.devices.xylo.imu.transform import Quantizer
    from numpy.testing import assert_allclose
    import numpy as np
    from copy import deepcopy
    from scipy.signal import lfilter

    np.random.seed(2023)

    # - Test values
    num_bits = 16
    sampling_period = 10

    # - Synthetic data generation
    T = 100
    corr_window = 100

    filt_b = [1 / corr_window]
    filt_a = [1, -(1 - 1 / corr_window)]

    input_signal = lfilter(filt_b, filt_a, np.random.randn(T, 3), axis=0)
    input_signal_rotated = deepcopy(input_signal)

    ## -- Add random rotation
    g = np.zeros((T, 3))

    for t in range(T):
        if T % (sampling_period * 2) == 0:
            col = np.random.randint(0, 3)
        g[t, col] = 9.8

    input_signal_rotated += g

    quantizer = Quantizer(
        shape=(3, 3),
        scale=0.999 / np.max(np.abs(input_signal_rotated)),
        num_bits=num_bits,
    )

    # - Build quantized rotation removal module
    q_rot_remove = RotationRemoval(
        num_avg_bitshift=11,
        sampling_period=sampling_period,
    )

    q_ref_signal, _, _ = quantizer(input_signal)
    q_signal, _, _ = quantizer(input_signal_rotated)
    signal_rotation_removed, _, _ = q_rot_remove(q_signal)

    signal_rotation_removed = signal_rotation_removed / (2 ** (num_bits - 1))
    q_ref_signal = q_ref_signal / (2 ** (num_bits - 1))

    ## Sign correction
    correction_factor = np.sign(signal_rotation_removed) * np.sign(input_signal)
    signal_rotation_removed = signal_rotation_removed * correction_factor

    assert_allclose(
        q_ref_signal.astype(float), signal_rotation_removed.astype(float), atol=0.5
    )
