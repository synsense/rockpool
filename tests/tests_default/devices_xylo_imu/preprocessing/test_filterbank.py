def test_import():
    from rockpool.devices.xylo.imu.preprocessing import FilterBank

    assert FilterBank is not None

    from rockpool.devices.xylo.imu.preprocessing import BandPassFilter

    assert BandPassFilter is not None


def test_bandpass_filter():
    """
    NOTE : work on this later with Saeid after explaining the `BandPassFilter` parameters
    CHECK ISSUE #259
    """
    import numpy as np
    from rockpool.devices.xylo.imu.preprocessing import Quantizer, BandPassFilter

    # Generate a test signal
    duration = 1.0  # seconds
    t = np.linspace(0, duration, 10)
    signal = np.sin(2 * np.pi * t)

    quantizer = Quantizer(shape=1, scale=0.999 / np.max(np.abs(signal)), num_bits=16)

    q_signal, _, _ = quantizer(signal)
    q_signal = q_signal.flatten()

    bandpass_filter = BandPassFilter(a1=-6400)

    # Apply the bandpass filter to the test signal
    filtered_signal = bandpass_filter(q_signal)

    # Assert that the filtered signal meets the expected criteria
    assert len(filtered_signal) == len(q_signal)


def test_filterbank():
    """
    IMPORTANT NOTE: NEED TO BE WORKED ON AND FIXED.
    CHECK ISSUE #259
    """
    from rockpool.devices.xylo.imu.preprocessing import (
        RotationRemoval,
        Quantizer,
        FilterBank,
    )
    from rockpool.nn.combinators import Sequential
    import numpy as np
    from copy import deepcopy
    from scipy.signal import lfilter
    import pytest

    np.random.seed(2023)

    # - Test values
    num_bits = 16
    num_bits_multiplier = num_bits + 10
    num_avg_bitshift = 11
    sampling_period = 10

    # - Synthetic data generation
    T = 100
    corr_window = 100

    filt_b = [1 / corr_window]
    filt_a = [1, -(1 - 1 / corr_window)]

    input_signal = lfilter(filt_b, filt_a, np.random.randn(1, T, 3), axis=0)
    input_signal_rotated = deepcopy(input_signal)

    ## -- Add random rotation
    g = np.zeros((T, 3))

    for t in range(T):
        if T % (sampling_period * 2) == 0:
            col = np.random.randint(0, 3)
        g[t, col] = 9.8

    input_signal_rotated += g

    mod_if = Sequential(
        Quantizer(
            scale=0.999 / np.max(np.abs(input_signal_rotated)),
            num_bits=num_bits,
        ),
        RotationRemoval(
            num_bits_in=num_bits,
            num_bits_out=num_bits,
            num_avg_bitshift=11,
            sampling_period=sampling_period,
        ),
        FilterBank(),
    )

    __B, __T, __C = input_signal.shape
    __F = __C * mod_if[2].numF
    q_filt_signal, _, _ = mod_if(input_signal)
    assert q_filt_signal.shape == (__B, __T, __F)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(q_filt_signal, np.zeros_like(q_filt_signal))
