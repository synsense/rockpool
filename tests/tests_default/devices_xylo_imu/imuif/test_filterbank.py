def test_import():
    from rockpool.devices.xylo.imu.imuif import FilterBank

    assert FilterBank is not None

    from rockpool.devices.xylo.imu.imuif import BandPassFilter

    assert BandPassFilter is not None


def test_BPF_FB_from_specification():
    import pytest
    from rockpool.devices.xylo.imu.imuif import BandPassFilter, FilterBank

    bpf = BandPassFilter.from_specification(0.1, 10)

    with pytest.raises(ValueError):
        bpf = BandPassFilter.from_specification(0.1, 0.1)

    with pytest.raises(ValueError):
        bpf = BandPassFilter.from_specification(0.1, 10, 2)

    fb = FilterBank()

    db = FilterBank.from_specification((1, 3), (0.1, 1), (1, 10), (40, 50))


def test_bandpass_filter():
    import numpy as np
    from rockpool.devices.xylo.imu.imuif import BandPassFilter
    from rockpool.devices.xylo.imu.transform import Quantizer

    # Generate a test signal
    duration = 1.0  # seconds
    t = np.linspace(0, duration, 10)
    signal = np.sin(2 * np.pi * t)

    quantizer = Quantizer(shape=1, scale=0.999 / np.max(np.abs(signal)), num_bits=16)

    q_signal, _, _ = quantizer(signal)
    q_signal = q_signal.flatten()

    bandpass_filter = BandPassFilter(a1=6400)

    # Apply the bandpass filter to the test signal
    filtered_signal = bandpass_filter(q_signal)

    # Assert that the filtered signal meets the expected criteria
    assert len(filtered_signal) == len(q_signal)


def test_filterbank():
    from rockpool.devices.xylo.imu.imuif import (
        RotationRemoval,
        FilterBank,
    )
    from rockpool.devices.xylo.imu.transform import Quantizer
    from rockpool.nn.combinators import Sequential
    import numpy as np
    from copy import deepcopy
    from scipy.signal import lfilter
    import pytest

    np.random.seed(2023)

    # - Test values
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
            num_bits=16,
        ),
        RotationRemoval(
            num_avg_bitshift=11,
            sampling_period=sampling_period,
        ),
        FilterBank(),
    )

    __B, __T, __C = input_signal.shape
    __F = mod_if[2].size_out
    q_filt_signal, _, _ = mod_if(input_signal)
    assert q_filt_signal.shape == (__B, __T, __F)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(q_filt_signal, np.zeros_like(q_filt_signal))


def test_signal_gain_drop():
    """
    Create a chirp signal and filter it with a bandpass filter.
    Compute the power of the filtered signal in the pass band and in the stop band.
    Assert that the power in the pass band is at least 3dB higher than the power in the stop band.
    """

    import numpy as np
    from rockpool.devices.xylo.imu.imuif import BandPassFilter
    from rockpool.devices.xylo.imu.transform import Quantizer

    f_init = 0.5
    f_end = 20.5
    f_sampling = 200
    t_end = 10
    band_width = 1

    quantizer = Quantizer(shape=(1, 1), scale=0.9, num_bits=16)

    def power_dB(signal: np.ndarray) -> float:
        """Compute the power of a signal in dB"""
        power = np.mean(np.abs(signal) ** 2)
        return 10 * np.log10(power)

    def f_idx(freq: float) -> int:
        """Compute the index of a frequency in the chirp signal"""
        return int(f_sampling * (freq - f_init) * t_end / (f_end - f_init))

    def get_quantized_chirp_signal() -> np.ndarray:
        """Return a quantized chirp signal"""
        t = np.linspace(0, t_end, int(t_end * f_sampling), endpoint=False)
        frequency = f_init + (f_end - f_init) * t / t_end
        phase = 2 * np.pi * np.cumsum(frequency) / f_sampling
        chirp_signal = np.sin(phase)
        quantized_data, _, _ = quantizer(chirp_signal)
        return quantized_data.flatten()

    low_cut_range = np.arange(f_init, f_end, band_width)
    high_cut_range = np.arange(f_init + band_width, f_end + band_width, band_width)

    # Sweep the `band_width` wide bandpass filters over the chirp signal

    for low, high in zip(low_cut_range, high_cut_range):
        chirp_signal = get_quantized_chirp_signal()
        band_pass = BandPassFilter.from_specification(low, high)
        filtered_data = band_pass(chirp_signal)

        # Compute the power of the pass and stop bands of the filtered signal
        idx_low_cut = f_idx(low)
        idx_high_cut = f_idx(high)

        pass_band = filtered_data[idx_low_cut:idx_high_cut]
        stop_band = np.concatenate(
            (filtered_data[:idx_low_cut], filtered_data[idx_high_cut:])
        )

        pass_band_power_dB = power_dB(pass_band)
        stop_band_power_dB = power_dB(stop_band)

        assert pass_band_power_dB - stop_band_power_dB > 3
