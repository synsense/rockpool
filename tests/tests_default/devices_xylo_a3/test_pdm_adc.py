import pytest


def test_imports():
    from rockpool.devices.xylo.syns65302 import PDMADC, MicrophonePDM


def test_pdm_adc():
    """
    this module tests the decimation filter used to rrecover the original signal from the PDM modulation bits.
    """
    import numpy as np
    from numpy.linalg import norm
    from rockpool.devices.xylo.syns65302 import MicrophonePDM, PDMADC

    # pdm microphone: just to extract the default parameters
    mic = MicrophonePDM()

    # pdm ADC
    pdm_adc = PDMADC()

    # produce a combination of sinusoid signals
    freq_min = 1_000
    freq_max = 5_000
    num_freq = 6
    freq_vec = freq_min + (freq_max - freq_min) * np.random.rand(num_freq)

    amp_vec = np.sort(np.random.rand(num_freq))
    phase_vec = 2 * np.pi * np.random.rand(num_freq)

    num_periods = 4
    duration = num_periods / freq_min

    sample_rate = mic.fs
    time_vec = np.arange(0, duration, step=1 / sample_rate)
    safe_amplitude = 0.9

    num_random_tests = 10

    for _ in range(num_random_tests):
        # produce the random signal
        sig_in = np.einsum(
            "i, ij -> j",
            amp_vec,
            np.sin(
                2 * np.pi * freq_vec.reshape(-1, 1) * time_vec.reshape(1, -1)
                + phase_vec.reshape(-1, 1)
            ),
        )

        # normalize its amplitude for PDM mic
        sig_in = safe_amplitude * sig_in / np.max(np.abs(sig_in))

        # apply PDM modulation in PDM mic
        sig_out, state, recording = pdm_adc((sig_in, sample_rate), record=False)

        # normalize the output to the number of bits in decimation filter
        sig_out = sig_out / 2 ** (pdm_adc[1].num_bits_output - 1)

        # length increases slightly due to filtering
        sample_rate_out = mic.fs / mic.sdm_OSR
        duration_out = len(sig_out) / sample_rate_out

        duration_err = abs(duration - duration_out) / duration
        EPS = 0.2

        assert duration_err < EPS

        # * compute the correlation factor between input and output
        # resample the output signal
        time_out = np.arange(len(sig_out)) / sample_rate_out

        sig_out_resampled = np.interp(time_vec, time_out, sig_out)

        corr = np.max(np.convolve(sig_in[::-1], sig_out_resampled)) / (
            norm(sig_in) * norm(sig_out_resampled)
        )

        MIN_CORR = 0.7
        assert corr > MIN_CORR
