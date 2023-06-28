from random import sample
import pytest


def test_imports():
    from rockpool.devices.xylo.xylo_a3 import (
        PolyPhaseFIR_DecimationFilter,
        PDM_Microphone,
    )


def test_polyphase_decimation_filter():
    """
    this module tests the decimation filter used to rrecover the original signal from the PDM modulation bits.
    """
    import numpy as np
    from numpy.linalg import norm
    from rockpool.devices.xylo.xylo_a3 import (
        PDM_Microphone,
        PolyPhaseFIR_DecimationFilter,
    )

    # pdm microphone and decimation filter
    mic = PDM_Microphone()
    lp = PolyPhaseFIR_DecimationFilter()

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
        pdm_bits, state, recording = mic.evolve(audio_in=(sig_in, mic.fs), record=True)

        # process the microphone output to recover the original signal
        sig_out = lp._evolve_no_decimation(sig_in=pdm_bits)
        sig_out = sig_out / 2 ** (lp.num_bits_output - 1)

        # length increases slightly due to filtering
        assert len(sig_out) >= len(sig_in)

        # compute the correlation factor between input and output
        corr = np.max(np.convolve(sig_in[::-1], sig_out)) / (
            norm(sig_in) * norm(sig_out)
        )

        MIN_CORR = 0.7
        assert corr > MIN_CORR
