import pytest


def test_imports():
    from rockpool.devices.xylo.syns65302 import (
        ChipButterworth,
        PDM_ADC,
        DivisiveNormalization,
    )


def test_filterbank():
    """
    this module tests the filterabnk + divisive normalization module.
    """
    import numpy as np
    from numpy.linalg import norm
    from rockpool.devices.xylo.syns65302 import (
        ChipButterworth,
        DivisiveNormalization,
        PDM_ADC,
    )
    from rockpool.devices.xylo.syns65302.xylo_a3_sim.digital_filterbank import NUM_FILETRS

    # just to extract the sampling rate
    pdm_adc = PDM_ADC()
    fs = pdm_adc[0].fs
    oversampling = pdm_adc[0].sdm_OSR
    num_bits_output = pdm_adc[1].num_bits_output

    # create a chirp signal
    freq_min = 1_00
    freq_max = 20_000
    duration = 1

    time_vec = np.arange(0, duration, step=1 / fs)

    freq_inst = freq_min + (freq_max - freq_min) / duration * time_vec
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs

    safe_amplitude = 0.9
    sig_in = safe_amplitude * np.sin(phase_inst)

    # downsample to audio sample rate
    sig_in = sig_in[::oversampling]

    # quantize
    sig_in = (2 ** (num_bits_output - 1) * sig_in).astype(np.int64)

    # compute the filterbank output
    fb = ChipButterworth()

    sig_filtered, _, _ = fb(sig_in)

    assert sig_filtered.shape[1] == NUM_FILETRS

    # aplly divisive normalization and spike generation
    dn = DivisiveNormalization()
    spikes, _, _ = dn(sig_filtered)

    sample_rate = fs / oversampling
    spike_rate = np.mean(spikes, axis=0) * sample_rate

    assert spikes.shape == sig_filtered.shape
