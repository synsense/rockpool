import pytest


def test_imports():
    from rockpool.devices.xylo.syns65302 import ChipButterworth, PDM_ADC


def test_filterbank():
    """
    this module tests the filterabnk + divisive normalization module.
    """
    import numpy as np
    from numpy.linalg import norm
    from rockpool.devices.xylo.syns65302 import PDM_ADC, ChipButterworth
    from rockpool.devices.xylo.syns65302.afe.digital_filterbank import NUM_FILETRS

    pdm_adc = PDM_ADC()
    fs = pdm_adc[0].fs

    # create a chirp signal
    freq_min = 1_00
    freq_max = 10_000
    duration = 0.1

    time_vec = np.arange(0, duration, step=1 / fs)

    freq_inst = freq_min + (freq_max - freq_min) / duration * time_vec
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs

    safe_amplitude = 0.9
    sig_in = safe_amplitude * np.sin(phase_inst)

    sig_out, _, _ = pdm_adc((sig_in, fs), record=False)

    # compute the filterbank output
    fb = ChipButterworth()

    sig_filtered, _, _ = fb(sig_out)

    assert sig_filtered.shape[1] == NUM_FILETRS
    assert sig_filtered.shape[0] == len(sig_out)
