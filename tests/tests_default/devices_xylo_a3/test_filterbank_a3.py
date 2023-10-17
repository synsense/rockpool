import pytest


def test_imports():
    from rockpool.devices.xylo.syns65302 import ChipButterworth, PDMADC


def test_filterbank():
    """
    this module tests the filterabnk + divisive normalization module.
    """
    import numpy as np
    from numpy.linalg import norm
    from rockpool.devices.xylo.syns65302 import PDMADC, ChipButterworth
    from rockpool.devices.xylo.syns65302.afe.params import NUM_FILTERS

    pdm_adc = PDMADC()
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

    assert sig_filtered.shape[1] == NUM_FILTERS
    assert sig_filtered.shape[0] == len(sig_out)


@pytest.mark.parametrize(
    "select_filters",
    [
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        (15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        (10, 3, 7, 0, 14),
    ],
)
def test_valid_filters(select_filters: tuple):
    """
    Test that the filterbank module can be instantiated with valid filter indices

    Args:
        select_filters (tuple): The indices of the filters to be used in the filter bank
    """
    from rockpool.devices.xylo.syns65302 import ChipButterworth

    _ch = ChipButterworth(select_filters=select_filters)
    assert _ch.size_out == len(select_filters)


@pytest.mark.parametrize(
    "select_filters", [(0, 1, 2, 16), (0, 1, 2, -1), (0, 0, 1, 2), (0, 0, 1, 2)]
)
def test_invalid_filters_value_error(select_filters: tuple):
    """
    Test that the filterbank module cannot be instantiated with invalid filter indices raising a ValueError

    Args:
        select_filters (tuple): The indices of the filters to be used in the filter bank
    """
    from rockpool.devices.xylo.syns65302 import ChipButterworth

    with pytest.raises(ValueError):
        _ch = ChipButterworth(select_filters=select_filters)


@pytest.mark.parametrize("select_filters", [[0, 1, 2], (0, 1, 2, "a"), (0, 1, 2, 3.5)])
def test_invalid_filters_type_error(select_filters: tuple):
    """
    Test that the filterbank module cannot be instantiated with invalid filter indices raising a ValueError

    Args:
        select_filters (tuple): The indices of the filters to be used in the filter bank
    """
    from rockpool.devices.xylo.syns65302 import ChipButterworth

    with pytest.raises(TypeError):
        _ch = ChipButterworth(select_filters=select_filters)
