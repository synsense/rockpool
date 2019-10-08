"""
Test layer implementing butterworth / mel-spaced filterbanks
"""


def test_butter_mel_filter():
    """
    Test the operation of the `.ButterMelFilter` layer
    """

    from rockpool.layers import ButterMelFilter
    from rockpool.timeseries import TSContinuous
    import numpy as np
    import os

    base_path = "/".join(os.path.realpath(__file__).split("/")[:-1])

    fs = 16000

    filter_layer = ButterMelFilter(fs=fs, num_workers=2)

    signal = np.load(base_path + "/files/TC4KW.npy")
    filtered = np.load(base_path + "/files/TC4KW_ButterMelWithLPF.npy")

    time = np.arange(0, len(signal)) / fs
    tsSignal = TSContinuous(time, signal)

    filter_output = filter_layer.evolve(tsSignal)

    assert (abs(filtered - filter_output.samples) < 1e-5).all()
