

def test_butter_mel_filter():

    from NetworksPython.layers import ButterMelFilter
    from NetworksPython.timeseries import TSContinuous
    import numpy as np
    import os

    base_path = "/".join(os.path.realpath(__file__).split('/')[:-1])

    fs = 16000

    filter_layer = ButterMelFilter(fs=fs)

    signal = np.load(base_path + "/files/TC4KW.npy")
    filtered = np.load(base_path + "/files/TC4KW_ButterMelWithLPF.npy")

    time = np.arange(0, len(signal)) / fs
    tsSignal = TSContinuous(time, signal)

    filter_output = filter_layer.evolve(tsSignal)

    assert(abs(filtered - filter_output.samples) < 1e-5).all()
