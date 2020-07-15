import os
import numpy as np
import pytest
from rockpool.timeseries import TSContinuous


def test_butter_mel_filter():

    ## IMPORT ##
    from rockpool.layers import ButterMelFilter

    base_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
    signal = np.load(base_path + "/files/increasing_frequency_signal.npy")
    fs, f_max, duration = (10e3, 5e3, 10.0)
    times = np.arange(0.0, duration, 1 / fs)

    ## ARGUMENTS ##
    # Sampling frequency fs
    with pytest.raises(AssertionError):
        lyr = ButterMelFilter(fs=0)

    # Cutoff frequency cutoff_fs
    lyr = ButterMelFilter(fs=fs, cutoff_fs=1.0)
    lyr = ButterMelFilter(fs=fs, cutoff_fs=1e3)
    with pytest.raises(AssertionError):
        lyr = ButterMelFilter(fs=fs, cutoff_fs=fs / 2)
    with pytest.raises(AssertionError):
        lyr = ButterMelFilter(fs=fs, cutoff_fs=0.0)

    # Filter order
    lyr = ButterMelFilter(fs=fs, order=6)
    with pytest.raises(AssertionError):
        lyr = ButterMelFilter(fs=fs, order=0)
    with pytest.raises(AssertionError):
        lyr = ButterMelFilter(fs=fs, order=2.4)

    # Number of workers num_workers
    lyr = ButterMelFilter(fs=fs, num_workers=10)
    with pytest.raises(AssertionError):
        lyr = ButterMelFilter(fs=fs, num_workers=0)
    with pytest.raises(AssertionError):
        lyr = ButterMelFilter(fs=fs, num_workers=1.5)

    # Mean-subtraction and normalize
    lyr = ButterMelFilter(fs=fs, mean_subtraction=True, normalize=True)

    ## METHODS ##
    # __init__()
    lyr = ButterMelFilter(fs=fs)

    # evolve()
    ts_input = TSContinuous(times=times, samples=signal)
    ts_output = lyr.evolve(ts_input)

    # get_analytical_filter_response
    freq, output = lyr.get_analytical_filter_response(int(duration * fs))

    # to_dict() and load_from_dict()
    config = lyr.to_dict()
    lyr = ButterMelFilter.load_from_dict(config)

    # save_layer() and load_from_layer()
    lyr.save_layer(base_path + "/files/lyr.json")
    lyr = ButterMelFilter.load_from_file(base_path + "/files/lyr.json")

    try:
        os.remove(base_path + "/files/lyr.json")
    finally:
        pass

    # reset_all() and terminate()
    lyr.reset_all()
    lyr.terminate()


def test_butter_filter():

    ## IMPORT ##
    from rockpool.layers import ButterFilter

    base_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
    signal = np.load(base_path + "/files/increasing_frequency_signal.npy")
    fs, f_max, duration = (10e3, 5e3, 10.0)
    times = np.arange(0.0, duration, 1 / fs)
    ts_input = TSContinuous(times=times, samples=signal)
    frequency = np.linspace(1000, 4000, 16, endpoint=True)
    bandwidth = 200

    ## ARGUMENTS ##
    # Sampling frequency fs
    with pytest.raises(AssertionError):
        lyr = ButterFilter(0, frequency, bandwidth)

    # Frequency
    lyr = ButterFilter(fs, 10.0, 10.0)
    lyr = ButterFilter(fs, np.linspace(100.0, 4000.0, 36), np.linspace(10.0, 400.0, 36))
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs, [1000.0, 2000.0, 3000.0], [100.0, 100.0])
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs, 4990.0, 20.0)
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs, 10.0, 20.0)

    # Filter order
    lyr = ButterFilter(fs, frequency, bandwidth, order=6)
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs, frequency, bandwidth, order=0)
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs, frequency, bandwidth, order=2.4)

    # Number of workers num_workers
    lyr = ButterFilter(fs, frequency, bandwidth, num_workers=10)
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs, frequency, bandwidth, num_workers=0)
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs, frequency, bandwidth, num_workers=1.5)

    # Mean-subtraction and normalize
    lyr = ButterFilter(fs, frequency, bandwidth, mean_subtraction=True, normalize=True)

    ## METHODS ##
    # __init__()
    lyr = ButterFilter(fs, frequency, bandwidth)

    # evolve()
    ts_output = lyr.evolve(ts_input)

    # get_analytical_filter_response()
    freq, output = lyr.get_analytical_filter_response(int(duration * fs))

    # to_dict() and load_from_dict()
    config = lyr.to_dict()
    lyr = ButterFilter.load_from_dict(config)

    # save_layer() and load_from_file()
    lyr.save_layer(base_path + "/files/lyr.json")
    lyr = ButterFilter.load_from_file(base_path + "/files/lyr.json")

    try:
        os.remove(base_path + "/files/lyr.json")
    finally:
        pass

    # reset_all() and terminate()
    lyr.reset_all()
    lyr.terminate()


test_butter_filter()
test_butter_mel_filter()
