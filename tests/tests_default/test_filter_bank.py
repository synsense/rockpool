import pytest

pytest.importorskip("scipy")


def test_butter_mel_filter():
    ## IMPORT ##
    from rockpool.nn.modules import ButterMelFilter
    from rockpool.timeseries import TSContinuous
    import numpy as np
    import pytest
    import scipy
    from tempfile import mkstemp
    import os

    base_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
    fs, f_max, duration = (10e3, 5e3, 10.0)
    times = np.arange(0.0, duration, 1 / fs)
    signal = np.reshape(scipy.signal.chirp(times, 0.0, duration, f_max), (-1, 1))

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
    ts_output, state, rec = lyr.evolve(signal)


def test_butter_filter():
    ## IMPORT ##
    from rockpool.nn.modules import ButterFilter
    from rockpool.timeseries import TSContinuous
    import numpy as np
    import pytest
    import scipy
    import os

    base_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
    fs, f_max, duration = (10e3, 5e3, 10.0)
    times = np.arange(0.0, duration, 1 / fs)
    signal = np.reshape(scipy.signal.chirp(times, 0.0, duration, f_max), (-1, 1))
    ts_input = TSContinuous(times=times, samples=signal)
    frequency = np.linspace(1000, 4000, 16, endpoint=True)
    bandwidth = 200

    ## ARGUMENTS ##
    # Sampling frequency fs
    with pytest.raises(ValueError):
        lyr = ButterFilter(fs=0, frequency=frequency, bandwidth=bandwidth)

    # Frequency
    lyr = ButterFilter(fs=fs, frequency=10.0, bandwidth=10.0)
    lyr = ButterFilter(
        fs=fs,
        frequency=np.linspace(100.0, 4000.0, 36),
        bandwidth=np.linspace(10.0, 400.0, 36),
    )
    with pytest.raises(ValueError):
        lyr = ButterFilter(
            fs=fs, frequency=[1000.0, 2000.0, 3000.0], bandwidth=[100.0, 100.0]
        )
    with pytest.raises(ValueError):
        lyr = ButterFilter(fs=fs, frequency=4990.0, bandwidth=20.0)
    with pytest.raises(ValueError):
        lyr = ButterFilter(fs=fs, frequency=10.0, bandwidth=20.0)

    # Filter order
    lyr = ButterFilter(fs=fs, frequency=frequency, bandwidth=bandwidth, order=6)
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs=fs, frequency=frequency, bandwidth=bandwidth, order=0)
    with pytest.raises(AssertionError):
        lyr = ButterFilter(fs=fs, frequency=frequency, bandwidth=bandwidth, order=2.4)

    # Number of workers num_workers
    lyr = ButterFilter(fs=fs, frequency=frequency, bandwidth=bandwidth, num_workers=10)
    with pytest.raises(AssertionError):
        lyr = ButterFilter(
            fs=fs, frequency=frequency, bandwidth=bandwidth, num_workers=0
        )
    with pytest.raises(AssertionError):
        lyr = ButterFilter(
            fs=fs, frequency=frequency, bandwidth=bandwidth, num_workers=1.5
        )

    # Mean-subtraction and normalize
    lyr = ButterFilter(
        fs=fs,
        frequency=frequency,
        bandwidth=bandwidth,
        mean_subtraction=True,
        normalize=True,
    )

    ## METHODS ##
    # __init__()
    lyr = ButterFilter(fs=fs, frequency=frequency, bandwidth=bandwidth)

    # evolve()
    ts_output = lyr.evolve(signal)
