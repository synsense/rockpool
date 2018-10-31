"""
Test TimeSeries methods
"""
import sys
import pytest
import numpy as np

def test_imports():
    """
    Test importing TimeSeries classes
    """
    from NetworksPython import TimeSeries
    from NetworksPython import TSContinuous
    from NetworksPython import TSEvent
    from NetworksPython.timeseries import TimeSeries
    from NetworksPython.timeseries import TSContinuous
    from NetworksPython.timeseries import TSEvent
    from NetworksPython.timeseries import GetPlottingBackend
    from NetworksPython.timeseries import SetPlottingBackend
    import NetworksPython.timeseries as ts


def test_backends():
    """
    Test using the plotting backend setting functions
    """
    from NetworksPython.timeseries import GetPlottingBackend, SetPlottingBackend

    bUseMPL, bUseHV = GetPlottingBackend()
    SetPlottingBackend("matplotlib")
    SetPlottingBackend("holoviews")


def test_continuous_operators():
    """
    Test creation and manipulation of a continuous time series
    """
    from NetworksPython import TSContinuous

    # - Creation
    ts = TSContinuous([0], [0])
    ts = TSContinuous([0, 1, 2, 3], [1, 2, 3, 4])
    ts2 = TSContinuous([1, 2, 3, 4], [5, 6, 7, 8])

    # - Samples don't match time
    with pytest.raises(AssertionError):
        TSContinuous([0, 1, 2], [0])

    # - Addition
    ts = ts + 1
    ts += 5
    ts = ts + ts2
    ts += ts2

    # - Subtraction
    ts = ts - 3
    ts -= 2
    ts = ts - ts2
    ts -= ts2

    # - Multiplication
    ts = ts * .9
    ts *= .2
    ts = ts * ts2
    ts *= ts2

    # - Division
    ts = ts / 2.
    ts /= 1.
    ts = ts / ts2
    ts /= ts2

    # - Floor division
    ts = ts // 1.
    ts //= 1.
    ts = ts // ts2
    ts //= ts2


def test_continuous_methods():
    from NetworksPython import TSContinuous

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])

    # - Interpolation
    assert ts1(0) == 0
    assert ts1(2) == 2
    assert ts1(1.5) == 1.5

    assert ts1.interpolate(0) == 0
    assert ts1.interpolate(2) == 2
    assert ts1.interpolate(1.5) == 1.5

    # - Delay
    ts2 = ts1.delay(1)
    assert ts1.tStart == 0
    assert ts2.tStart == 1

    # - Contains
    assert ts1.contains(0)
    assert ~ts1.contains(-1)
    assert ts1.contains([0, 1, 2])
    assert ~ts1.contains([0, 1, 2, 3])

    # - Resample
    ts2 = ts1.resample([.1, 1.1, 1.9])
    ts3 = ts1.resample_within(0, 2, .1)

    # - Merge
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts3 = ts1.merge(ts2, bRemoveDuplicates = True)
    assert np.size(ts3.mfSamples) == 3
    assert np.size(ts1.mfSamples) == 3
    assert np.size(ts2.mfSamples) == 3

    ts3 = ts1.merge(ts2, bRemoveDuplicates = False)
    assert np.size(ts3.mfSamples) == 6

    # - Append
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts3 = ts1.append_t(ts2)
    assert np.size(ts3.vtTimeTrace) == 6

    ts3 = ts1.append(ts2)
    assert ts3.nNumTraces == 2

    # - Concatenate
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts3 = ts1.concatenate_t(ts2)
    assert np.size(ts3.vtTimeTrace) == 6

    ts3 = ts1.concatenate(ts2)
    assert ts3.nNumTraces == 2

    # - isempty
    assert ~ts1.isempty()
    assert TSContinuous([], []).isempty()

    # - clip
    ts2 = ts1.clip([.5, 1.5])

    # - Min / Max
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    assert ts1.min() == 0
    assert ts1.max() == 2


def test_continuous_inplace_mutation():
    from NetworksPython import TSContinuous

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])

    # - Delay
    ts1.delay(1, bInPlace = True)
    assert ts1.tStart == 1

    # - Resample
    ts1.resample([.125, 1.1, 1.9], bInPlace = True)
    assert ts1.tStart == .125

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts1.resample_within(0, 1, .1, bInPlace = True)
    assert ts1.tStop == 1

    # - Merge
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.merge(ts2, bRemoveDuplicates = True, bInPlace = True)
    assert np.size(ts1.mfSamples) == 3

    ts3 = ts1.merge(ts2, bRemoveDuplicates = False, bInPlace = True)
    assert np.size(ts3.mfSamples) == 6

    # - Append
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.append_t(ts2, bInPlace = True)
    assert np.size(ts1.vtTimeTrace) == 6

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.append(ts2, bInPlace = True)
    assert ts1.nNumTraces == 2

    # - Concatenate
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.concatenate_t(ts2, bInPlace = True)
    assert np.size(ts1.vtTimeTrace) == 6

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.concatenate(ts2, bInPlace = True)
    assert ts1.nNumTraces == 2

    # - clip
    ts1.clip([.5, 1.5], bInPlace = True)
    assert ts1.tStart == .5


def test_event_operators():
    """
    Test creation and manipulation of a continuous time series
    """
    from NetworksPython import TSEvent

    # - Creation
    ts = TSEvent([0], [0])
    ts = TSEvent([0, 1, 2, 3], [1, 2, 3, 4])
    ts2 = TSEvent([1, 2, 3, 4], [5, 6, 7, 8])

    # - Samples don't match time
    with pytest.raises(AssertionError):
        TSEvent([0, 1, 2], 0, [0])

    # - Addition
    ts = ts + 1
    ts += 5
    ts = ts + ts2
    ts += ts2

    # - Subtraction
    ts = ts - 3
    ts -= 2
    ts = ts - ts2
    ts -= ts2

    # - Multiplication
    ts = ts * .9
    ts *= .2
    ts = ts * ts2
    ts *= ts2

    # - Division
    ts = ts / 2.
    ts /= 1.
    ts = ts / ts2
    ts /= ts2

    # - Floor division
    ts = ts // 1.
    ts //= 1.
    ts = ts // ts2
    ts //= ts2


def test_TSEvent_raster():
    """
    Test TSEvent raster function on merging other time series events
    """
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([0, 30], 0)
    for i in range(1, 4):
        testTSEvent.merge(TSEvent(None, i))

    raster = testTSEvent.raster(tDt=1)[2]
    assert raster.shape == (31, 4)


def test_TSEvent_raster_explicit_nNumChannels():
    """
    Test TSEvent raster method when the function is initialized with explicit number of Channels
    """
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([0, 30], 0, nNumChannels=5)
    for i in range(1, 4):
        testTSEvent.merge(TSEvent(None, i))

    raster = testTSEvent.raster(tDt=1)[2]
    assert raster.shape == (31, 5)


def test_TSEvent_empty():
    """
    Test TSEvent instantiation with empty objects or None
    """
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([], [])
    assert testTSEvent.nNumChannels == 0

    testTSEvent = TSEvent(None, None)
    assert testTSEvent.nNumChannels == 0
