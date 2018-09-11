'''
Test TimeSeries methods
'''
import sys
import pytest

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)

def test_imports():
    '''
    Test importing TimeSeries classes
    '''
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
    '''
    Test using the plotting backend setting functions
    '''
    from NetworksPython.timeseries import GetPlottingBackend, SetPlottingBackend
    bUseMPL, bUseHV = GetPlottingBackend()
    SetPlottingBackend('matplotlib')
    SetPlottingBackend('holoviews')


def test_continuous_operators():
    '''
    Test creation and manipulation of a continuous time series
    '''
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


def test_merge():
    from NetworksPython import TSContinuous, TSEvent
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])


def test_TSEvent_raster():
    '''
    Test TSEvent raster function on merging other time series events
    '''
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([0, 30], 0)
    for i in range(1, 4):
        testTSEvent.merge(TSEvent(None, i))

    raster = testTSEvent.raster(tDt=1)[2]
    assert raster.shape == (31, 4)


def test_TSEvent_raster_explicit_nNumChannels():
    '''
    Test TSEvent raster method when the function is initialized with explicit number of Channels
    '''
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([0, 30], 0, nNumChannels=5)
    for i in range(1, 4):
        testTSEvent.merge(TSEvent(None, i))

    raster = testTSEvent.raster(tDt=1)[2]
    assert raster.shape == (31, 5)

def test_TSEvent_empty():
    '''
    Test TSEvent instantiation with empty objects or None
    '''
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([], [])
    assert testTSEvent.nNumChannels == 0

    testTSEvent = TSEvent(None, None)
    assert testTSEvent.nNumChannels == 0
