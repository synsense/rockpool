'''
Test TimeSeries methods
'''
import sys

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)


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
