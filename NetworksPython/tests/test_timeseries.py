'''
Test TimeSeries methods
'''
import sys

strNetworkPath = sys.path[0] + "/../.."
sys.path.insert(1, strNetworkPath)


def test_TSEvent_raster():
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([0, 30], 0)
    for i in range(1, 4):
        testTSEvent.merge(TSEvent(None, i))

    raster = testTSEvent.raster(tDt=1)[1]
    assert raster.shape == (31, 5)
