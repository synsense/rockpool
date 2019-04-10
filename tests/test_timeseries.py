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
    from NetworksPython.timeseries import set_plotting_backend
    from NetworksPython.timeseries import get_plotting_backend
    import NetworksPython.timeseries as ts


def test_backends():
    """
    Test using the plotting backend setting functions
    """
    from NetworksPython.timeseries import set_plotting_backend, get_plotting_backend

    bUseMPL, bUseHV = get_plotting_backend()
    set_plotting_backend("matplotlib")
    set_plotting_backend("holoviews")


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
    ts = ts * 0.9
    ts *= 0.2
    ts = ts * ts2
    ts *= ts2

    # - Division
    ts = ts / 2.0
    ts /= 1.0
    ts = ts / ts2
    ts /= ts2

    # - Floor division
    ts = ts // 1.0
    ts //= 1.0
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
    ts2 = ts1.resample([0.1, 1.1, 1.9])
    ts3 = ts1.resample_within(0, 2, 0.1)

    # - Merge
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts3 = ts1.merge(ts2, bRemoveDuplicates=True)
    assert np.size(ts3.mfSamples) == 3
    assert np.size(ts1.mfSamples) == 3
    assert np.size(ts2.mfSamples) == 3

    ts3 = ts1.merge(ts2, bRemoveDuplicates=False)
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
    ts2 = ts1.clip([0.5, 1.5])

    # - Min / Max
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    assert ts1.min() == 0
    assert ts1.max() == 2


def test_continuous_inplace_mutation():
    from NetworksPython import TSContinuous

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])

    # - Delay
    ts1.delay(1, inplace=True)
    assert ts1.tStart == 1

    # - Resample
    ts1.resample([0.125, 1.1, 1.9], inplace=True)
    assert ts1.tStart == 0.125

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts1.resample_within(0, 1, 0.1, inplace=True)
    assert ts1.tStop == 1

    # - Merge
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.merge(ts2, bRemoveDuplicates=True, inplace=True)
    assert np.size(ts1.mfSamples) == 3

    ts3 = ts1.merge(ts2, bRemoveDuplicates=False, inplace=True)
    assert np.size(ts3.mfSamples) == 6

    # - Append
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.append_t(ts2, inplace=True)
    assert np.size(ts1.vtTimeTrace) == 6

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.append(ts2, inplace=True)
    assert ts1.nNumTraces == 2

    # - Concatenate
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.concatenate_t(ts2, inplace=True)
    assert np.size(ts1.vtTimeTrace) == 6

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.concatenate(ts2, inplace=True)
    assert ts1.nNumTraces == 2

    # - clip
    ts1.clip([0.5, 1.5], inplace=True)
    assert ts1.tStart == 0.5


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
        TSEvent([0, 1, 2], [0, 1])

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
    ts = ts * 0.9
    ts *= 0.2
    ts = ts * ts2
    ts *= ts2

    # - Division
    ts = ts / 2.0
    ts /= 1.0
    ts = ts / ts2
    ts /= ts2

    # - Floor division
    ts = ts // 1.0
    ts //= 1.0
    ts = ts // ts2
    ts //= ts2


def test_TSEvent_raster():
    """
    Test TSEvent raster function on merging other time series events
    """
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([0, 30], 0)
    for i in range(1, 4):
        testTSEvent.merge(TSEvent(None, i), inplace=True)

    raster = testTSEvent.raster(dt=1)
    assert raster.shape == (31, 4)


def test_TSEvent_raster_explicit_nNumChannels():
    """
    Test TSEvent raster method when the function is initialized with explicit number of Channels
    """
    from NetworksPython import TSEvent

    testTSEvent = TSEvent([0, 30], 0, nNumChannels=5)
    for i in range(1, 4):
        testTSEvent.merge(TSEvent(None, i), inplace=True)

    raster = testTSEvent.raster(dt=1)
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

    testTSEvent = TSEvent()
    assert testTSEvent.nNumChannels == 0


def test_TSEvent_append_c():
    """
    Test merge method of TSEvent
    """
    from NetworksPython import TSEvent

    # - Generate a few TSEvent objects
    empty_series = TSEvent()
    series_list = []
    series_list.append(TSEvent([1, 2], [2, 0], tStart=-1, tStop=2))
    series_list.append(TSEvent([0, 1, 4], [1, 1, 0], tStart=0, tStop=6))
    series_list.append(TSEvent([1], [0], tStart=0, tStop=2, nNumChannels=5))

    # Merging two series
    appended_fromtwo = series_list[0].append_c(series_list[2])
    assert (
        appended_fromtwo.nNumChannels == 8
    ), "Wrong channel count for appended series."
    assert appended_fromtwo.tStart == -1, "Wrong tStart for appended series."
    assert appended_fromtwo.tStop == 2, "Wrong tStop for appended series."
    assert (
        appended_fromtwo.vtTimeTrace == np.array([1, 1, 2])
    ).all(), "Wrong time trace for appended series."
    assert (appended_fromtwo.vnChannels == np.array([2, 3, 0])).all() or (
        appended_fromtwo.vnChannels == np.array([3, 2, 0])
    ).all(), "Wrong channels for appended series."

    # Merging with empty series
    appended_empty_first = empty_series.append_c(series_list[0])
    assert (
        appended_empty_first.tStart == series_list[0].tStart
    ), "Wrong tStart when appending with empty."
    assert (
        appended_empty_first.tStop == series_list[0].tStop
    ), "Wrong tStop when appending with empty."
    assert (
        appended_empty_first.nNumChannels == series_list[0].nNumChannels
    ), "Wrong channel count when appending with empty."
    assert (
        appended_empty_first.vnChannels == series_list[0].vnChannels
    ).all(), "Wrong channels when appending with empty"
    assert (
        appended_empty_first.vtTimeTrace == series_list[0].vtTimeTrace
    ).all(), "Wrong time trace when appending with empty"
    appended_empty_last = series_list[0].append_c(empty_series)
    assert (
        appended_empty_last.tStart == series_list[0].tStart
    ), "Wrong tStart when appending with empty."
    assert (
        appended_empty_last.tStop == series_list[0].tStop
    ), "Wrong tStop when appending with empty."
    assert (
        appended_empty_last.nNumChannels == series_list[0].nNumChannels
    ), "Wrong channel count when appending with empty."
    assert (
        appended_empty_last.vnChannels == series_list[0].vnChannels
    ).all(), "Wrong channels when appending with empty"
    assert (
        appended_empty_last.vtTimeTrace == series_list[0].vtTimeTrace
    ).all(), "Wrong time trace when appending with empty"

    # Merging with list of series
    appended_with_list = empty_series.append_c(series_list)
    assert (
        appended_with_list.nNumChannels == 10
    ), "Wrong channel count when appending with list."
    assert appended_with_list.tStart == -1, "Wrong tStart when appending with list."
    assert appended_with_list.tStop == 6, "Wrong tStop when appending with list."
    assert (
        appended_with_list.vtTimeTrace == np.array([0, 1, 1, 1, 2, 4])
    ).all(), "Wrong time trace when appending with list."
    # Allow permutations of events 1 to 3 because they have same time
    assert (
        appended_with_list.vnChannels[[0, 4, 5]] == np.array([4, 0, 3])
    ).all() and set(appended_with_list.vnChannels[1:4]) == set(
        (2, 4, 5)
    ), "Wrong channels when appending with list."


def test_TSEvent_append_t():
    """
    Test merge method of TSEvent
    """
    from NetworksPython import TSEvent

    # - Generate a few TSEvent objects
    empty_series = TSEvent()
    series_list = []
    series_list.append(TSEvent([1, 2], [2, 0], tStart=-1, tStop=2))
    series_list.append(TSEvent([0, 1, 4], [1, 1, 0], tStart=0, tStop=6))
    series_list.append(TSEvent([1], [0], tStart=0, tStop=2, nNumChannels=5))

    # Merging two series
    appended_fromtwo = series_list[0].append_t(series_list[2])
    assert (
        appended_fromtwo.nNumChannels == 5
    ), "Wrong channel count for appended series."
    assert appended_fromtwo.tStart == -1, "Wrong tStart for appended series."
    assert appended_fromtwo.tStop == 4, "Wrong tStop for appended series."
    assert (
        appended_fromtwo.vtTimeTrace == np.array([1, 2, 3])
    ).all(), "Wrong time trace for appended series."
    assert (
        appended_fromtwo.vnChannels == np.array([2, 0, 0])
    ).all(), "Wrong channels for appended series."

    # Merging with empty series
    appended_empty_first = empty_series.append_t(series_list[0])
    assert (
        appended_empty_first.tStart == empty_series.tStart
    ), "Wrong tStart when appending with empty."
    assert (
        appended_empty_first.tStop == series_list[0].tDuration + empty_series.tStart
    ), "Wrong tStop when appending with empty."
    assert (
        appended_empty_first.nNumChannels == series_list[0].nNumChannels
    ), "Wrong channel count when appending with empty."
    assert (
        appended_empty_first.vnChannels == series_list[0].vnChannels
    ).all(), "Wrong channels when appending with empty"
    assert (
        appended_empty_first.vtTimeTrace == np.array([2, 3])
    ).all(), "Wrong time trace when appending with empty"
    appended_empty_last = series_list[0].append_t(empty_series)
    assert (
        appended_empty_last.tStart == series_list[0].tStart
    ), "Wrong tStart when appending with empty."
    assert (
        appended_empty_last.tStop == series_list[0].tStop
    ), "Wrong tStop when appending with empty."
    assert (
        appended_empty_last.nNumChannels == series_list[0].nNumChannels
    ), "Wrong channel count when appending with empty."
    assert (
        appended_empty_last.vnChannels == series_list[0].vnChannels
    ).all(), "Wrong channels when appending with empty"
    assert (
        appended_empty_last.vtTimeTrace == series_list[0].vtTimeTrace
    ).all(), "Wrong time trace when appending with empty"

    # Merging with list of series
    appended_with_list = empty_series.append_t(series_list, offset=[3, 2, 1])
    assert (
        appended_with_list.nNumChannels == 5
    ), "Wrong channel count when appending with list."
    assert appended_with_list.tStart == 0, "Wrong tStart when appending with list."
    assert appended_with_list.tStop == 17, "Wrong tStop when appending with list."
    assert (
        appended_with_list.vtTimeTrace == np.array([5, 6, 8, 9, 12, 16])
    ).all(), "Wrong time trace when appending with list."
    assert (
        appended_with_list.vnChannels == np.array([2, 0, 1, 1, 0, 0])
    ).all(), "Wrong channels when appending with list."


def test_TSEvent_merge():
    """
    Test merge method of TSEvent
    """
    from NetworksPython import TSEvent

    # - Generate a few TSEvent objects
    empty_series = TSEvent()
    series_list = []
    series_list.append(TSEvent([1, 2], [2, 0], tStart=-1, tStop=2))
    series_list.append(TSEvent([0, 1, 4], [1, 1, 0], tStart=0, tStop=6))
    series_list.append(TSEvent([1], [0], tStart=0, tStop=2, nNumChannels=5))

    # Merging two series
    merged_fromtwo = series_list[0].merge(series_list[2])
    assert merged_fromtwo.nNumChannels == 5, "Wrong channel count for merged series."
    assert merged_fromtwo.tStart == -1, "Wrong tStart for merged series."
    assert merged_fromtwo.tStop == 2, "Wrong tStop for merged series."
    assert (
        merged_fromtwo.vtTimeTrace == np.array([1, 1, 2])
    ).all(), "Wrong time trace for merged series."
    assert (merged_fromtwo.vnChannels == np.array([2, 0, 0])).all() or (
        merged_fromtwo.vnChannels == np.array([0, 2, 0])
    ).all(), "Wrong channels for merged series."

    # Merging with empty series
    merged_empty_first = empty_series.merge(series_list[0])
    assert (
        merged_empty_first.tStart == series_list[0].tStart
    ), "Wrong tStart when merging with empty."
    assert (
        merged_empty_first.tStop == series_list[0].tStop
    ), "Wrong tStop when merging with empty."
    assert (
        merged_empty_first.nNumChannels == series_list[0].nNumChannels
    ), "Wrong channel count when merging with empty."
    assert (
        merged_empty_first.vnChannels == series_list[0].vnChannels
    ).all(), "Wrong channels when merging with empty"
    assert (
        merged_empty_first.vtTimeTrace == series_list[0].vtTimeTrace
    ).all(), "Wrong time trace when merging with empty"
    merged_empty_last = series_list[0].merge(empty_series)
    assert (
        merged_empty_last.tStart == series_list[0].tStart
    ), "Wrong tStart when merging with empty."
    assert (
        merged_empty_last.tStop == series_list[0].tStop
    ), "Wrong tStop when merging with empty."
    assert (
        merged_empty_last.nNumChannels == series_list[0].nNumChannels
    ), "Wrong channel count when merging with empty."
    assert (
        merged_empty_last.vnChannels == series_list[0].vnChannels
    ).all(), "Wrong channels when merging with empty"
    assert (
        merged_empty_last.vtTimeTrace == series_list[0].vtTimeTrace
    ).all(), "Wrong time trace when merging with empty"

    # Merging with list of series
    merged_with_list = empty_series.merge(series_list, delay=[3, 2, 1])
    assert (
        merged_with_list.nNumChannels == 5
    ), "Wrong channel count when merging with list."
    assert merged_with_list.tStart == 0, "Wrong tStart when merging with list."
    assert merged_with_list.tStop == 8, "Wrong tStop when merging with list."
    assert (
        merged_with_list.vtTimeTrace == np.array([2, 2, 3, 4, 5, 6])
    ).all(), "Wrong time trace when merging with list."
    assert (merged_with_list.vnChannels == np.array([1, 0, 1, 2, 0, 0])).all() or (
        merged_with_list.vnChannels == np.array([0, 1, 1, 2, 0, 0])
    ).all(), "Wrong channels when merging with list."


def test_save_load():
    """
    Test saving and loading function for timeseries
    """
    from NetworksPython import TSEvent, TSContinuous, load_ts_from_file
    from os import remove

    # - Generate time series objects
    vtTimeTrace = [1, 3, 6]
    mfSamples = np.random.randn(3)
    vnChannels = [0, 1, 1]
    tsc = TSContinuous(
        vtTimeTrace, mfSamples, tStart=-1, tStop=8, bPeriodic=True, strName="continuous"
    )
    tse = TSEvent(
        vtTimeTrace,
        vnChannels,
        nNumChannels=3,
        tStart=-1,
        tStop=8,
        bPeriodic=True,
        strName="events",
    )
    # - Store objects
    tsc.save("test_tsc")
    tse.save("test_tse")
    # - Load objects
    tscl = load_ts_from_file("test_tsc.npz")
    tsel = load_ts_from_file("test_tse.npz")
    # - Verify that attributes are still correct
    assert (tscl.vtTimeTrace == vtTimeTrace).all(), "TSContinuous: vtTimeTrace changed."
    assert (
        tscl.mfSamples == mfSamples.reshape(-1, 1)
    ).all(), "TSContinuous: mfSamples changed."
    assert tscl.strName == "continuous", "TSContinuous: strName changed."
    assert tscl.tStart == -1, "TSContinuous: tStart changed."
    assert tscl.tStop == 8, "TSContinuous: tStop changed."
    assert tscl.bPeriodic, "TSContinuous: bPeriodic changed."
    assert (tsel.vtTimeTrace == vtTimeTrace).all(), "TSEvent: vtTimeTrace changed."
    assert (tsel.vnChannels == vnChannels).all(), "TSEvent: vnChannels changed."
    assert tsel.strName == "events", "TSEvent: strName changed."
    assert tsel.tStart == -1, "TSEvent: tStart changed."
    assert tsel.tStop == 8, "TSEvent: tStop changed."
    assert tsel.bPeriodic, "TSEvent: bPeriodic changed."
    assert tsel.nNumChannels == 3, "TSEvent: nNumChannels changed."
    # - Remove saved files
    remove("test_tsc.npz")
    remove("test_tse.npz")
