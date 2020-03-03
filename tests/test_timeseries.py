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
    from rockpool import TimeSeries
    from rockpool import TSContinuous
    from rockpool import TSEvent
    from rockpool.timeseries import TimeSeries
    from rockpool.timeseries import TSContinuous
    from rockpool.timeseries import TSEvent
    from rockpool.timeseries import set_global_ts_plotting_backend
    from rockpool.timeseries import get_global_ts_plotting_backend
    import rockpool.timeseries as ts


def test_backends():
    """
    Test using the plotting backend setting functions
    """
    from rockpool.timeseries import (
        set_global_ts_plotting_backend,
        get_global_ts_plotting_backend,
    )

    get_global_ts_plotting_backend()
    try:
        set_global_ts_plotting_backend("matplotlib")
        set_global_ts_plotting_backend("holoviews")
    except RuntimeError:
        pass


def test_continuous_operators():
    """
    Test creation and manipulation of a continuous time series
    """
    from rockpool import TSContinuous

    # - Creation
    ts = TSContinuous([0], [0])
    ts = TSContinuous([0, 1, 2, 3], [1, 2, 3, 4])
    ts2 = TSContinuous([1, 2, 3, 4], [5, 6, 7, 8])

    # - Samples don't match time
    with pytest.raises(ValueError):
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
    from rockpool import TSContinuous

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])

    # - Interpolation
    assert ts1(0) == 0
    assert ts1(2) == 2
    assert ts1(1.5) == 1.5

    assert ts1._interpolate(0) == 0
    assert ts1._interpolate(2) == 2
    assert ts1._interpolate(1.5) == 1.5

    # - Delay
    ts2 = ts1.delay(1)
    assert ts1.t_start == 0
    assert ts2.t_start == 1

    ts20 = ts1.start_at_zero()
    assert ts20.t_start == 0

    ts25 = ts1.start_at(5)
    assert ts25.t_start == 5

    # - Contains
    assert ts1.contains(0)
    assert ~ts1.contains(-1)
    assert ts1.contains([0, 1, 2])
    assert ~ts1.contains([0, 1, 2, 3])

    # - Resample
    ts2 = ts1.resample([0.1, 1.1, 1.9])

    # - Merge
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts3 = ts1.merge(ts2, remove_duplicates=True)
    assert np.size(ts3.samples) == 3
    assert np.size(ts1.samples) == 3
    assert np.size(ts2.samples) == 3

    ts3 = ts1.merge(ts2, remove_duplicates=False)
    assert np.size(ts3.samples) == 6

    # - Append
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts3 = ts1.append_t(ts2)
    assert np.size(ts3.times) == 6

    ts3 = ts1.append_c(ts2)
    assert ts3.num_channels == 2

    # - isempty
    assert ~ts1.isempty()
    assert TSContinuous().isempty()

    # - clip
    ts2 = ts1.clip(0.5, 1.5)

    # - Min / Max
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    assert ts1.min == 0
    assert ts1.max == 2


def test_continuous_indexing():
    from rockpool import TSContinuous

    # - Generate series
    times = np.arange(6) * 0.1
    samples = np.arange(6).reshape(-1, 1) + np.arange(4)
    ts = TSContinuous(times, samples)

    # - Indexing time
    ts0 = ts[:]
    assert (ts0.times == times).all()
    assert (ts0.samples == samples).all()

    ts1 = ts[0.2:0.5]
    assert (ts1.times == times[[2, 3, 4]]).all()
    assert (ts1.samples == samples[[2, 3, 4]]).all()

    ts2 = ts[0.1:0.5:-0.2]
    assert (ts2.times == times[[3, 1]]).all()
    assert (ts2.samples == samples[[3, 1]]).all()

    ts3 = ts[[0.5, 0, 0.1]]
    assert (ts3.times == times[[5, 0, 1]]).all()
    assert (ts3.samples == samples[[5, 0, 1]]).all()
    ts4 = ts[0.2]
    assert (ts4.times == times[[2]]).all()
    assert (ts4.samples == samples[2]).all()

    # - Indexing channels
    ts0 = ts[:, :]
    assert (ts0.times == times).all()
    assert (ts0.samples == samples).all()

    ts1 = ts[None, 1:3]
    assert (ts1.times == times).all()
    assert (ts1.samples == samples[:, 1:3]).all()

    ts2 = ts[:, 1::-2]
    assert (ts2.times == times).all()
    assert (ts2.samples == samples[:, [3, 1]]).all()

    ts3 = ts[None, [2, 0, 3]]
    assert (ts3.times == times).all()
    assert (ts3.samples == samples[:, [2, 0, 3]]).all()

    ts4 = ts[:, 2]
    assert (ts4.times == times).all()
    assert (ts4.samples == samples[:, [2]]).all()

    # - Indexing channels and time
    ts0 = ts[:0.4, [3, 1]]
    assert (ts0.times == times[:4]).all()
    assert (ts0.samples == samples[:4, [3, 1]]).all()


def test_continuous_call():
    from rockpool import TSContinuous

    # - Generate series
    times = np.arange(1, 5) * 0.1
    samples = np.arange(4).reshape(-1, 1) + np.arange(2) * 2
    ts = TSContinuous(times, samples)
    ts_empty = TSContinuous()
    ts_single = TSContinuous(2, [3, 2])

    # - Call ts
    assert np.allclose(ts(0.1), np.array([[0, 2]]))
    assert np.allclose(ts(0.25), np.array([[1.5, 3.5]]))
    assert (np.isnan(ts(0))).all() and ts(0).shape == (1, 2)
    samples_ts = ts([0, 0.1, 0.25])
    assert np.allclose(samples_ts[1:], np.array([[0, 2], [1.5, 3.5]]))
    assert (np.isnan(samples_ts[0])).all() and samples_ts.shape == (3, 2)

    # - Call ts_empty
    assert ts_empty(1).shape == (1, 0)
    assert ts_empty([0, 1, 4]).shape == (3, 0)

    # - Call ts_single
    assert (ts_single(2) == np.array([[3, 2]])).all()
    assert np.isnan(ts_single(0)).all() and ts_single(0).shape == (1, 2)
    samples_single = ts_single([0, 2, 1, 3, 2, 4])
    assert (
        np.isnan(samples_single)
        == np.array(
            [2 * [True], 2 * [False], 2 * [True], 2 * [True], 2 * [False], 2 * [True]]
        )
    ).all()
    assert (samples_single[1] == np.array([3, 2])).all()
    assert (samples_single[4] == np.array([3, 2])).all()


def test_continuous_clip():
    from rockpool import TSContinuous

    # - Generate series
    times = np.arange(1, 6) * 0.1
    samples = np.arange(5).reshape(-1, 1) + np.arange(2) * 2
    ts = TSContinuous(times, samples)
    ts_empty = TSContinuous()

    # - Clip ts in time
    assert (ts.clip(0.2, 0.4, include_stop=True).times == times[1:4]).all()
    assert (ts.clip(0.2, 0.4, include_stop=True).samples == samples[1:4]).all()
    assert (ts.clip(0.2, 0.4, include_stop=False).times == times[1:3]).all()
    assert (ts.clip(0.2, 0.4, include_stop=False).samples == samples[1:3]).all()
    ts_limits = ts.clip(0.2, 0.35, include_stop=True, sample_limits=True)
    assert np.allclose(ts_limits.times, np.array([0.2, 0.3, 0.35]))
    expected_samples = np.vstack((samples[1:3], [2.5, 4.5]))
    assert np.allclose(ts_limits.samples, expected_samples)
    ts_beyond = ts.clip(0.4, 0.6, sample_limits=False)
    assert (ts_beyond.times == times[-2:]).all()
    assert (ts_beyond.samples == samples[-2:]).all()
    # - Clip ts channels
    assert (ts.clip(channels=1).times == times).all()
    assert (ts.clip(channels=1).samples == samples[:, [1]]).all()
    assert (ts.clip(channels=[1, 0]).times == times).all()
    assert (ts.clip(channels=[1, 0]).samples == samples[:, [1, 0]]).all()
    # - Clip ts channels and time
    ts_ch_t = ts.clip(0.2, 0.4, channels=1, include_stop=True)
    assert (ts_ch_t.times == times[1:4]).all()
    assert (ts_ch_t.samples == samples[1:4, [1]]).all()

    # - Clip empty
    ts_et = ts_empty.clip(0.2, 0.4, sample_limits=False)
    assert ts_et.isempty()
    assert ts_et.t_start == 0.2 and ts_et.t_stop == 0.4
    with pytest.raises(IndexError):
        ts_empty.clip(channels=0)
        ts_empty.clip(2, 4, channels=0)


def test_continuous_inplace_mutation():
    from rockpool import TSContinuous

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])

    # - Delay
    ts1.delay(1, inplace=True)
    assert ts1.t_start == 1

    # - Resample
    ts1.resample([0.125, 1.1, 1.9], inplace=True)
    assert ts1.t_start == 0.125

    # - Merge
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.merge(ts2, remove_duplicates=True, inplace=True)
    assert np.size(ts1.samples) == 3

    ts3 = ts1.merge(ts2, remove_duplicates=False, inplace=True)
    assert np.size(ts3.samples) == 6

    # - Append
    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.append_t(ts2, inplace=True)
    assert np.size(ts1.times) == 6

    ts1 = TSContinuous([0, 1, 2], [0, 1, 2])
    ts2 = TSContinuous([0, 1, 2], [1, 2, 3])
    ts1.append_c(ts2, inplace=True)
    assert ts1.num_channels == 2

    # - clip
    ts1.clip(0.5, 1.5, inplace=True)
    assert ts1.t_start == 0.5

    # - Start at 0
    ts1.start_at_zero(inplace=True)
    assert ts1.t_start == 0


def test_continuous_append_c():
    """
    Test append_c method of TSContinuous
    """
    from rockpool import TSContinuous

    # - Generate a few TSContinuous objects
    samples = np.random.randint(10, size=(2, 4))
    empty_series = TSContinuous()
    series_list = []
    series_list.append(TSContinuous([1, 2], samples[:2, :2], t_start=-1, t_stop=2))
    series_list.append(TSContinuous([1], samples[0, -2:], t_start=0, t_stop=2))

    # Appending two series
    appended_fromtwo = series_list[0].append_c(series_list[1])
    assert appended_fromtwo.t_start == -1, "Wrong t_start for appended series."
    assert appended_fromtwo.t_stop == 2, "Wrong t_stop for appended series."
    assert (
        appended_fromtwo.times == np.array([1, 2])
    ).all(), "Wrong time trace for appended series."
    assert (
        (appended_fromtwo.samples[:, :2] == samples[:2, :2]).all()
        and (appended_fromtwo.samples[0, -2:] == samples[0, -2:]).all()
        and (np.isnan(appended_fromtwo.samples[1, -2:])).all()
    ), "Wrong samples for appended series."

    # Appending with empty series
    appended_empty_first = empty_series.append_c(series_list[0])
    assert (
        appended_empty_first.t_start == empty_series.t_start
    ), "Wrong t_start when appending with empty."
    assert (
        appended_empty_first.t_stop == empty_series.t_stop
    ), "Wrong t_stop when appending with empty."
    assert (
        appended_empty_first.samples.flatten() == empty_series.samples.flatten()
    ).all(), "Wrong samples when appending with empty"
    assert (
        appended_empty_first.times == empty_series.times
    ).all(), "Wrong time trace when appending with empty"

    appended_empty_last = series_list[0].append_c(empty_series)
    assert (
        appended_empty_last.t_start == series_list[0].t_start
    ), "Wrong t_start when appending with empty."
    assert (
        appended_empty_last.t_stop == series_list[0].t_stop
    ), "Wrong t_stop when appending with empty."
    assert (
        appended_empty_last.samples == series_list[0].samples
    ).all(), "Wrong samples when appending with empty"
    assert (
        appended_empty_last.times == series_list[0].times
    ).all(), "Wrong time trace when appending with empty"


def test_continuous_append_t():
    """
    Test append method of TSEvent
    """
    from rockpool import TSContinuous

    # - Generate a few TSEvent objects
    samples = np.random.randint(10, size=(2, 6))
    empty_series = TSContinuous(t_start=1)
    series_list = []
    series_list.append(TSContinuous([1, 2], samples[:2, :2], t_start=-1, t_stop=3))
    series_list.append(TSContinuous([1], samples[0, 2:4], t_start=0, t_stop=2))
    series_list.append(TSContinuous([1, 3], samples[:2, 4:], t_start=0, t_stop=3))

    # Appending two series
    appended_fromtwo = series_list[0].append_t(series_list[1])
    assert appended_fromtwo.t_start == -1, "Wrong t_start for appended series."
    assert appended_fromtwo.t_stop == 6, "Wrong t_stop for appended series."
    assert (
        appended_fromtwo.times == np.array([1, 2, 5])
    ).all(), "Wrong time trace for appended series."
    assert (
        (appended_fromtwo.samples[:2] == samples[:2, :2]).all()
        and (appended_fromtwo.samples[2:] == samples[[0], 2:4])
    ).all(), "Wrong samples for appended series."

    # Appending with empty series
    appended_empty_first = empty_series.append_t(series_list[0])
    assert (
        appended_empty_first.t_start == empty_series.t_start
    ), "Wrong t_start when appending with empty."
    assert (
        appended_empty_first.t_stop == series_list[0].duration + empty_series.t_start
    ), "Wrong t_stop when appending with empty."
    assert (
        appended_empty_first.samples == series_list[0].samples
    ).all(), "Wrong samples when appending with empty"
    assert (
        appended_empty_first.times == series_list[0].times + 2
    ).all(), "Wrong time trace when appending with empty"

    appended_empty_last = series_list[0].append_t(empty_series)
    assert (
        appended_empty_last.t_start == series_list[0].t_start
    ), "Wrong t_start when appending with empty."
    assert (
        # - 1 is offset, which is np.median(np.diff(series_list[0].times))=1
        appended_empty_last.t_stop
        == series_list[0].t_stop + empty_series.duration
    ), "Wrong t_stop when appending with empty."
    assert (
        appended_empty_last.samples == series_list[0].samples
    ).all(), "Wrong samples when appending with empty"
    assert (
        appended_empty_last.times == series_list[0].times
    ).all(), "Wrong time trace when appending with empty"

    # Appending multiple time series
    appended_fromthree = series_list[0].append_t(series_list[1:], offset=None)
    exptd_offset = np.median(np.diff(series_list[0].times))
    exptd_ts2_delay = exptd_offset + series_list[0].t_stop - series_list[1].t_start
    # - No offset between ts2 and ts3 because ts2 has only one element
    exptd_ts3_delay = exptd_ts2_delay + series_list[1].t_stop - series_list[2].t_start
    exptd_ts2_times = series_list[1].times + exptd_ts2_delay
    exptd_ts3_times = series_list[2].times + exptd_ts3_delay

    assert (
        appended_fromthree.times
        == np.r_[series_list[0].times, exptd_ts2_times, exptd_ts3_times]
    ).all(), "Wrong time trace when appending from list"
    assert (
        appended_fromthree.samples
        == np.vstack([series.samples for series in series_list])
    ).all(), "Wrong samples when appending from list"

    # - Generating from list of TSContinuous
    appended_fromlist = TSContinuous.concatenate_t(series_list)

    assert (
        appended_fromthree.times == appended_fromlist.times
    ).all(), "Wrong time trace when appending from list"
    assert (
        appended_fromthree.samples == appended_fromlist.samples
    ).all(), "Wrong samples when appending from list"


def test_continuous_merge():
    """
    Test merge method of TSContinuous
    """
    from rockpool import TSContinuous

    # - Generate a few TSEvent objects
    samples = np.random.randint(10, size=(2, 6))
    empty_series = TSContinuous(t_start=1)
    series_list = []
    series_list.append(TSContinuous([1, 2], samples[:2, :2], t_start=-1, t_stop=3))
    series_list.append(TSContinuous([1.5], samples[0, 2:4], t_start=0, t_stop=4))
    series_list.append(TSContinuous([2, 2.5], samples[:2, 4:6], t_start=-2, t_stop=3))

    # Merging two series
    merged_fromtwo = series_list[0].merge(series_list[1])
    assert merged_fromtwo.t_start == -1, "Wrong t_start for merged series."
    assert merged_fromtwo.t_stop == 4, "Wrong t_stop for merged series."
    assert (
        merged_fromtwo.times == np.array([1, 1.5, 2])
    ).all(), "Wrong time trace for merged series."
    correct_samples = np.vstack((samples[0, :2], samples[0, 2:4], samples[1, :2]))
    assert (
        merged_fromtwo.samples == correct_samples
    ).all(), "Wrong samples for merged series."

    # Merging with empty series
    merged_empty_first = empty_series.merge(series_list[0])
    assert (
        merged_empty_first.t_start == series_list[0].t_start
    ), "Wrong t_start when merging with empty."
    assert (
        merged_empty_first.t_stop == series_list[0].t_stop
    ), "Wrong t_stop when merging with empty."
    assert (
        merged_empty_first.samples == series_list[0].samples
    ).all(), "Wrong samples when merging with empty"
    assert (
        merged_empty_first.times == series_list[0].times
    ).all(), "Wrong time trace when merging with empty"

    merged_empty_last = series_list[0].merge(empty_series)
    assert (
        merged_empty_last.t_start == series_list[0].t_start
    ), "Wrong t_start when merging with empty."
    assert (
        merged_empty_last.t_stop == series_list[0].t_stop
    ), "Wrong t_stop when merging with empty."
    assert (
        merged_empty_last.samples == series_list[0].samples
    ).all(), "Wrong samples when merging with empty"
    assert (
        merged_empty_last.times == series_list[0].times
    ).all(), "Wrong time trace when merging with empty"

    # Merging with list of series
    merged_with_list = empty_series.merge(series_list, remove_duplicates=True)
    assert (
        merged_with_list.num_channels == 2
    ), "Wrong channel count when merging with list."
    assert merged_with_list.t_start == -2, "Wrong t_start when merging with list."
    assert merged_with_list.t_stop == 4, "Wrong t_stop when merging with list."
    assert (
        merged_with_list.times == np.array([1, 1.5, 2, 2.5])
    ).all(), "Wrong time trace when merging with list."
    assert (
        merged_with_list.samples
        == np.vstack((samples[0, :2], samples[0, 2:4], samples[1, :2], samples[1, 4:6]))
    ).all(), "Wrong samples when merging with list."


def test_event_call():
    from rockpool import TSEvent

    # - Generate series
    times = [1, 3, 4, 5, 7]
    channels = [0, 0, 1, 2, 1]
    ts = TSEvent(times, channels)
    ts_empty = TSEvent()

    # - Call ts with times
    assert (ts()[0] == times).all()
    assert (ts()[1] == channels).all()
    assert (ts(5)[0] == times[-2:]).all()
    assert (ts(5)[1] == channels[-2:]).all()
    assert (ts(None, 5)[0] == times[:-2]).all()
    assert (ts(None, 5)[1] == channels[:-2]).all()
    assert (ts(2, 5)[0] == times[1:3]).all()
    assert (ts(2, 5)[1] == channels[1:3]).all()
    assert (ts(2, 5, include_stop=True)[0] == times[1:4]).all()
    assert (ts(2, 5, include_stop=True)[1] == channels[1:4]).all()
    assert (ts(8, 9)[0] == []).all()
    assert (ts(8, 9)[1] == []).all()
    # - Call ts with channels
    assert (ts(channels=[0, 2])[0] == [1, 3, 5]).all()
    assert (ts(channels=[0, 2])[1] == [0, 0, 2]).all()
    assert (ts(channels=0)[0] == [1, 3]).all()
    assert (ts(channels=0)[1] == [0, 0]).all()
    with pytest.raises(IndexError):
        ts(channels=4)
    # - Call ts with channels and time
    assert (ts(2, 6, channels=[0, 2])[0] == [3, 5]).all()
    assert (ts(2, 6, channels=[0, 2])[1] == [0, 2]).all()

    # - Call empty
    assert (ts_empty(2, 5)[0] == []).all()
    assert (ts_empty(2, 5)[1] == []).all()
    with pytest.raises(IndexError):
        assert (ts_empty(2, 5, channels=4)[0] == []).all()
        assert (ts_empty(2, 5, channels=4)[1] == []).all()


def test_event_indexing():
    from rockpool import TSEvent

    # - Generate series
    times = [1, 3, 4, 5, 7]
    channels = [0, 0, 1, 2, 1]
    ts = TSEvent(times, channels)
    ts_empty = TSEvent()

    # - Indexing ts
    assert (ts[2].times == [4]).all()
    assert (ts[2].channels == [1]).all()
    assert (ts[2:5].times == times[2:5]).all()
    assert (ts[2:5].channels == channels[2:5]).all()
    assert (ts[[4, 0, 1]].times == np.asarray(times)[[4, 0, 1]]).all()
    assert (ts[[4, 0, 1]].channels == np.asarray(channels)[[4, 0, 1]]).all()

    # - Indexing empty
    with pytest.raises(IndexError):
        ts_empty[0]
    with pytest.raises(IndexError):
        ts_empty[[0, 1]]


def test_event_raster():
    """
    Test TSEvent raster function on merging other time series events
    """
    from rockpool import TSEvent

    # - Build a test event time series
    testTSEvent = TSEvent([0, 30], 0, num_channels=4)

    # - Default operation, ignoring end time step
    raster = testTSEvent.raster(dt=1)
    assert raster.shape == (30, 4)

    # - Include end time step
    raster = testTSEvent.raster(dt=1, include_t_stop=True)
    assert raster.shape == (31, 4)

    # - Use a dt that is a non-modulo of duration
    raster = TSEvent([0, 1], 0).raster(dt=0.9)
    assert raster.shape == (2, 1)

    raster = TSEvent([0, 1], 0).raster(dt=0.9, include_t_stop=True)
    assert raster.shape == (2, 1)

    raster = TSEvent([0, 1], 0).raster(dt=1.1, include_t_stop=True)
    assert raster.shape == (1, 1)


def test_event_raster_explicit_num_channels():
    """
    Test TSEvent raster method when the function is initialized with explicit number of Channels
    """
    from rockpool import TSEvent

    testTSEvent = TSEvent([0, 30], 0, num_channels=5)

    raster = testTSEvent.raster(dt=1)
    assert raster.shape == (30, 5)


def test_event_empty():
    """
    Test TSEvent instantiation with empty objects or None
    """
    from rockpool import TSEvent

    testTSEvent = TSEvent([], [])
    assert testTSEvent.num_channels == 0

    testTSEvent = TSEvent(None, None)
    assert testTSEvent.num_channels == 0

    testTSEvent = TSEvent()
    assert testTSEvent.num_channels == 0


def test_event_clip():
    from rockpool import TSEvent

    # - Generate series
    times = [1, 3, 4, 5, 7]
    channels = [0, 0, 1, 2, 1]
    ts = TSEvent(times, channels)
    ts_empty = TSEvent(num_channels=2)

    # - Clip ts in time
    assert (ts.clip(2, 4, include_stop=True).times == times[1:3]).all()
    assert (ts.clip(2, 4, include_stop=True).channels == channels[1:3]).all()
    assert (ts.clip(2, 4, include_stop=False).times == times[1:2]).all()
    assert (ts.clip(2, 4, include_stop=False).channels == channels[1:2]).all()
    assert ts.clip(2, 4, include_stop=False).num_channels == 3
    assert (ts.clip(4, 6, remap_channels=True).times == times[2:4]).all()
    assert (ts.clip(4, 6, remap_channels=True).channels == [0, 1]).all()
    assert ts.clip(8, 9).isempty()
    # - Clip ts channels
    assert (ts.clip(channels=[0, 1]).times == np.array([1, 3, 4, 7])).all()
    assert (ts.clip(channels=[0, 1]).channels == np.array([0, 0, 1, 1])).all()
    # - Clip ts channels and time
    assert (ts.clip(2, 6, channels=[0, 2]).times == np.array([3, 5])).all()
    assert (ts.clip(2, 6, channels=[0, 2]).channels == np.array([0, 2])).all()

    # - Clip empty
    assert ts_empty.clip(2, 3).isempty()
    assert ts_empty.clip(2, 3).num_channels == 2
    assert ts_empty.clip(2, 3).t_start == 2
    assert ts_empty.clip(2, 3).t_stop == 3
    assert ts_empty.clip(channels=0).isempty()
    assert ts_empty.clip(channels=0).num_channels == 2
    assert ts_empty.clip(channels=0).t_start == 0
    assert ts_empty.clip(channels=0).t_stop == 0
    assert ts_empty.clip(2, 3, channels=0).isempty()
    assert ts_empty.clip(2, 3, channels=0).num_channels == 2
    assert ts_empty.clip(2, 3, channels=0, remap_channels=True).num_channels == 1
    assert ts_empty.clip(2, 3, channels=0).t_start == 2
    assert ts_empty.clip(2, 3, channels=0).t_stop == 3


def test_event_append_c():
    """
    Test append_c method of TSEvent
    """
    from rockpool import TSEvent

    # - Generate a few TSEvent objects
    empty_series = TSEvent()
    series_list = []
    series_list.append(TSEvent([1, 2], [2, 0], t_start=-1, t_stop=2))
    series_list.append(TSEvent([0, 1, 4], [1, 1, 0], t_start=0, t_stop=6))
    series_list.append(TSEvent([1], [0], t_start=0, t_stop=2, num_channels=5))

    # Appending two series
    appended_fromtwo = series_list[0].append_c(series_list[2])
    assert (
        appended_fromtwo.num_channels == 8
    ), "Wrong channel count for appended series."
    assert appended_fromtwo.t_start == -1, "Wrong t_start for appended series."
    assert appended_fromtwo.t_stop == 2, "Wrong t_stop for appended series."
    assert (
        appended_fromtwo.times == np.array([1, 1, 2])
    ).all(), "Wrong time trace for appended series."
    assert (appended_fromtwo.channels == np.array([2, 3, 0])).all() or (
        appended_fromtwo.channels == np.array([3, 2, 0])
    ).all(), "Wrong channels for appended series."

    # Appending with empty series
    appended_empty_first = empty_series.append_c(series_list[0])
    assert (
        appended_empty_first.t_start == series_list[0].t_start
    ), "Wrong t_start when appending with empty."
    assert (
        appended_empty_first.t_stop == series_list[0].t_stop
    ), "Wrong t_stop when appending with empty."
    assert (
        appended_empty_first.num_channels == series_list[0].num_channels
    ), "Wrong channel count when appending with empty."
    assert (
        appended_empty_first.channels == series_list[0].channels
    ).all(), "Wrong channels when appending with empty"
    assert (
        appended_empty_first.times == series_list[0].times
    ).all(), "Wrong time trace when appending with empty"
    appended_empty_last = series_list[0].append_c(empty_series)
    assert (
        appended_empty_last.t_start == series_list[0].t_start
    ), "Wrong t_start when appending with empty."
    assert (
        appended_empty_last.t_stop == series_list[0].t_stop
    ), "Wrong t_stop when appending with empty."
    assert (
        appended_empty_last.num_channels == series_list[0].num_channels
    ), "Wrong channel count when appending with empty."
    assert (
        appended_empty_last.channels == series_list[0].channels
    ).all(), "Wrong channels when appending with empty"
    assert (
        appended_empty_last.times == series_list[0].times
    ).all(), "Wrong time trace when appending with empty"

    # Appending with list of series
    appended_with_list = empty_series.append_c(series_list)
    assert (
        appended_with_list.num_channels == 10
    ), "Wrong channel count when appending with list."
    assert appended_with_list.t_start == -1, "Wrong t_start when appending with list."
    assert appended_with_list.t_stop == 6, "Wrong t_stop when appending with list."
    assert (
        appended_with_list.times == np.array([0, 1, 1, 1, 2, 4])
    ).all(), "Wrong time trace when appending with list."
    # Allow permutations of events 1 to 3 because they have same time
    assert (
        appended_with_list.channels[[0, 4, 5]] == np.array([4, 0, 3])
    ).all() and set(appended_with_list.channels[1:4]) == set(
        (2, 4, 5)
    ), "Wrong channels when appending with list."


def test_event_append_t():
    """
    Test append_t method of TSEvent
    """
    from rockpool import TSEvent

    # - Generate a few TSEvent objects
    empty_series = TSEvent()
    series_list = []
    series_list.append(TSEvent([1, 2], [2, 0], t_start=-1, t_stop=2))
    series_list.append(TSEvent([0, 1, 4], [1, 1, 0], t_start=0, t_stop=6))
    series_list.append(TSEvent([1], [0], t_start=0, t_stop=2, num_channels=5))

    # Appending two series
    appended_fromtwo = series_list[0].append_t(series_list[2])
    assert (
        appended_fromtwo.num_channels == 5
    ), "Wrong channel count for appended series."
    assert appended_fromtwo.t_start == -1, "Wrong t_start for appended series."
    assert appended_fromtwo.t_stop == 4, "Wrong t_stop for appended series."
    assert (
        appended_fromtwo.times == np.array([1, 2, 3])
    ).all(), "Wrong time trace for appended series."
    assert (
        appended_fromtwo.channels == np.array([2, 0, 0])
    ).all(), "Wrong channels for appended series."

    # Appending with empty series
    appended_empty_first = empty_series.append_t(series_list[0])
    assert (
        appended_empty_first.t_start == empty_series.t_start
    ), "Wrong t_start when appending with empty."
    assert (
        appended_empty_first.t_stop == series_list[0].duration + empty_series.t_start
    ), "Wrong t_stop when appending with empty."
    assert (
        appended_empty_first.num_channels == series_list[0].num_channels
    ), "Wrong channel count when appending with empty."
    assert (
        appended_empty_first.channels == series_list[0].channels
    ).all(), "Wrong channels when appending with empty"
    assert (
        appended_empty_first.times == np.array([2, 3])
    ).all(), "Wrong time trace when appending with empty"
    appended_empty_last = series_list[0].append_t(empty_series)
    assert (
        appended_empty_last.t_start == series_list[0].t_start
    ), "Wrong t_start when appending with empty."
    assert (
        appended_empty_last.t_stop == series_list[0].t_stop
    ), "Wrong t_stop when appending with empty."
    assert (
        appended_empty_last.num_channels == series_list[0].num_channels
    ), "Wrong channel count when appending with empty."
    assert (
        appended_empty_last.channels == series_list[0].channels
    ).all(), "Wrong channels when appending with empty"
    assert (
        appended_empty_last.times == series_list[0].times
    ).all(), "Wrong time trace when appending with empty"

    # Appending with list of series
    appended_with_list = empty_series.append_t(series_list, offset=[3, 2, 1])
    assert (
        appended_with_list.num_channels == 5
    ), "Wrong channel count when appending with list."
    assert appended_with_list.t_start == 0, "Wrong t_start when appending with list."
    assert appended_with_list.t_stop == 17, "Wrong t_stop when appending with list."
    assert (
        appended_with_list.times == np.array([5, 6, 8, 9, 12, 16])
    ).all(), "Wrong time trace when appending with list."
    assert (
        appended_with_list.channels == np.array([2, 0, 1, 1, 0, 0])
    ).all(), "Wrong channels when appending with list."

    # - Generating from list of TSContinuous
    # First offset changed to match `appended_with_list`
    # (where empty series has t_stop=0, causing first series to be shifted as it has t_start = -1)
    appended_fromlist = TSEvent.concatenate_t(series_list, offset=[4, 2, 1])

    assert (
        appended_with_list.times == appended_fromlist.times
    ).all(), "Wrong time trace when appending from list"
    assert (
        appended_with_list.channels == appended_fromlist.channels
    ).all(), "Wrong channels when appending from list"


def test_event_merge():
    """
    Test merge method of TSEvent
    """
    from rockpool import TSEvent

    # - Generate a few TSEvent objects
    empty_series = TSEvent()
    series_list = []
    series_list.append(TSEvent([1, 2], [2, 0], t_start=-1, t_stop=2))
    series_list.append(TSEvent([0, 1, 4], [1, 1, 0], t_start=0, t_stop=6))
    series_list.append(TSEvent([1], [0], t_start=0, t_stop=2, num_channels=5))

    # Merging two series
    merged_fromtwo = series_list[0].merge(series_list[2])
    assert merged_fromtwo.num_channels == 5, "Wrong channel count for merged series."
    assert merged_fromtwo.t_start == -1, "Wrong t_start for merged series."
    assert merged_fromtwo.t_stop == 2, "Wrong t_stop for merged series."
    assert (
        merged_fromtwo.times == np.array([1, 1, 2])
    ).all(), "Wrong time trace for merged series."
    assert (merged_fromtwo.channels == np.array([2, 0, 0])).all() or (
        merged_fromtwo.channels == np.array([0, 2, 0])
    ).all(), "Wrong channels for merged series."

    # Merging with empty series
    merged_empty_first = empty_series.merge(series_list[0])
    assert (
        merged_empty_first.t_start == series_list[0].t_start
    ), "Wrong t_start when merging with empty."
    assert (
        merged_empty_first.t_stop == series_list[0].t_stop
    ), "Wrong t_stop when merging with empty."
    assert (
        merged_empty_first.num_channels == series_list[0].num_channels
    ), "Wrong channel count when merging with empty."
    assert (
        merged_empty_first.channels == series_list[0].channels
    ).all(), "Wrong channels when merging with empty"
    assert (
        merged_empty_first.times == series_list[0].times
    ).all(), "Wrong time trace when merging with empty"
    merged_empty_last = series_list[0].merge(empty_series)
    assert (
        merged_empty_last.t_start == series_list[0].t_start
    ), "Wrong t_start when merging with empty."
    assert (
        merged_empty_last.t_stop == series_list[0].t_stop
    ), "Wrong t_stop when merging with empty."
    assert (
        merged_empty_last.num_channels == series_list[0].num_channels
    ), "Wrong channel count when merging with empty."
    assert (
        merged_empty_last.channels == series_list[0].channels
    ).all(), "Wrong channels when merging with empty"
    assert (
        merged_empty_last.times == series_list[0].times
    ).all(), "Wrong time trace when merging with empty"

    # Merging with list of series
    merged_with_list = empty_series.merge(series_list, delay=[3, 2, 1])
    assert (
        merged_with_list.num_channels == 5
    ), "Wrong channel count when merging with list."
    assert merged_with_list.t_start == 0, "Wrong t_start when merging with list."
    assert merged_with_list.t_stop == 8, "Wrong t_stop when merging with list."
    assert (
        merged_with_list.times == np.array([2, 2, 3, 4, 5, 6])
    ).all(), "Wrong time trace when merging with list."
    assert (merged_with_list.channels == np.array([1, 0, 1, 2, 0, 0])).all() or (
        merged_with_list.channels == np.array([0, 1, 1, 2, 0, 0])
    ).all(), "Wrong channels when merging with list."


def test_save_load():
    """
    Test saving and loading function for timeseries
    """
    from rockpool import TSEvent, TSContinuous, load_ts_from_file
    from os import remove

    # - Generate time series objects
    times = [1, 3, 6]
    samples = np.random.randn(3)
    channels = [0, 1, 1]
    tsc = TSContinuous(
        times, samples, t_start=-1, t_stop=8, periodic=True, name="continuous"
    )
    tse = TSEvent(
        times,
        channels,
        num_channels=3,
        t_start=-1,
        t_stop=8,
        periodic=True,
        name="events",
    )
    # - Store objects
    tsc.save("test_tsc")
    tse.save("test_tse")
    # - Load objects
    tscl = load_ts_from_file("test_tsc.npz")
    tsel = load_ts_from_file("test_tse.npz")
    # - Verify that attributes are still correct
    assert (tscl.times == times).all(), "TSContinuous: times changed."
    assert (
        tscl.samples == samples.reshape(-1, 1)
    ).all(), "TSContinuous: samples changed."
    assert tscl.name == "continuous", "TSContinuous: name changed."
    assert tscl.t_start == -1, "TSContinuous: t_start changed."
    assert tscl.t_stop == 8, "TSContinuous: t_stop changed."
    assert tscl.periodic, "TSContinuous: periodic changed."
    assert (tsel.times == times).all(), "TSEvent: times changed."
    assert (tsel.channels == channels).all(), "TSEvent: channels changed."
    assert tsel.name == "events", "TSEvent: name changed."
    assert tsel.t_start == -1, "TSEvent: t_start changed."
    assert tsel.t_stop == 8, "TSEvent: t_stop changed."
    assert tsel.periodic, "TSEvent: periodic changed."
    assert tsel.num_channels == 3, "TSEvent: num_channels changed."
    # - Remove saved files
    remove("test_tsc.npz")
    remove("test_tse.npz")


def test_event_raster_periodic_iss5():
    from rockpool import TSEvent

    # - Build a periodic event time series
    ts = TSEvent([0, 1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1, 0], periodic=True)

    # - Test rasterisation
    raster = ts.raster(t_start=0, t_stop=10, dt=1)
    assert np.all(
        raster
        == [
            [True, False],
            [False, True],
            [True, False],
            [False, True],
            [True, False],
            [False, True],
            [True, False],
            [False, True],
            [True, False],
            [False, True],
        ]
    )

    # - Build a non-periodic event time series
    ts = TSEvent([0, 1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1, 0], periodic=False)

    # - Test rasterisation
    raster = ts.raster(t_start=0, t_stop=10, dt=1)
    assert np.all(
        raster
        == [
            [True, False],
            [False, True],
            [True, False],
            [False, True],
            [True, False],
            [False, True],
            [True, False],
            [False, False],
            [False, False],
            [False, False],
        ]
    )


def test_event_delay():
    from rockpool import TSEvent

    ts = TSEvent([2, 4, 6], [0, 1, 0])
    assert ts.t_start == 2

    ts0 = ts.delay(1)
    assert ts0.t_start == 3

    ts.delay(1, inplace=True)
    assert ts.t_start == 3

    ts0 = ts.start_at_zero()
    assert ts.t_start == 3
    assert ts0.t_start == 0

    ts0 = ts.start_at_zero(inplace=True)
    assert ts.t_start == 0
    assert ts0.t_start == 0

    ts5 = ts.start_at(5)
    assert ts5.t_start == 5
    ts5.start_at(4, inplace=True)
    assert ts5.t_start == 4


def test_rounding():
    from rockpool import TSContinuous

    t = np.arange(0, 1, 0.001)
    v = np.sin(t)

    ts = TSContinuous(t, v)

    ts.t_stop = t[-1] - 1e-10 


