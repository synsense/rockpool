import pytest


def test_imports():
    from rockpool.devices.xylo.syns61201 import AFESim
    from rockpool.devices.xylo.syns61201.afe_spike_generation import (
        _encode_spikes,
        # _encode_spikes_cpp,
        _encode_spikes_jax,
        _encode_spikes_python,
    )


# ===========================================================================
#        metrics for measuring distortion between two spike trains
#
# NOTE: this metric is needed since due to floating-point precision issues,
#       the generated spikes may be different.
# ===========================================================================
import numpy as np
from numpy.linalg import norm
import time


def rel_error_levy_global(v1, v2):
    """Levy distance between to CDFs: a good measure of distance between two spike trains"""
    w1 = v1.ravel()
    w2 = v2.ravel()

    return np.max(np.abs(w1.cumsum() - w2.cumsum())) / np.min(
        [norm(w1, 1), norm(w2, 1)]
    )


def rel_error_levy_local(v1, v2):
    """Levy distance between to CDFs: a good measure of distance between two spike trains"""
    # we do reverse since the difference between spikes get more amplified as we approach the end
    w1 = v1.ravel()[::-1]
    w2 = v2.ravel()[::-1]

    w1_acc = w1.cumsum()
    w2_acc = w2.cumsum()

    diff_acc = np.abs(w1_acc - w2_acc)
    min_acc = np.min(np.vstack([w1_acc, w2_acc]), axis=0)

    dist_levy_local = diff_acc / (1 + min_acc)

    # take mean to get rid of outliers at the start and end
    return np.mean(dist_levy_local)


def test_spike_generation():
    import pytest

    pytest.importorskip("jax")

    from rockpool.devices.xylo.syns61201 import AFESim
    from rockpool.devices.xylo.syns61201.afe_spike_generation import (
        _encode_spikes,
        # _encode_spikes_cpp,
        _encode_spikes_jax,
        _encode_spikes_python,
    )
    from rockpool.timeseries import TSEvent

    # * parameters of AFESim
    vcc = 1.1
    thr_up = 0.5
    v2i_gain = 3.333e-7
    c_iaf = 5.0e-12
    leakage = 1.0e-9

    num_channels = 16
    raster_dt = 1 / 100
    max_num_spikes = 15
    dt = 1 / 50_000

    # * signal specification
    sig_len = 220_000
    amplitude = 10.0e-3  # 10 mV

    sig_rect = amplitude * np.random.rand(num_channels, sig_len).T

    params = {
        "initial_state": np.zeros(num_channels),
        "dt": dt,
        "data": sig_rect,
        "v2i_gain": v2i_gain,
        "c_iaf": c_iaf,
        "leakage": leakage,
        "thr_up": thr_up,
        "vcc": vcc,
    }

    # ===========================================================================
    #                            Jax version
    # ===========================================================================
    # * first run: dummy to compile jit in case jit is used

    start = time.time()
    spikes_jax, final_state_jax = _encode_spikes_jax(**params)

    duration_jax_first = time.time() - start

    # run it again
    start = time.time()
    spikes_jax, final_state_jax = _encode_spikes_jax(**params)

    # raster the spikes
    spikes_raster_jax = TSEvent.from_raster(spikes_jax, dt=dt).raster(
        dt=raster_dt, add_events=True
    )
    spikes_raster_jax[spikes_raster_jax > max_num_spikes] = max_num_spikes

    duration_jax = time.time() - start

    # ===========================================================================
    #                            C++ version
    # ===========================================================================
    # start = time.time()
    # spikes_cpp, final_state_cpp = _encode_spikes_cpp(**params)

    # spikes_raster_cpp = np.asarray(spikes_cpp).T
    # final_state_cpp = np.asarray(final_state_cpp)

    # duration_cpp = time.time() - start

    # ===========================================================================
    #                           Python version
    # ===========================================================================
    start = time.time()
    spikes_py, final_state_py = _encode_spikes_python(**params)

    spikes_raster_py = np.asarray(spikes_py).T
    final_state_py = np.asarray(final_state_py)

    duration_py = time.time() - start

    # ===========================================================================
    #                            Report run-time
    # ===========================================================================
    print("jax version first run-time: ", duration_jax_first)
    print("jax version second run-time: ", duration_jax)
    # print("C++ version second run-time: ", duration_cpp)
    print("python version run-time: ", duration_py)

    # ===========================================================================
    #                            Validate spike rates
    # ===========================================================================
    spike_rate_jax = spikes_jax.mean(0) / dt
    # spike_rate_cpp = spikes_cpp.mean(0) / dt
    spike_rate_py = spikes_py.mean(0) / dt

    EPS = 0.001
    REL_ERR_MAX = 0.001

    rel_error_jax = norm(spike_rate_jax - spike_rate_py) / (
        np.sqrt(norm(spike_rate_py) * norm(spike_rate_jax)) + EPS
    )
    assert rel_error_jax < REL_ERR_MAX

    # rel_error_cpp = norm(spike_rate_cpp - spike_rate_py) / (
    #     np.sqrt(norm(spike_rate_py) * norm(spike_rate_cpp)) + EPS
    # )
    # assert rel_error_cpp < REL_ERR_MAX

    # ===========================================================================
    #               Validate local difference between the spikes
    # NOTE: this is needed due to slight variation between the spikes because of
    #       floating-point precision issues
    # ===========================================================================
    assert rel_error_levy_local(spikes_py, spikes_jax) < REL_ERR_MAX
    assert rel_error_levy_global(spikes_py, spikes_jax) < REL_ERR_MAX

    # assert rel_error_levy_local(spikes_py, spikes_cpp) < REL_ERR_MAX
    # assert rel_error_levy_global(spikes_py, spikes_cpp) < REL_ERR_MAX


def test_afesim2():
    """tests the spike generation in AFESim improved version"""

    from rockpool.devices.xylo.syns61201 import AFESim

    # * signal specification
    amplitude = 100.0e-3  # 10 mV
    fs = 120_000
    duration = 5  # in seconds
    sig_len = int(duration * fs)

    sig_rect = amplitude * np.random.rand(sig_len).T

    afesim = AFESim(fs=fs)
    _, num_channels = afesim.shape

    # * spikes generated by AFESim
    spikes, *_ = afesim.evolve(
        input=sig_rect,
        record=True,
    )

    assert spikes.shape == (sig_len, num_channels)
    assert spikes.max() <= 1
    assert spikes.min() >= 0.0

    # * rastered spikes
    spikes_rast = afesim.raster(spikes=spikes)

    # check the maximum number of spikes in a raster period
    assert 0 <= spikes_rast.max() <= afesim.max_spike_per_raster_period

    raster_len = int(duration / afesim.raster_period)
    assert spikes_rast.shape[0] == raster_len

    # * check average spike rate: both for the original and also rastered version
    spike_rate_avg_original = spikes.mean(0) * fs
    spike_rate_avg_rastered = spikes_rast.mean(0) / afesim.raster_period

    EPS = 0.01
    REL_ERR_MAX = 0.01

    rel_distance = norm(spike_rate_avg_original - spike_rate_avg_rastered) / (
        np.sqrt(norm(spike_rate_avg_rastered) * norm(spike_rate_avg_original)) + EPS
    )
    assert rel_distance <= REL_ERR_MAX
