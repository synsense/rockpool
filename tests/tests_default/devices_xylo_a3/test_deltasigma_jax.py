from turtle import shape
import pytest


def test_imports():
    from rockpool.devices.xylo.syns65302.afe.pdm import DeltaSigma


def test_deltasigma_jax():
    """
    this module verifies that the jax and the python version of deltasigma module are compatible.

    NOTE: since deltasigma module is quite sensitive to noise and jax version uses float32 rather than float64,
    there might be some difeerences in the generated PDM signals.

    This test verifies that although the PDM signals may be different, the resulting recovered signals are still close.
    """
    from rockpool.devices.xylo.syns65302.afe.pdm import DeltaSigma
    from rockpool.devices.xylo.syns65302.afe.params import (
        PDM_SAMPLING_RATE,
    )
    import numpy as np
    from numpy.linalg import norm
    import time

    # parameters of DeltaSigma module
    bandwidth = 20_000
    fs = PDM_SAMPLING_RATE
    order = 4
    amplitude = 2.3

    ds = DeltaSigma(amplitude=amplitude, bandwidth=bandwidth, order=order, fs=fs)

    # build a sinusoid signal
    freq_min = 1000
    freq_max = 10_000
    num_freq = 5
    freq_vec = freq_min + (freq_max - freq_min) * np.random.rand(num_freq)
    phase_vec = 2 * np.pi * np.random.rand(num_freq)
    amp_vec = np.sort(np.random.rand(num_freq))

    num_periods = 200
    duration = num_periods / freq_min

    time_vec = np.arange(0, duration, step=1 / fs)
    sig_in = np.einsum(
        "i, ij -> j",
        amp_vec,
        np.sin(
            2 * np.pi * freq_vec.reshape(-1, 1) * time_vec.reshape(1, -1)
            + phase_vec.reshape(-1, 1)
        ),
    )

    safe_amplitude = 0.9
    sig_in = safe_amplitude * sig_in / np.max(np.abs(sig_in))

    sig_out_Q_jax, sig_out_jax, _, state_jax = ds.evolve(
        sig_in=sig_in,
        sample_rate=fs,
        python_version=False,
        record=True,
    )

    sig_out_Q_py, sig_out_py, _, state_py = ds.evolve(
        sig_in=sig_in,
        sample_rate=fs,
        python_version=True,
        record=True,
    )

    assert sig_out_Q_py.shape == sig_out_Q_jax.shape
    assert sig_out_py.shape == sig_out_jax.shape
    assert state_py.shape == state_jax.shape

    # * verify and comapre jax version with pythion version
    # correlation between outputs
    sig_out_Q_corr = np.mean(sig_out_Q_jax * sig_out_Q_py) / amplitude**2

    # recover the signals
    sig_rec_jax, _ = ds.recover(
        bin_in=sig_out_Q_jax,
    )

    sig_rec_py, _ = ds.recover(bin_in=sig_out_Q_py)

    assert sig_rec_py.shape == sig_rec_jax.shape

    # compute relative distance between original and recovered versions
    delay = abs(len(sig_in) - np.argmax(np.convolve(sig_in[::-1], sig_rec_jax)))
    sig_list = [sig_in[:-delay], sig_rec_jax[delay:], sig_rec_py[delay:]]
    norm_distance_mat = np.zeros((3, 3))

    for i, sig1 in enumerate(sig_list):
        for j, sig2 in enumerate(sig_list):
            norm_dist = norm(sig1 - sig2) / np.sqrt(norm(sig1) * norm(sig2))
            norm_distance_mat[i, j] = norm_dist

    EPS = 0.1

    assert np.all(norm_distance_mat < EPS)
