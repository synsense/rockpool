# ---------------------------------------------------------------------------
# This module tests the jjax implementation of deltasigma module
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 30.06.2023
# ---------------------------------------------------------------------------

from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.pdm_adc import (
    DeltaSigma,
    PDM_SAMPLING_RATE,
    PDM_FILTER_DECIMATION_FACTOR,
)
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time


def test_jax():
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

    start = time.time()
    sig_out_Q_jax, sig_out_jax, _, state_jax = ds.evolve(
        sig_in=sig_in,
        sample_rate=fs,
        python_version=False,
        record=True,
    )
    duration_jax_1 = time.time() - start

    start = time.time()
    sig_out_Q_jax, sig_out_jax, _, state_jax = ds.evolve(
        sig_in=sig_in,
        sample_rate=fs,
        python_version=False,
        record=True,
    )
    duration_jax_2 = time.time() - start

    start = time.time()
    sig_out_Q_py, sig_out_py, _, state_py = ds.evolve(
        sig_in=sig_in,
        sample_rate=fs,
        python_version=True,
        record=True,
    )
    duration_py = time.time() - start

    # * verify and comapre jax version with pythion version
    # correlation between outputs
    sig_out_Q_corr = np.mean(sig_out_Q_jax * sig_out_Q_py) / amplitude**2

    # recover the signals
    sig_rec_jax, _ = ds.recover(
        bin_in=sig_out_Q_jax,
    )

    sig_rec_py, _ = ds.recover(bin_in=sig_out_Q_py)

    # compute relative distance between original and recovered versions
    delay = abs(len(sig_in) - np.argmax(np.convolve(sig_in[::-1], sig_rec_jax)))
    sig_list = [sig_in[:-delay], sig_rec_jax[delay:], sig_rec_py[delay:]]
    norm_distance_mat = np.zeros((3, 3))

    for i, sig1 in enumerate(sig_list):
        for j, sig2 in enumerate(sig_list):
            norm_dist = norm(sig1 - sig2) / np.sqrt(norm(sig1) * norm(sig2))
            norm_distance_mat[i, j] = norm_dist

    print(
        "normalized distance between original signla and jax and python version:\n",
        norm_distance_mat,
    )

    plt.figure(figsize=(16, 10))

    plt.subplot(211)
    num_samples = 500
    plt.plot(time_vec, sig_in, label="input signal [short slice]")
    plt.plot(time_vec, sig_out_Q_jax, label="PDM modulation: jax")
    plt.plot(time_vec, sig_out_Q_py, label="PDM modulation: py")
    plt.xlim([duration - num_samples / fs, duration])
    plt.legend()
    plt.title(
        f"duration: jax 1st run:{duration_jax_1:0.3f}, jax 2nd run:{duration_jax_2:0.3f}, python:{duration_py:0.3f}\n"
        + f"correlation between jax and python PDM signal: {sig_out_Q_corr:0.3f}"
    )
    plt.grid(True)
    plt.xlabel("time (s)")

    plt.subplot(212)
    plt.plot(time_vec[:-delay], sig_in[:-delay], label="ínput signal")
    plt.plot(time_vec[:-delay], sig_rec_jax[delay:], label="jax + delay compensated")
    plt.plot(time_vec[:-delay], sig_rec_py[delay:], label="python + delay compensated")
    plt.text(0.75 * duration, 0, f"matrix of relative distances:\n{norm_distance_mat}")
    plt.grid(True)
    plt.xlabel("time (s)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test_jax()
