import pytest

pytest.skip("REFACTOR THIS TEST", allow_module_level=True)

# this function tests the spike generation and divisive normalization module
from ctypes import c_int
from mimetypes import init
import numpy as np
from numpy.linalg import norm
from xylo_a2_spike_generation import lif_spike_gen

from rockpool.timeseries import TSEvent
from rockpool.devices.xylo.syns65300.afe_sim_empirical import _encode_spikes
import matplotlib.pyplot as plt
import time


############################################################
# metrics for measuring the distortion in various scenrios
############################################################
def rel_error_2(v1, v2):
    w1 = v1.ravel()
    w2 = v2.ravel()

    return norm(w1 - w2) / np.sqrt(norm(w1) * norm(w2))


def rel_error_1(v1, v2):
    w1 = v1.ravel()
    w2 = v2.ravel()

    return norm(w1 - w2, 1) / np.sqrt(norm(w1, 1) * norm(w2, 1))


def rel_error_levy_global(v1, v2):
    w1 = v1.ravel()
    w2 = v2.ravel()

    return np.max(np.abs(w1.cumsum() - w2.cumsum())) / np.min(
        [norm(w1, 1), norm(w2, 1)]
    )


def rel_error_levy_local(v1, v2):
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
    """
    this function checks how the spike generation module in C++ comapres with that implemented in Python.
    It seems that due to floating point numerical issues, sometimes there is a slight distortion in the spikes.
    In this part, we try to quantify that distortion.
    """

    # parameters of AFESim
    vcc = 1.1

    num_channels = 16
    raster_dt = 1 / 100
    max_num_spikes = 15
    dt = 1 / 50_000

    initial_state = np.zeros(num_channels)
    v2i_gain = 3.333e-7
    c_iaf = 5.0e-12
    leakage = 1.0e-9
    threshold = 0.5

    # signal specification
    sig_len = 220_000
    amplitude = 10.0e-3  # 10 mV

    sig_rect = amplitude * np.random.rand(num_channels, sig_len).T

    # def _encode_spikes(
    #     inital_state: np.ndarray,
    #     dt: float,
    #     data: np.ndarray,
    #     v2i_gain: float,
    #     c_iaf: float,
    #     leakage: float,
    #     thr_up: float,
    #     max_output: float,
    # ) -> Tuple[np.ndarray, np.ndarray]:

    ##################################
    # first run: dummy to compile jit in case jit is used
    ##################################
    start = time.time()
    spikes_py, final_state_py = _encode_spikes(
        initial_state=initial_state,
        dt=dt,
        data=sig_rect,
        v2i_gain=v2i_gain,
        c_iaf=c_iaf,
        leakage=leakage,
        thr_up=threshold,
        max_output=vcc,
    )

    duration_py_first = time.time() - start
    ##################################

    # run it again
    start = time.time()
    spikes_py, final_state_py = _encode_spikes(
        initial_state=initial_state,
        dt=dt,
        data=sig_rect,
        v2i_gain=v2i_gain,
        c_iaf=c_iaf,
        leakage=leakage,
        thr_up=threshold,
        max_output=vcc,
    )

    # raster the spikes
    spikes_raster_py = TSEvent.from_raster(spikes_py, dt=dt).raster(
        dt=raster_dt, add_events=True
    )
    spikes_raster_py[spikes_raster_py > max_num_spikes] = max_num_spikes

    duration_py = time.time() - start

    # spike generation with c++ modules
    # py::arg("initial_state"),
    # py::arg("data"),
    # py::arg("v2i_gain")=3.333e-7,
    # py::arg("c_iaf")=5.0e-12,
    # py::arg("leakage")=1.0e-9,
    # py::arg("threshold")=0.5,
    # py::arg("vcc")=1.1,
    # py::arg("dt")=1.0/48000,
    # py::arg("raster_dt")=0.01,
    # py::arg("max_num_spikes")=15

    start = time.time()
    spikes_raster_cpp, final_state_cpp = lif_spike_gen(
        initial_state=initial_state,
        data=sig_rect.T,
        v2i_gain=v2i_gain,
        c_iaf=c_iaf,
        leakage=leakage,
        threshold=threshold,
        vcc=vcc,
        dt=dt,
        raster_dt=raster_dt,
        max_num_spikes=max_num_spikes,
    )

    spikes_raster_cpp = np.asarray(spikes_raster_cpp).T
    final_state_cpp = np.asarray(final_state_cpp)

    duration_cpp = time.time() - start

    # compute the distortion metric

    rel_distance_final_state = rel_error_2(final_state_cpp, final_state_py)

    # relative errors of channels
    rel_distance_spike_channel = []
    time_vec = np.arange(spikes_raster_py.shape[0]) * raster_dt

    for channel in range(num_channels):
        if channel % 4 == 0:
            plt.figure(figsize=(10, 16))
            plt.ylabel("difference of spikes")

        plt.subplot(4, 1, (channel % 4 + 1))
        rel_distance = rel_error_levy_local(
            spikes_raster_cpp[:, channel], spikes_raster_py[:, channel]
        )
        rel_distance_spike_channel.append(rel_distance)

        # plt.plot(time_vec, spikes_raster_cpp[:,channel], '+')
        # plt.plot(time_vec, spikes_raster_py[:,channel], '.')
        plt.plot(
            time_vec, spikes_raster_cpp[:, channel] - spikes_raster_py[:, channel], "."
        )
        plt.title(f"relative error between spikes: {rel_distance:0.8f}")
        plt.grid(True)
        if (channel % 4 + 1) == 4:
            plt.xlabel("time (sec)")

    # summary of the investigation
    print("\n")
    print(" summary of the results ".center(100, "="))
    print(f"relative distance of final states: {rel_distance_final_state}")
    print(f"distortion in spikes of various channels:\n{rel_distance_spike_channel}")
    print(
        f"maximum distortion in spikes of various channels: {np.max(rel_distance_spike_channel)}"
    )
    print(
        f"duration of a single run: C++ => {duration_cpp}, Python [1st run]=> {duration_py_first}, Python [2nd run] => {duration_py}"
    )
    print("+=" * 50)

    # plot the results visually as well
    plt.show()


def main():
    test_spike_generation()


if __name__ == "__main__":
    main()
