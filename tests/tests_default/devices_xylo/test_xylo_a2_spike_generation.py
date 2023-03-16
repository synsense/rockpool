import pytest

def test_imports():
    from rockpool.devices.xylo.syns61201 import AFESim
    from rockpool.devices.xylo.syns61201.afe_spike_generation import _encode_spikes, _encode_spikes_cpp, _encode_spikes_jax, _encode_spikes_python


def test_AFESim():
    import numpy as np
    from rockpool.devices.xylo.syns61201 import AFESim

    fs = 110_000

    afe_py = AFESim(fs=fs, add_noise=False, add_mismatch=False, add_offset=False)

    # afe_cpp = AFE_cpp(fs=fs, add_noise=False, add_mismatch=False, add_offset=False)

    # create a random noise
    amplitude = 60e-3
    freq = 1_000
    duration = 1

    time_vec = np.arange(0, duration, step=1 / fs)
    # sig_in = amplitude * np.sin(2*np.pi*freq*time_vec)
    sig_in = amplitude * np.random.randn(*time_vec.shape)

    spikes_py, *_ = afe_py(sig_in)
    # spikes_cpp, *_ = afe_cpp(sig_in)

    rate_py = np.mean(spikes_py.astype(np.float64), axis=0) * fs
    # rate_cpp = np.mean(spikes_cpp.astype(np.float64), axis=0) * fs

    print("rate_py: ", rate_py)
    # print("rate_cpp: ", rate_cpp)

    # numerical_error = norm(rate_py - rate_cpp) / np.min(
        # [np.max(rate_py), np.max(rate_cpp)]
    # )
    # print("relative error between rates:", numerical_error)

    # assert (
        # numerical_error < 0.01
    # ), "There is a large difference between C++ and Python version beyond numerical imprecision!"


# def main():
#     test()


# if __name__ == "__main__":
#     main()
