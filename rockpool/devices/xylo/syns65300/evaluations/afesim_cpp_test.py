# -----------------------------------------------------------
# This module checkes if the C++ version of spike generation is working well.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 12.01.2023
# -----------------------------------------------------------
import numpy as np
from numpy.linalg import norm
from rockpool.devices.xylo.syns65300.afe_sim_empirical import AFESimEmpirical as AFE_py
from rockpool.devices.xylo.syns65300.afe_sim_empirical_cpp import AFESimEmpirical as AFE_cpp

import matplotlib.pyplot as plt


def test():
    fs=110_000

    afe_py = AFE_py(
        fs=fs,
        add_noise=False,
        add_mismatch=False,
        add_offset=False
    )

    afe_cpp = AFE_cpp(
        fs=fs,
        add_noise=False,
        add_mismatch=False,
        add_offset=False
    )

    # create a random noise
    amplitude = 60e-3
    freq = 1_000
    duration = 1

    time_vec = np.arange(0, duration, step=1/fs)

    #### mixture of two sinusoid
    #sig_in = amplitude * np.sin(2*np.pi*freq*time_vec)
    #sig_in = amplitude * np.random.randn(*time_vec.shape)
    #sig_in1 = amplitude * np.sin(2*np.pi*freq*time_vec) + amplitude * np.random.randn(*time_vec.shape)
    #sig_in2 = amplitude * np.sin(2*np.pi*freq/2*time_vec) + amplitude * np.random.randn(*time_vec.shape)
    
    #sig_in = np.concatenate([sig_in1[:len(sig_in1)//2], sig_in2[:len(sig_in2)//2]])

    ##### chirp signal
    inst_freq = 200 + 600 * time_vec/time_vec[-1]
    phase = 2 * np.pi * np.cumsum(inst_freq) /fs
    sig_in = amplitude * np.sin(phase)  + amplitude * np.random.randn(*time_vec.shape)

    spikes_py, *_ = afe_py(sig_in)
    spikes_cpp, *_ = afe_cpp(sig_in)

    rate_cpp = np.mean(spikes_cpp.astype(np.float64), axis=0) * fs
    rate_py = np.mean(spikes_py.astype(np.float64), axis=0) * fs
    


    print("rate_py: ", rate_py)
    print("rate_cpp: ", rate_cpp)

    numerical_error = norm(rate_py - rate_cpp)/np.min([np.max(rate_py), np.max(rate_cpp)])
    print("relative error between rates:", numerical_error)

    bad_channels = np.argsort(np.abs(rate_py - rate_cpp))[-4:]

    plt.figure(figsize=(10,10))

    for i in range(4):
        plt.subplot(4,1,i+1)
        plt.plot(time_vec, spikes_py[:,bad_channels[i]].cumsum())
        plt.plot(time_vec, spikes_cpp[:,bad_channels[i]].cumsum())
        plt.grid(True)
        plt.legend(['Python', 'C++'])
        plt.ylabel("number of spikes")
        if i==0:
            plt.title("deviation between python and C++ in 4 worst channels")
    plt.xlabel("time (sec)")
    plt.show()
    
    assert numerical_error < 0.01, "There is a large difference between C++ and Python version beyond numerical imprecision!"

    


def main():
    test()

if __name__ == '__main__':
    main()
