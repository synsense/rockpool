# -----------------------------------------------------------
# This module provides some test cases for the divisive normalization module.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 23.01.2023
# -----------------------------------------------------------
import numpy as np
from numpy.linalg import norm

from xylo_a3_sim.divisive_normalization import DivisiveNormalization, jax_spike_gen, fjax_spike_gen, py_spike_gen
import matplotlib.pyplot as plt
import time

from pandas import DataFrame


def test_DN():
    # create a sinusoid signal
    fs = 48_000
    freq = 1_000
    num_periods = 1000
    duration = num_periods/freq
    time_vec = np.arange(0, duration, step=1/fs)
    
    sig_in = np.sin(2*np.pi*freq*time_vec)
    
    # quantize the signal
    Q = 14
    EPS = 0.00001
    sig_in = (2**(Q-1)*sig_in/(1+EPS)).astype(np.int64)

    num_channels = 4
    sig_in = np.asarray([i*sig_in for i in range(1,num_channels+1)], dtype=np.int64).T

    if sig_in.ndim == 1:
        sig_in = sig_in.reshape(-1,1)

    
    # apply DN
    dn = DivisiveNormalization(
        num_channels=num_channels,
        spike_rate_scale_bitshift1=12,
        spike_rate_scale_bitshift2=0,
        fs=fs,
    )
    
    start = time.time()
    spikes, recordings = dn.evolve(
        sig_in=sig_in,
        mode_in=1,
        joint_normalization=True,
        record=True,
    )
    duration_DN = time.time() - start
    spike_rate = spikes.mean(0)*fs

    print(f"DN run-time for a signal of shape:{sig_in.shape} => {duration_DN} sec")
    
    # plot the results
    plt.figure(figsize=(12,12))
    plt.subplot(511)
    plt.plot(time_vec, spikes)
    plt.ylabel("spikes")
    plt.grid(True)

    plt.subplot(512)
    plt.plot(time_vec, recordings['state_low_res_filter'])
    plt.ylabel("low-res filter")
    plt.grid(True)

    plt.subplot(513)
    plt.plot(time_vec, recordings['state_high_res_filter'])
    plt.ylabel("high-res filter")
    plt.grid(True)

    plt.subplot(514)
    plt.plot(time_vec, recordings['state_IAF'])
    plt.ylabel("IAF state")
    plt.grid(True)

    plt.subplot(515)
    plt.plot(time_vec, recordings['spike_gen_thresholds'])
    plt.ylabel("spike-gen threshold")
    plt.grid(True)

    plt.xlabel(
        f"""
            time (sec)
            run_time: {duration_DN} sec
            spike-rates: {spike_rate}
            target spike-rate: {dn.p * fs:0.2f}
        """
    )
    plt.show()



def test_jax_vs_python_DN():
    #===========================================================================
    #                        create the input signal
    #===========================================================================
    fs = 48_000
    freq = 1_000
    num_periods = 10000
    duration = num_periods/freq
    time_vec = np.arange(0, duration, step=1/fs)

    # quantize the signal
    num_channels = 16
    Q = 22
    EPS = 0.00001

    # sinusoid signal
    sig_in = np.sin(2*np.pi*freq*time_vec)
    sig_in = (2**(Q-1)*sig_in/(np.max(np.abs(sig_in)) * (1+EPS))).astype(np.int64)
    sig_in = np.asarray([i*sig_in for i in range(1,num_channels+1)], dtype=np.int64).T
    
    # sompletely random signal
    sig_in = np.random.randn(len(time_vec), num_channels)
    random_gain = np.random.randn(num_channels)
    sig_in = np.einsum('ij,j->ij', sig_in, random_gain)

    sig_in = (2**(Q-1)*sig_in/(np.max(np.abs(sig_in)) * (1+EPS))).astype(np.int64)
    
   
    if sig_in.ndim == 1:
        sig_in = sig_in.reshape(-1,1)
    
    #===========================================================================
    #                            Settings for the simulation
    #===========================================================================
    mode_vec = np.ones(num_channels, dtype=np.int64)
    spike_rate_scale_bitshift1 = 8 * np.ones(num_channels)
    spike_rate_scale_bitshift2 = 0 * np.ones(num_channels)
    low_pass_bitshift = 12 * np.ones(num_channels, dtype=np.int64)
    EPS_vec = (10000 * np.random.rand(num_channels)).astype(np.int64) + 10
    fixed_threshold_vec = 2**22 * np.ones(num_channels, dtype=np.int64)


    #===========================================================================
    #                            jax int32 version
    #===========================================================================
    # first : jit compilation
    start = time.time()
    spikes_jax, _ = jax_spike_gen(
        sig_in=sig_in,
        mode_vec=mode_vec,
        spike_rate_scale_bitshift1=spike_rate_scale_bitshift1,
        spike_rate_scale_bitshift2=spike_rate_scale_bitshift2,
        low_pass_bitshift=low_pass_bitshift,
        EPS_vec=EPS_vec,
        fixed_threshold_vec=fixed_threshold_vec,
        record=True,
    )
    duration_jax_compile = time.time() - start

    # second: run it again to see the real speed
    start = time.time()
    spikes_jax, recording_jax = jax_spike_gen(
        sig_in=sig_in,
        mode_vec=mode_vec,
        spike_rate_scale_bitshift1=spike_rate_scale_bitshift1,
        spike_rate_scale_bitshift2=spike_rate_scale_bitshift2,
        low_pass_bitshift=low_pass_bitshift,
        EPS_vec=EPS_vec,
        fixed_threshold_vec=fixed_threshold_vec,
        record=True
    )
    duration_jax = time.time() - start

    spike_rate_jax = spikes_jax.mean(0) * fs

    print("\n", " jax DN benchmarking ".center(100, "+"))
    print("spike rate (jax):\n", spike_rate_jax)
    print(f"compilation time of jax for an input of shape {sig_in.shape}: {duration_jax_compile} sec")
    print(f"run time of jax for an input of shape {sig_in.shape}: {duration_jax} sec")

    #===========================================================================
    #                            jax float32 version
    #===========================================================================
    
    # first : jit compilation
    start = time.time()
    spikes_fjax, _ = fjax_spike_gen(
        sig_in=sig_in,
        mode_vec=mode_vec,
        spike_rate_scale_bitshift1=spike_rate_scale_bitshift1,
        spike_rate_scale_bitshift2=spike_rate_scale_bitshift2,
        low_pass_bitshift=low_pass_bitshift,
        EPS_vec=EPS_vec,
        fixed_threshold_vec=fixed_threshold_vec,
        record=True,
    )
    duration_fjax_compile = time.time() - start

    # second: run it again to see the real speed
    start = time.time()
    spikes_fjax, recording_fjax = fjax_spike_gen(
        sig_in=sig_in,
        mode_vec=mode_vec,
        spike_rate_scale_bitshift1=spike_rate_scale_bitshift1,
        spike_rate_scale_bitshift2=spike_rate_scale_bitshift2,
        low_pass_bitshift=low_pass_bitshift,
        EPS_vec=EPS_vec,
        fixed_threshold_vec=fixed_threshold_vec,
        record=True
    )
    duration_fjax = time.time() - start

    spike_rate_fjax = spikes_fjax.mean(0) * fs

    print("\n", " fjax DN benchmarking ".center(100, "+"))
    print("spike rate (fjax):\n", spike_rate_fjax)
    print(f"compilation time of fjax for an input of shape {sig_in.shape}: {duration_fjax_compile} sec")
    print(f"run time of fjax for an input of shape {sig_in.shape}: {duration_fjax} sec")
    

    #===========================================================================
    #                            python version
    #===========================================================================

    start = time.time()
    
    spikes_py, recording_py = py_spike_gen(
        sig_in=sig_in,
        mode_vec=mode_vec,
        spike_rate_scale_bitshift1=spike_rate_scale_bitshift1,
        spike_rate_scale_bitshift2=spike_rate_scale_bitshift2,
        low_pass_bitshift=low_pass_bitshift,
        EPS_vec=EPS_vec,
        fixed_threshold_vec=fixed_threshold_vec,
        record=True
    )
    duration_dn = time.time() - start


    spike_rate_py = spikes_py.mean(0) * fs

    print("\n", " python DN benchmarking ".center(100, "+"))
    print("spike rate (python):\n", spike_rate_py)
    print(f"run time of python for an input of shape {sig_in.shape}: {duration_dn} sec\n")


    #===========================================================================
    #                evaluating the error beween various version
    #===========================================================================
    ## rates
    rate_list = [spike_rate_py, spike_rate_jax, spike_rate_fjax]
    
    rel_error_rate = np.zeros( (len(rate_list), len(rate_list)) )
    
    for i,first in enumerate(rate_list):
        for j,second in enumerate(rate_list):
            rel_error_rate[i,j] = norm(first - second)/np.sqrt(norm(first) * norm(second))
    
    print("\n\n", " comparison between python, jax, fjax version ".center(100, "+"))
    
    print("rate relative error matrix between python, jax, fjax version")
    print(DataFrame(rel_error_rate), "\n")
    
    if (np.max(rel_error_rate)<0.001):
        print("tests for spike rate: passed succsessfully!\n\n")
    
    
    ## spikes
    spike_list = [spikes_py, spikes_jax, spikes_fjax]
    
    rel_error_spike = np.zeros( (len(spike_list), len(spike_list)) )
    
    for i,first in enumerate(spike_list):
        for j,second in enumerate(spike_list):
            rel_error_spike[i,j] = norm(first[:] - second[:])/np.sqrt(norm(first[:]) * norm(second[:]))
            
    
    print("spike relative error matrix between python, jax, fjax version:")
    print(DataFrame(rel_error_spike), "\n")
    
    if (np.max(rel_error_spike)<0.001):
        print("tests for spike: passed succsessfully!\n\n")
    
    
    ## states
    
    # check the state dictionaries
    for k in recording_jax.keys():
        v_py = recording_py[k][:]
        v_jax = recording_jax[k][:]
        v_fjax = recording_fjax[k][:]
        
        k_list = [v_py, v_jax, v_fjax]
        
        rel_error_k = np.zeros( (len(k_list), len(k_list)) )
        
        for i,first in enumerate(k_list):
            for j,second in enumerate(k_list):
                rel_error_k[i,j] = norm(first[:] - second[:])/np.sqrt(norm(first[:]) * norm(second[:]))
        
        print(f"{k} - relative error matrix between python, jax, fjax version:")
        print(DataFrame(rel_error_k),"\n")
        
        if (np.max(rel_error_k)<0.001):
            print(f"tests for - {k} - : passed succsessfully!\n\n")
        else:
            print(f"tests for - {k} - : not passed!\n\n")
    
        

def main():
    #test_DN()
    test_jax_vs_python_DN()


if __name__ == '__main__':
    main()
