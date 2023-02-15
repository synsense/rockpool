# -----------------------------------------------------------
# This module provides some test cases for the filterbank module in Xylo-A3.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 19.01.2023
# -----------------------------------------------------------

from random import sample
import numpy as np
from xylo_a3_sim.digital_filterbank import ChipButterworth
from xylo_a3_sim.pdm_adc import PDM_ADC
from rockpool.timeseries import TSContinuous
import matplotlib.pyplot as plt

import time
from numpy.linalg import norm
from tqdm import tqdm



def test_filterbank_chirp():
    """ this is a simple test of filters for a chirp signal. """

    # just to extract the default audio sampling rate
    pdm_adc = PDM_ADC()
    
    # filterbank
    fb = ChipButterworth()
    
    # produce a chirp signal
    fmin = 100
    fmax = 20_000
    sample_rate = pdm_adc.pdm_microphone.fs/pdm_adc.pdm_microphone.sdm_OSR
    duration = 10
    
    num_samples = int(duration * sample_rate)
    dt = duration/num_samples
    
    
    
    f_inst = np.linspace(fmin, fmax, num_samples)
    phi_inst = 2 * np.pi * np.cumsum(f_inst) * dt
    
    sig_in = np.sin(phi_inst)

    # simple sinusoid signal
    # freq=1_000
    # sig_in = np.sin(2*np.pi*freq * np.arange(0,duration, step=dt))
    
    # quantize the signal
    B_in = fb.bd_list[0].B_in 
    EPS = 0.00001
    sig_in = (sig_in/(np.max(np.abs(sig_in)) * (1+EPS)) * 2**(B_in-1) ).astype(np.int64)
    

    #===========================================================================
    #                            Python Multi-processor version
    #===========================================================================
    start = time.time()

    num_workers = 1
    sig_out_py, recording_py = fb.evolve(
        sig_in=sig_in,
        num_workers=num_workers,
        scale_out=False,
        python_version=True,  # force evaluation by python
        record=True,
    )

    duration_py = time.time() - start
    
    
    print("\n", f" python version of filters: num-workers: {num_workers} ".center(100, "+"), "\n")
    print("run time: ", duration_py, "\n")


    #===========================================================================
    #                            Jax version
    #===========================================================================

    start = time.time()

    sig_out_jax, recording_jax = fb.evolve(
        sig_in=sig_in,
        scale_out=False,
        record=True,
    )

    duration_jax_compile = time.time() - start


    start = time.time()

    sig_out_jax, recording_jax = fb.evolve(
        sig_in=sig_in,
        scale_out=False,
        record=True,
    )

    duration_jax = time.time() - start

    # NOTE: We add this scaling here since python version still has this issue of Bitshift at the output of MA filter.
    sig_out_jax = (sig_out_jax >> fb.bd_list[0].B_wf)


    print("\n", " jax version of filters ".center(100, "+"), "\n")
    print("compile time: ", duration_jax_compile, "\n")
    print("run time: ", duration_jax, "\n")


    #===========================================================================
    #                            comparing the results
    #===========================================================================
    rel_error = np.asarray([norm(a[:]-b[:])/np.sqrt(norm(a[:])*norm(b[:])) for (a,b) in zip(sig_out_py.T, sig_out_jax.T)])
    
    print("\n", " comparing jax with python version ".center(100, "+"), "\n")
    print("relative error of output of filters (jax float32 <==> python exact int64):")
    print(rel_error)
    
    

    #===========================================================================
    #              sanity check for center frequency of filters
    #===========================================================================

    sig_out_list = [sig_out_py, sig_out_jax]
    name_list = ["python version", "jax version"]
    
    for sig_out, name in zip(sig_out_list, name_list):
        # normalize the signal for better viuslaization
        sig_out = np.abs(sig_out/np.max(np.abs(sig_out)))

        # find the time at which each channel reaches its maximum amplitude
        # apply LSQ to find the scaling parameter
        T_max = np.argmax(sig_out, axis=0)*dt
        filter_idx = np.arange(fb.numF)

        A = np.asarray([[1]*len(T_max), np.cumsum([1]*len(T_max))]).T
        y = np.log(T_max)
        params = np.linalg.lstsq(A, y, rcond=None)[0]

        log_alpha = params[1]
        exp_filter_scaling_param = np.exp(log_alpha)
        print(f"relative scaling of designed filters => empirical via LSQ: {exp_filter_scaling_param}, design: {(20_000/100)**(1/15)}")



        plt.figure(figsize=(12,12))
        TSContinuous.from_clocked(sig_out, dt=1/sample_rate).plot(stagger=1)
        plt.grid(True)
        plt.xlabel(
            "time (sec)\n\n"+\
            f"relative scaling of designed filters => empirical via LSQ: {exp_filter_scaling_param}, design: {(20_000/100)**(1/15)}"
        )
        plt.ylabel("relative instantenous power in various filters")
        plt.title(f"{name}: Xylo-A3 digital filterbank output for chirp => fmin:{fmin} Hz, fmax:{fmax} Hz, sweep-time:{duration} sec")

        plt.plot(T_max, filter_idx, 'b*')
    
    plt.show()


def test_filterbank_worstcase():
    """ this test provides the worst-case simulation of the filters. """

    # just to extract the default audio sampling rate
    pdm_adc = PDM_ADC()
    
    # filterbank
    fb = ChipButterworth()
    

    # produce an impules signal
    sample_rate = pdm_adc.pdm_microphone.fs/pdm_adc.pdm_microphone.sdm_OSR
    duration = 1.0
    
    num_samples = int(duration * sample_rate)
    dt = duration/num_samples

    impulse = np.zeros(num_samples)
    impulse[0] = 1.0
    
    # quantize the signal
    B_in = fb.bd_list[0].B_in
    EPS = 0.00001
    impulse = (impulse/(np.max(np.abs(impulse)) * (1+EPS)) * 2**(B_in-1) ).astype(np.int64)

    # collect the output of the filters for an impulse signal
    num_workers = 4
    impulse_res = fb.evolve(sig_in=impulse, num_workers=num_workers)

    # now use the impulse responses to find the worst case signal
    worst_case_amplitude = []

    for filt_idx, imp_res_filt in enumerate(impulse_res.T):
        # compute the worst case signal: roll the signal to put the main part in the middle
        worst_case_input = np.sign(imp_res_filt[::-1])
        worst_case_input = np.roll(worst_case_input, len(worst_case_input)//2)

        # quantize this signal
        worst_case_input = (worst_case_input/(np.max(np.abs(worst_case_input)) * (1+EPS)) * 2**(B_in-1)).astype(np.int64)

        # use this worst case signal to trigger the filter again
        worst_case_output = fb._filter_AR(sig_in=worst_case_input, bd=fb.bd_list[filt_idx])

        # compute the maximum amplitude
        max_amplitude = np.max(np.abs(worst_case_output))

        worst_case_amplitude.append(max_amplitude)

    # number of bits needed for worst case overflow avoidance: 
    # (i)   get rid of the shift due to B_wf 
    # (ii)  note that input amplitude is at most 2**(B_in - 1) - 1
    additional_num_bits_worst_case_emp = np.ceil(np.log2(worst_case_amplitude)) - (B_in-1) - np.asarray([bd.B_wf for bd in fb.bd_list])

    # worst case number of bits in designed
    additional_num_bits_worst_case_design = [bd.B_worst_case for bd in fb.bd_list]

    plt.figure(figsize=(12,12))
    plt.plot(additional_num_bits_worst_case_emp)
    plt.plot(additional_num_bits_worst_case_design, '*')
    plt.grid(True)
    plt.xlabel("filter ID")
    plt.title("additional bits in B_wi (AR filter) compared to B_in to avoid overflow")
    plt.title("additional number of bits in filter")
    plt.legend(["empirical", "design"])
    plt.show()


def test_filterbank_jax_worst_case():
    """ this function does the worst-case analysis for the filterbank using the jax version."""
    # just to extract the default audio sampling rate
    pdm_adc = PDM_ADC()
    
    # filterbank
    fb = ChipButterworth()
    
    # sample rate
    sample_rate = pdm_adc.pdm_microphone.fs/pdm_adc.pdm_microphone.sdm_OSR

    # number of quantization bits
    Q = fb.bd_list[0].B_in + 4
    
    # compute the impulse response of AR part of the filters
    duration = 1
    num_samples = int(duration * sample_rate)
    impulse = np.zeros(num_samples)
    impulse[0] = 1
    
    # quantize the input signal
    EPS = 0.00001
    impulse = (impulse * (2**(Q-1))/(1+EPS)).astype(np.int64)
    
    # compute the output of the filterbank
    sig_out, recording = fb.evolve(
        sig_in=impulse,
        scale_out=False,
        record=True,
        python_version=False,
    )
    
    # compute the output of AR filters from the computed states
    ar_out = recording["filter_AR_state"][:,:,0]
    
    worst_case_ar_amplitude_amplification = []
    
    for filt_idx, ar_response in enumerate(ar_out.T):
        # requantize the output
        worst_case_in = ( ar_response/np.max(np.abs(ar_response))/(1+EPS) * 2**(Q-1) ).astype(np.int64)
        worst_case_in = worst_case_in[::-1]
        
        # compute the output of the filters
        sig_out, recording = fb.evolve(
            sig_in=worst_case_in,
            scale_out=False,
            record=True,
            python_version=False,
        )
        
        # compute the corresponding AR output
        worst_case_out = recording["filter_AR_state"][:,filt_idx,0]
        
        # compute the amplification factor
        worst_case_ar_amplitude_amplification.append(np.max(np.abs(worst_case_out))/np.max(np.abs(worst_case_in)))
        
    # how many extra bits are needed
    worst_case_extra_bits = np.ceil(np.log2(worst_case_ar_amplitude_amplification)).astype(np.int64)

    # remove the effect of B_wf bits added to avoid dead zone in the filter (they are already taken into account)
    worst_case_extra_bits -= np.asarray([bd.B_wf for bd in fb.bd_list])
    
    # what is the designed extra bits
    designed_worst_case_extra_bits = np.asarray([bd.B_worst_case for bd in fb.bd_list])
    
    
    plt.figure(figsize=(12,12))
    plt.plot(designed_worst_case_extra_bits)
    plt.plot(worst_case_extra_bits, ".")
    plt.legend(["design", "empirical"])
    plt.grid(True)
    plt.xlabel("filter idx")
    plt.ylabel("number of extra bits needed")

    plt.title("number of extra bits needed for AR part in worst-case")
    
    plt.show()    
    
    
        
    

def test_freq_response():
    """this functions validates the frequency responses of the filters via numerical simulations."""
    # just to extract the default audio sampling rate
    pdm_adc = PDM_ADC()
    
    # filterbank
    fb = ChipButterworth()
    
    # produce a chirp signal
    fmin = 100
    fmax = 20_000
    sample_rate = pdm_adc.pdm_microphone.fs/pdm_adc.pdm_microphone.sdm_OSR
    duration = 1
    time_vec = np.arange(0,duration, step=1/sample_rate)

    # list of frequencies to probe
    num_freq = 1000
    freq_vec = 10**np.linspace(np.log10(fmin), np.log10(fmax), num_freq)

    freq_response = []
    for freq in tqdm(freq_vec):
        # produce a sinusoid signal
        sig_in = np.sin(2*np.pi*freq*time_vec)

        # quantize the sinal
        B_in = fb.bd_list[0].B_in + 4
        EPS = 0.00001
        sig_in = (sig_in/(np.max(np.abs(sig_in)) * (1+EPS)) * 2**(B_in-1) ).astype(np.int64)

        # compute the frequency response
        sig_out, _ = fb.evolve(
            sig_in=sig_in,
            scale_out=True,
            record=False,
        )

        freq_res = np.mean(sig_out**2, axis=0)/np.mean(sig_in**2)

        freq_response.append(freq_res)

    freq_response = np.asarray(freq_response)
    freq_response = freq_response/np.max(freq_response)

    # convert into dB
    freq_response_dB = 10*np.log10(freq_response)

    # plot the results
    plt.figure(figsize=(16,10))
    plt.semilogx(freq_vec, freq_response_dB)
    plt.xlabel("freq (Hz)")
    plt.ylabel("freq response [dB]")
    plt.grid(True)
    plt.title("empirical frequency response of Xylo-A3 filterbank")
    plt.ylim([-10,1])

    plt.show()


    

def main():
    test_filterbank_chirp()
    #test_filterbank_worstcase()
    #test_filterbank_jax_worst_case()
    #test_freq_response()
    


if __name__ == '__main__':
    main()
    
    
    
    
    