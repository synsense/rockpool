# -----------------------------------------------------------------------------------------------------------------------
# This module contains the final design for anti-aliasing filter with oversampled ADC.
# The result is an equivalent-ADC that has much better anti-aliasing performance that the ordinary non-oversampled ADC.
#
# NOTE:
#       (i)  we tried several filters with emphasis of FIR filters but their results especially attenuation around fs/2 was
#            not as good as we expected. One problem was that we had to work with filters with only few taps whose selecivity
#            was not that good.
#       (ii) in our final design, we had to use IIR filters at the end. Among IIR filters, we had to use Elliptic filters to 
#            obtain the sharpest transition from passband to stopband. Other filters such as Chebychev and Butterworth were not 
#            that good for this purpose. 
#            In fact for filters in the filterbank we use Butterworth filters because we intentionally wanted shallow filters
#            to avoid notch between adjacent filters. But here the problem was different and we had to use Elliptic filters.
#            
#            Other reasons we decided to use IIR filters:
#               (a)  they have much better selectivity.
#               (b)  they are much easier to implement with only a few multipliers bit FIR filters typically need more multipliers.
#               (c)  their phase delay is in general badly-distorted but in our case we needed just a not-so-selective filter
#                    and its phase response was almost linear and its transient was super fast.
#               (b)  we just needed to make sure that the quantization is not an issue in Elliptic filters. The main reason was that 
#                    the Butterworth filters have a very structured `b` which a clean zero-pattern but in Elliptic filters the zero-pttern
#                    is quite complicated. So we had to use enough bits to quantize `b` coefficients of the filter.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 19.04.2023
# ------------------------------------------------------------------------------------------------------------------------

#* reguired packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp 


from alias.xylo_a3_agc_specs import AUDIO_SAMPLE_RATE, LOW_PASS_CORNER
from alias.low_pass_design_iir import DecimationFilter_IIR, FilterOverflowError
from alias.oversampled_ADC import OversampledADC_IIR
from alias.aliasing_evaluation import AliasingEval




#===========================================================================
#*                           Designed filter
#===========================================================================
audio_fs = AUDIO_SAMPLE_RATE
target_audio_passband = LOW_PASS_CORNER     # 20_000 Hz
adc_oversampling = 4
adc_num_bits = 10
filter_order = 4
filter_type = 'ellip'
padd_band_ripple_dB = 0.5
stop_band_attenuation_dB = 30
num_bits_filter_Q = 18


decimation_filter = DecimationFilter_IIR(
    audio_passband=target_audio_passband,
    audio_fs=audio_fs,
    adc_oversampling_factor=adc_oversampling,
    filter_order=filter_order,
    filter_type=filter_type,
    pass_band_ripple_dB=padd_band_ripple_dB,
    stop_band_attenuation_dB=stop_band_attenuation_dB,
    num_bits_in=adc_num_bits,
    num_bits_filter_Q=num_bits_filter_Q,
)

print(f" filter used for the simulations: ADC oversampling factor {adc_oversampling} ".center(100, "+"))
print(decimation_filter)
#===========================================================================
#*                       Analog simulation module
#===========================================================================
simulation_oversampling = 20

adc = OversampledADC_IIR(
    audio_fs=audio_fs,
    target_audio_passband=target_audio_passband,
    adc_oversampling=adc_oversampling,
    adc_num_bits=adc_num_bits,
    decimation_filter=decimation_filter,
    simulation_oversampling=simulation_oversampling
)


#===========================================================================
#*                    Performance evaluation module
#===========================================================================
eval_module = AliasingEval(
    module=adc,
)




#===========================================================================
#*                     Various verifications
#===========================================================================
#! frequency response of the filter with direct evaluation
def freq_response():
    # test frequencies
    num_freq = 400
    min_freq = 10000
    max_freq = decimation_filter.audio_fs
    freq_vec = np.exp(np.linspace(np.log(min_freq), np.log(max_freq), num_freq))
    
    num_periods = 100
    
    freq_response, rel_distortion = decimation_filter.freq_response(
        freq_vec=freq_vec,
        num_periods=num_periods,
    )
    
    freq_response_dB = 20*np.log10(freq_response)
    rel_distortion_dB = 20*np.log10(rel_distortion)
    
    # plot
    plt.figure(figsize=(10,12))
    
    plt.subplot(211)
    plt.title(f"empirical freq response and distortion of the filter for ADC oversampling: {adc_oversampling}")
    plt.semilogx(freq_vec, freq_response_dB)
    plt.grid(True, which="both")
    plt.ylim([-40, 2])
    plt.axvline(x=target_audio_passband, color='g', linewidth=2, label=f"3dB ~ {target_audio_passband/1000:0.1f} KHz")
    plt.axvline(x=audio_fs/2, color='r', linewidth=2, label=f"audio fs/2 ~ {audio_fs/2000:0.1f} KHz")
    plt.legend()
    plt.ylabel("freq response [dB]")
    
    plt.subplot(212)
    plt.semilogx(freq_vec, rel_distortion_dB)
    plt.ylim([-80, 10])
    plt.grid(True)
    plt.ylabel("relative distortion [dB]")
    
    plt.xlabel("freq (Hz)")
    plt.show()
    

#! performance of the filter with random sinusoid signals
def sinusoid_performance():
    # test frequencies
    num_freq = 400
    min_freq = 10000
    max_freq = decimation_filter.audio_passband + 2_000
    freq_vec = np.exp(np.linspace(np.log(min_freq), np.log(max_freq), num_freq))
    
    
    rel_distortion_vec = []
    max_range_covered_vec = []
    max_stable_range_covered_vec = []
    
    num_periods = 100
    
    
    try:
        for freq in freq_vec:
            duration = num_periods/freq
            time_vec = np.arange(0, duration, step=1/decimation_filter.fs)

            # produce a random audio signal
            phase = np.random.rand(1)[0] * 2*np.pi
            
            sig_in = np.sin(2*np.pi*freq*time_vec + phase)
            
            EPS = 0.0000001
            sig_in_Q = (2**(decimation_filter.num_bits_in-1) * sig_in / (np.max(np.abs(sig_in)) * (1+EPS))).astype(np.int64)
            sig_out_Q, rel_distortion = decimation_filter.evolve_full(sig_in=sig_in_Q)
            
            sig_out_Q_stable = sig_out_Q[len(sig_out_Q)//6:]
            
            max_range_covered = np.max(np.abs(sig_out_Q))/(2**(decimation_filter.num_bits_in-1))
            max_stable_range_covered = np.max(np.abs(sig_out_Q_stable))/(2**(decimation_filter.num_bits_in-1))
            
            rel_distortion_vec.append(rel_distortion)
            max_stable_range_covered_vec.append(max_stable_range_covered)
            max_range_covered_vec.append(max_range_covered)
            
            print(f"filter worst-case verification with sinusoid input => freq: {freq:0.1f} Hz,\trelative distortion: {rel_distortion:0.8f},\tmax range covered: {max_range_covered:0.5f},\tmax stable range covered: {max_stable_range_covered:0.5f}")
        
        rel_distortion_vec = np.asarray(rel_distortion_vec)
        max_range_covered_vec = np.asarray(max_range_covered_vec)
        max_stable_range_covered_vec = np.asarray(max_stable_range_covered_vec)
        
        print("\n\n")
        print(" worst-case sinusoid verification results ".center(100, "+"))
        print("worst-case verification of the filter with sinusoid input was passed: no overflow or underflow!")
        print(f"frequency of nonzero relative distortion: {100 * np.sum(rel_distortion_vec>0.0)/len(rel_distortion_vec):0.2f} % of the test cases!")
        print(f"relative distortion in nonzero cases: mean: {np.mean(rel_distortion_vec[rel_distortion_vec>0.0])}, min: {np.min(rel_distortion_vec[rel_distortion_vec>0.0])}, max: {np.max(rel_distortion_vec[rel_distortion_vec>0.0])}")
        print(f"the average dynamic range covered: {np.mean(max_range_covered_vec):0.2f}")
        print(f"the average stable dynamic range covered: {np.mean(max_stable_range_covered_vec):0.2f}")
        
    except FilterOverflowError as e: 
        print(f"overflow or underflow error: worst-case verification of the filter with random input was not passed!\n\n{e}\n")


#! performance of the filter with random signals
def random_performance():
    num_iter = 10
    rel_distortion_vec = []
    
    num_periods = 100
    min_freq = 100
    duration = num_periods/min_freq
    num_samples = int(decimation_filter.fs * duration)
    
    
    try:
        for epoch in range(num_iter):
            # produce a random audio signal
            sig_in = np.random.randn(num_samples)
            EPS = 0.0000001
            sig_in_Q = (2**(decimation_filter.num_bits_in-1) * sig_in / (np.max(np.abs(sig_in)) * (1+EPS))).astype(np.int64)
            sig_out_Q, rel_distortion = decimation_filter.evolve_full(sig_in=sig_in_Q)
            
            sig_in_Q = (2**(decimation_filter.num_bits_in-1) * sig_out_Q / (np.max(np.abs(sig_out_Q)) * (1+EPS))).astype(np.int64)
            sig_out_Q, rel_distortion = decimation_filter.evolve_full(sig_in=sig_in_Q)
            
            rel_distortion_vec.append(rel_distortion)
            
            print(f"filter worst-case verification with random input => epoch {epoch}/{num_iter}: relative distortion: {rel_distortion}")
        
        rel_distortion_vec = np.asarray(rel_distortion_vec)
        
        print("\n\n")
        print(" random input verification results ".center(100, "+"))
        print("worst-case verification of the filter with random input was passed: no overflow or underflow!")
        print(f"frequency of nonzero relative distortion: {100 * np.sum(rel_distortion_vec>0.0)/len(rel_distortion_vec):0.2f} % of the test cases!")
        print("average relative distortion in nonzero cases: ", np.mean(rel_distortion_vec[rel_distortion_vec>0.0]))
        print()
        
    except FilterOverflowError as e: 
        print(f"overflow or underflow error: worst-case verification of the filter with random input was not passed!\n\n{e}\n")
        

#! worst-case performance of the filter
def worst_case_performance():
    """ this modules assures that there is no overflow in the filter """
    
    #* AR part of the filter
    num_periods = 1000
    duration = num_periods/decimation_filter.audio_fs
    num_samples = int(duration * decimation_filter.fs)
    
    impulse = np.zeros(num_samples)
    impulse[0] = 1
    
    impulse_Q = ((2**(decimation_filter.num_bits_in -1)-1)*impulse).astype(np.int64)
    
    try:
        response_AR = decimation_filter.evolve_AR(sig_in=impulse_Q)
        
        input_AR_worst_case = np.concatenate([np.sign(response_AR[::-1]), np.zeros_like(response_AR)])
        input_AR_worst_case_Q = ( (2**(decimation_filter.num_bits_in -1)-1) * input_AR_worst_case).astype(np.int64)
        
        response_worst_case_AR = decimation_filter.evolve_AR(sig_in=input_AR_worst_case_Q)
    except FilterOverflowError as e:
        print(f"an overflow error happened in the worst-case analysis of the AR part of the filter!\n\n{e}\n")
        return

    print("worst-case analysis for the AR part was passed!")
    print(f"maximum amplitude in the AR part encountered in worst-case analysis: {np.max(np.abs(response_worst_case_AR))}")
    print(f"number of bits devoted to AR: {decimation_filter.B_w}, maximum possible amplitude in AR part: {2**(decimation_filter.B_w-1)}!\n\n")
    
    
    #* MA part of the filter 
    try:
        response_AR = decimation_filter.evolve_AR(sig_in=impulse_Q)
        response_MA = decimation_filter.evolve_MA(sig_in=response_AR)
        
        input_MA_worst_case = np.concatenate([np.sign(response_MA[::-1]), np.zeros_like(response_MA)])
        input_MA_worst_case_Q = ( (2**(decimation_filter.num_bits_in -1)-1) * input_MA_worst_case).astype(np.int64)
        
        response_worst_case = decimation_filter.evolve_AR(sig_in=input_MA_worst_case_Q)
        response_worst_case_MA = decimation_filter.evolve_MA(sig_in=response_worst_case)
        
    except FilterOverflowError as e:
        print(f"an overflow error happened in the worst-case analysis of the MA part of the filter!\n\n{e}\n")
        return
    
    print("worst-case analysis for the MA part of the filter was passed!")
    print(f"maximum amplitude in the MA part encountered in worst-case analysis: {np.max(np.abs(response_worst_case_MA))}!")
    print(f"number of bits devoted to the MA part: {decimation_filter.B_out}, maximum possible amplitude in MA part: {2**(decimation_filter.B_out-1)-1}!\n\n")
    
    
    #* full filter including the surplus factor
    response_truncated, rel_distortion = decimation_filter.evolve_full(sig_in=input_MA_worst_case_Q)
    print(f"maximum possible amplitude in the output: {2**(decimation_filter.num_bits_in-1)}!")
    print(f"maximum amplitude in the worst-case: {np.max(np.abs(response_truncated))}, relative truncation distortion: {100*rel_distortion:0.2f} %!")
    
    
    

#! how filter behaves in tersm of aliasing in the presence of other low-pass filteirng down within the chip
def aliasing_performance():
    """ this module verifies the aliasing performance of the filter in the analog simulation """
    
    #* frequency response of the filter
    num_freq_samples = 10_000

    f_vec, f_response = sp.freqz(
        b=adc.decimation_filter.filt_Q[0]*adc.decimation_filter.surplus_factor,
        a=adc.decimation_filter.filt_Q[1]/adc.decimation_filter.filt_Q[1][0],
        worN=num_freq_samples,
        fs=adc_oversampling*audio_fs,
    )
    
    f_response_dB = 20*np.log10(np.abs(f_response))
    cross_20_dB = f_vec[np.sum(f_response_dB>-20)-1]
    loss_20K = f_response_dB[np.sum(f_vec<=target_audio_passband)]
    
    loss_low_freq_dB = f_response_dB[0]
    loss_low_freq = 10**(loss_low_freq_dB/20)
    
    # compute the phase response as well
    phase = np.angle(f_response)
    phase_diff = np.mod(np.diff(phase), 2*np.pi)
    phase_diff[phase_diff>np.pi] -= 2*np.pi
    phase_diff[phase_diff<-np.pi] += 2*np.pi
    
    phase_rec = np.cumsum(np.concatenate([[phase[0]], phase_diff]))
    
    
    #* aliasing performance of the filter
    # NOTE:
    # freqs lower than fs: correspond to the freq response of the filter
    # freqs higher than fs: correspond to the aliasing response of the filter
    num_freq_samples = 200
    start_freq = 100
    end_freq = 5*audio_fs
    
    freq_vec = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), num_freq_samples))
    num_periods = 100
    
    
    alias_response, clipping_dist = eval_module.evolve(
        freq_vec=freq_vec, 
        num_periods=num_periods
    )
    
    # just for simplicity of plotting
    clipping_dist += 10**(-2.48)
    
    
    #* plot the results
    plt.figure(figsize=(12,14))
    
    plt.subplot(511)
    plt.plot(decimation_filter.filt_Q[0], '.', markersize=10, label='b_Q')
    plt.plot(decimation_filter.filt_Q[1], '.', markersize=10, label='a_Q')
    plt.legend()
    
    plt.title(f'filter taps:\nb:{decimation_filter.filt[0]}, a:{decimation_filter.filt[1]}\n'+\
            f'b_Q:{decimation_filter.filt_Q[0]}, a_Q:{decimation_filter.filt_Q[1]}, surplus_Q:{decimation_filter.surplus_factor_Q}\n'+\
            f"IIR decimation filter: 20dB cross-freq: {int(cross_20_dB)} Hz, 20KHz loss: {loss_20K:0.2f} dB\nfs:{int(adc_oversampling*audio_fs)}, adc-ovs factor:{adc_oversampling}"
    )
    plt.ylabel('low-pass filter')
    plt.grid(True)
    
    
    plt.subplot(512)
    plt.plot(f_vec, phase_rec)
    plt.axvline(x=decimation_filter.audio_passband, linewidth=2, color='g', label=f'filter 3dB ~ {decimation_filter.audio_passband/1000:0.1f} KHz')
    plt.ylabel('phase response')
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(513)
    min_level_dB = -40
    max_level_dB = 2
    plt.semilogx(f_vec, 20 * np.log10(np.abs(f_response)))
    plt.grid(True, which="both")
    plt.ylim([min_level_dB, max_level_dB])
    plt.axvline(x=adc_oversampling*audio_fs/2, linewidth=2, color='r', label=f'{adc_oversampling/2} fs')
    plt.plot([target_audio_passband], [loss_20K], '.', markersize=10)
    plt.plot([cross_20_dB], [-20], '.', markersize=10)
    plt.legend()
    plt.ylabel("frequency response")
    
    
    plt.subplot(514)
    plt.ylabel("empirical aliasing response")
    plt.ylim([-50,2])
    plt.axvline(x=target_audio_passband, color='g', label=f'filter 3dB = {target_audio_passband/1000:0.1f} KHz')
    plt.axvline(x=audio_fs/2, color = 'r', linewidth=2, label = f'audio fs/2 ~ {audio_fs/2000:0.1f} KHz')
    plt.axvline(x=max(freq_vec), color = 'k', linewidth=2, label=f'{int(max(freq_vec)/audio_fs)} audio fs ~ {max(freq_vec)/1000:0.1f} KHz')
    plt.semilogx(freq_vec, 20*np.log10(alias_response), label="alias resp")
    plt.plot([freq_vec[0]], [alias_response[0]], label=f"0Hz loss: {loss_low_freq:0.3f}, {loss_low_freq_dB:0.3f} dB")
    plt.grid(True, which="both")    
    plt.legend()
    
    plt.subplot(515)
    plt.xlabel("freq (Hz)")
    plt.ylabel("clipping distortion")
    plt.ylim([-50,0])
    plt.axvline(x=audio_fs/2, linewidth=2, color='r', label=f'audio fs/2 ~ {audio_fs/2000:0.1f} KHz')
    plt.axvline(x=max(freq_vec), linewidth=2, color='k', label=f'{int(max(freq_vec)/audio_fs)} fs ~ {int(max(freq_vec))}')
    plt.semilogx(freq_vec, 20*np.log10(clipping_dist), label="clipping distortion")
    plt.grid(True, which="both")    
    plt.legend()
    
    plt.show()
    
  
def main():
    freq_response()
    #sinusoid_performance()
    #random_performance()
    #worst_case_performance()
    #aliasing_performance()
    
    
if __name__ == '__main__':
    main()