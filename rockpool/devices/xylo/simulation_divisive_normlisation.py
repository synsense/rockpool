"""
this function simulates the divisive normalization and investigates its effect
on cross-channel and temporal correlation of a multi-channel spike signal
"""

# required packages

# we turn off the warnings from JAX
import sys
import warnings

# ignore JAX warnings.
warnings.filterwarnings('ignore')


import numpy as np
import matplotlib.pyplot as plt

from xylo_divisive_normalisation import DivisiveNormalisation, build_lfsr

from scipy.special import ndtri

from scipy import signal





## set the parameters for the simulation

shape = (2, 1)  # shape of the Module

chan_num = 2  # number of input channels (for correlation only 2 is enough)

bits_counter = 10  # number of bits in counter
E_frame_counter = np.zeros(chan_num)  # initialize internal counters to 0

IAF_counter = np.zeros(chan_num)  # initialize the internal state of IAF counters

bits_lowpass = 16  # number of bits for implementation of low-pass filter
bits_shift_lowpass = 5  # number of bit shifts for low-pass filter
init_lowpass = 0  # initial value of low-pass filter

fs = 10e3  # global clock frequency in Hz

frame_dt = 50e-3  # frame duration in sec

bits_lfsr = 10  # number of LFSR bits

lfsr_filename = './lfsr_data.txt'  # file name for loading the LFSR code
code_lfsr = build_lfsr(lfsr_filename)  # build the LFSR code

p_local = 12  # ratio  between the local and global clock


## auxiliary functions



# produce two random variables in range {0,1,..., dim-1} with a given joint histogram
def joint_rv_generator(hist: np.ndarray, length: int):
    # extract the dimension from the histogram
    (dim1, dim2) = hist.shape
    if dim1 != dim2:
        warnings.warn('not a valid histogram')
        return
    # save the dimension
    dim = dim1

    # check the positivity
    if any(hist.ravel() < 0):
        warnings.warn('histogram with negative components!')
        return

    # random noise
    rand_vec = np.random.rand(length).reshape(length, 1)

    # convert the histogram into a vector
    hist_vec = hist.ravel()
    hist_vec.reshape(1, hist_vec.size)

    # use the values of histogram for intervaling the random variable
    hist_vec_cumsum = np.cumsum(hist_vec)

    # find the random index in the range [0, dim*dim-1]
    rand_index = np.sum(rand_vec > hist_vec_cumsum, axis=1)

    # convert the random index into (i,j) index in the range [0, dim-1], [0, dim-1]
    rand_index_i = np.floor(rand_index / dim).astype('int')
    rand_index_j = np.floor(rand_index % dim).astype('int')

    # output vector
    output = np.zeros((length, 2))
    output[:, 0] = rand_index_i
    output[:, 1] = rand_index_j

    return output




## Test 1:
# see how across_channel independent input spikes get statistically correlated
# as a result of using common LFSRs in the implementation
# no correlation in time


print('\n', '='*15, "Test 1", '='*15)

sim_time = 10  # simulation time in sec
num_in_sample = np.ceil(fs * sim_time).astype('int')  # number of samples in input spikes signals

# probability of firing of the Poisson point process
poisson_prob_in = 0.1

input_rate = np.ceil(fs * poisson_prob_in).astype('int')

# joint histogram and spike generation
marginal_hist = np.array([1 - poisson_prob_in, poisson_prob_in]).reshape(2, 1)
joint_hist = np.outer(marginal_hist, marginal_hist)

input_spike = joint_rv_generator(joint_hist, num_in_sample)

# build the 2-channel simulator
simulator_test1 = DivisiveNormalisation(shape=(chan_num, 1), chan_num=chan_num, bits_counter=bits_counter,
                                        bits_lowpass=bits_lowpass,
                                        bits_shift_lowpass=bits_shift_lowpass, fs=fs, frame_dt=frame_dt,
                                        bits_lfsr=bits_lfsr, code_lfsr=code_lfsr, p_local=p_local)

# simulate the system
output_spike, state = simulator_test1.evolve(input_spike=input_spike, record=True)

# compute the metrics
corr_factor_mat_in, cov_mat_in, mean_vec_in = simulator_test1.corr_metric(input_spike)
corr_factor_mat_out, cov_mat_out, mean_vec_out = simulator_test1.corr_metric(output_spike)

print(f'empirical rate at the input: {mean_vec_in*fs}')
print(f'empirical correlation factor at the input: {corr_factor_mat_in[0,1]}\n\n')

print(f'empirical rate at the output: {mean_vec_out*fs*p_local}')
print(f'empirical correlation factor at the output: {corr_factor_mat_out[0,1]}')



## Test 2:
# produce correlated firing patterns across channles
# no correlation in time

print('\n', '='*15, "Test 2", '='*15, '\n')

sim_time = 10  # simulation time in sec
num_in_sample = np.ceil(fs * sim_time).astype('int')  # number of samples in input spikes signals

# probability of firing of the Poisson point process
poisson_prob_in = 0.1

input_rate = np.ceil(fs * poisson_prob_in).astype('int')

# a toy example for creating correlation:
# if one channel fires, the other does not
# note that in this case "poisson_prob_in" should be in [0,0.5]
joint_hist = np.zeros((2, 2))

joint_hist[0, 0] = 1 - 2 * poisson_prob_in
joint_hist[0, 1] = poisson_prob_in
joint_hist[1, 0] = poisson_prob_in
joint_hist[1, 1] = 0  # this avoid simultaneous firings

# produce correlated spikes
input_spike = joint_rv_generator(joint_hist, num_in_sample)

# build the 2-channel simulator
simulator_test2 = DivisiveNormalisation(shape=(chan_num, 1), chan_num=chan_num, bits_counter=bits_counter,
                                        bits_lowpass=bits_lowpass,
                                        bits_shift_lowpass=bits_shift_lowpass, fs=fs, frame_dt=frame_dt,
                                        bits_lfsr=bits_lfsr, code_lfsr=code_lfsr, p_local=p_local)

# simulate the system
output_spike, state = simulator_test2.evolve(input_spike=input_spike, record=True)

# compute the metrics
corr_factor_mat_in, cov_mat_in, mean_vec_in = simulator_test2.corr_metric(input_spike)
corr_factor_mat_out, cov_mat_out, mean_vec_out = simulator_test2.corr_metric(output_spike)

print(f'empirical rate at the input: {mean_vec_in*fs}')
print(f'empirical correlation factor at the input: {corr_factor_mat_in[0,1]}\n\n')

print(f'empirical rate at the output: {mean_vec_out*fs*p_local}')
print(f'empirical correlation factor at the output: {corr_factor_mat_out[0,1]}')

## Test 3
# take the effect of time correlation into account
print('\n', '=' * 15, "Test 3", '=' * 15, '\n')

sim_time = 10  # simulation time in sec
num_in_sample = np.ceil(fs * sim_time).astype('int')  # number of samples in input spikes signals

# Method:
# (i) produce correlated Gaussian variables (X1,X2)
# (ii) produce spike as (u(X1-trunc_thre), u(x2-trunc_thre)) with step function u(.)
# and a truncation threshold "trunc_thre"

# compute the truncation_threshold
corr_factor = 0.999  # correlation factor
poisson_prob_in = 0.1  # firing rates

trunc_thre = ndtri(1 - poisson_prob_in)

# i.i.d. Gaussain RV
G = np.random.randn(2 * num_in_sample).reshape(num_in_sample, 2)

# parameters of the 1-st order filter
a_filt = np.array([1, -corr_factor])
b_filt = np.array([np.sqrt(1 - corr_factor ** 2)])

# rescale the first row of G to converge to the stationary solution at the first step
G[0, :] /= b_filt

G_corr = signal.lfilter(b_filt, a_filt, G, axis=0)

# quantize using the threshold to obtain the correlated spikes (correlation just in time)
input_spike = (G_corr > trunc_thre)

# build the 2-channel simulator
# change p_local for short-correlation checking
p_local = 2
simulator_test3 = DivisiveNormalisation(shape=(chan_num, 1), chan_num=chan_num, bits_counter=bits_counter,
                                        bits_lowpass=bits_lowpass,
                                        bits_shift_lowpass=bits_shift_lowpass, fs=fs, frame_dt=frame_dt,
                                        bits_lfsr=bits_lfsr, code_lfsr=code_lfsr, p_local=p_local)

# simulate the system
output_spike, state = simulator_test3.evolve(input_spike=input_spike, record=True)

# also investigate the correlation across time
# compute the correlation up to a specific delay "corr_length"
# method: convert each channel into "corr_length" channels each having
# a different delay

corr_length = 10
channel = 1

# at the input
delay_sample_in = np.zeros((input_spike.shape[0] - corr_length + 1, corr_length))
for i in range(corr_length):
    delay_sample_in[:, i] = input_spike[i:i + delay_sample_in.shape[0], channel]

# at the output
# note that the time-scale of output is smaller by a factor p_local
corr_length_out = p_local * corr_length
delay_sample_out = np.zeros((output_spike.shape[0] - corr_length_out + 1, corr_length_out))

for i in range(corr_length_out):
    delay_sample_out[:, i] = output_spike[i:i + delay_sample_out.shape[0], channel]

# compute covariance matrix and correlation factors
corr_factor_mat_in, cov_mat_in, mean_vec_in = simulator_test3.corr_metric(delay_sample_in)
corr_factor_mat_out, cov_mat_out, mean_vec_out = simulator_test3.corr_metric(delay_sample_out)

# average out corr_factor_mat_out by a factor p_local
corr_factor_mat_out_adjusted = corr_factor_mat_out[::p_local, ::p_local]

"""
print(f'* input * empirical correlation-factor matrix for channel {channel}: {corr_factor_mat_in}')
print(f'* output * empirical correlation-factor matrix for channel {channel}: {corr_factor_mat_out}')
"""

plt.figure(3, figsize=[8, 15])
plt.subplot(121)
fig = plt.imshow(np.abs(corr_factor_mat_in))
plt.colorbar(fig, shrink=0.25)
plt.title(f'* input * correlation in time (length={corr_length})')
plt.ylabel('channel #1')
plt.subplot(122)
fig = plt.imshow(np.abs(corr_factor_mat_out_adjusted))
plt.title(f'* output * correlation in time (length={corr_length})')
plt.colorbar(fig, shrink=0.25)

print('End of Simulation!')
