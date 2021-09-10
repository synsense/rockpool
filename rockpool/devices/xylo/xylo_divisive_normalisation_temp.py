""""
from rockpool import TSEvent

from rockpool.nn.modules.module import Module

from rockpool.typehints import P_int, P_float

from rockpool.parameters import Parameter, State, SimulationParameter

from tqdm.autonotebook import tqdm, trange
"""


import numpy as np

import matplotlib.pyplot as plt

import warnings



class DivisiveNormalisation:
    def __init__(
        self,
        chan_num: int = 1,
        bits_counter: int=10,
        E_frame_counter: np.array=None,
        IAF_counter: np.array=None,
        bits_lowpass: int = 16,
        bits_shift_lowpass: int = 5,
        init_lowpass: 'uint'= 0,
        fs: float = 10e3,
        frame_dt: float = 50e-3,
        bits_lfsr: int = 10,
        code_lfsr: np.array =None,
        p_local: int = 12,
        
    ):
        #super().__init__(shape, spiking_input=True, spiking_output=True)
        
        # number of input channels
        self.chan_num=chan_num
        
        # initilaize the value of the counter to zero and set its size
        if (E_frame_counter==None):
            self.E_frame_counter=np.zeros(self.chan_num, 'uint')
        elif (E_frame_counter.size !=chan_num):
            warnings.warn(f'we need one initialization per counter for each cahnnel:\n  number of channels: {self.chan_num} \n size of initialization: {E_frame_counter.size}')
            self.E_frame_counter=np.zeros(self.chan_num, 'unit')
        else:
            self.E_frame_counter=np.copy(E_frame_counter).astype('uint')

        self.bits_counter=bits_counter
        
        # initialize the state of IAF_counter
        if (IAF_counter==None):
            self.IAF_counter=np.zeros(self.chan_num, 'uint')
        elif (IAF_counter.size !=chan_num):
            warnings.warn(f'we need one initialization per counter for each cahnnel:\n  number of channels: {self.chan_num} \n size of initialization: {IAF_counter.size}')
            self.IAF_counter=np.zeros(self.chan_num, 'uint')
        else:
            self.IAF_counter=np.copy(IAF_counter).astype('uint')
        
        ##
        # set the parameters of the low-pass filter (implemented by shifts)
        self.bits_lowpass=bits_lowpass
        self.bits_shift_lowpass=bits_shift_lowpass
        
        # at the moment we implement averaging filter such that its value is
        # always a positive initeger. We can do much better by using some 
        # floating implemntation by using some bits as decimals
        # this is left for future
        self.init_lowpass=init_lowpass
        
        
        
        ##
        # set the global clock frequency
        self.fs=fs
        
        # set the period of the frame
        self.frame_dt=frame_dt

        # set the number of bits and also code for LFSR
        self.bits_lfsr=bits_lfsr
        if(any(code_lfsr==None)):
            warnings.warn('an LFSR code is obligatory for system simulation')
        elif(code_lfsr.size != 2**self.bits_lfsr-1):
            warnings.warn('length of LFSR is not compatible with its number of bits')
        else:
            self.code_lfsr=np.copy(code_lfsr)
        
        
        # - set the ratio between the rate of local and global clocks
        # - note that due to return-to-zero pulses, the spike rate increases
        #   by only p_local/2 -> set p_local to be an even number
        self.p_local=int((1+p_local)/2)*2
        if(self.p_local != p_local):
            warnings.warn(f'p_local={p_local} was rounded to an even integer!')






    def evolve(
        self, input_spike: np.ndarray, record: bool = False
    ) -> (np.array, dict):
        # check the dimensionality first
        if (input_spike.shape[1] != self.chan_num):
            warnings.warn(f'number of columns in the input is not the same as number of channels f{self.chan_num}')
            return

        # - Convert input spikes with duration 'dt' to frames of duration 'frame_dt'-> counter output
        # - output is counter output E(t) of duration 'frame_dt'
        # - input : (T, chan_num) -> T is units of 'dt'
        # - E: (T_frames, chan_num) -> units of 'frame_dt'
        
        # number of 'dt' inside 'frame_dt' 
        frame_len=int(np.ceil(self.frame_dt*self.fs))

        # number of frames in the output
        n_frame=int(input_spike.shape[0]/frame_len)

        # counter state
        E=np.zeros((n_frame, input_spike.shape[1]))

        # do cumsum to obtain the counter state
        for i in range(n_frame):
            E[i,:]=np.sum(input_spike[i*frame_len: (i+1)*frame_len], axis=0)
        
        # add the effect of initial values in E_frame_counter
        E[0,:]+=self.E_frame_counter
        
        # clip the counter state when it saturates
        E = np.clip(E, 0, 2 ** self.bits_counter).astype("uint")
        
        
        # regsiter the value of E_frame_counter 
        self.E_frame_counter=E[-1,:]


        # - Perform low-pass filter on E(t)-> M(t)
        # - M(t) = s * E(t) + (1-s) M(t-1)
        #   with s=1/2**bits_shift_lowpass
        # - M: (T_frame, chan_num) -> units of 'frame_dt'
        
        M = np.zeros((E.shape[0] + 1, E.shape[1])).astype("uint")
        
        # load the initialization of the filter
        M[0,:]=self.init_lowpass
        
        # we have implemented the improved version of filer to avoid truncation
        # 
        for t in range(E.shape[0]):
            # M[t+1, :] = (E[t, :] >> int(self.bits_shift_lowpass) + M[t, :] - M[t, :] >> int(self.bits_shift_lowpass))
            M[t + 1, :] = (E[t, :] + (M[t, :] << int(self.bits_shift_lowpass)) - M[t, :]) >> int(
                self.bits_shift_lowpass
            )
        
        M = M[1:, :]
        
        # take the limitted number of counter bits into account
        # we should make sure that teh controller does not allow count-back to zero
        # and keeps the value of the counter at its maximum 
        M = np.clip(M, 0, 2 ** self.bits_lowpass -1).astype("uint")

        # use the value of E(t) at each frame t to produce a pseudo-random
        # Poisson spike train using LFSR
        # as the value of LFSR varies with global clock rate f_s, we have 'frame_dt*f_s'
        # samples in each frame
        # the timing of the output in in units of 'dt'
        
        # maximum number of spikes within a frame period
        max_spike_in_frame=int(np.ceil(self.frame_dt*self.fs))
        
        # total number of spikes at the output
        len_output_spike=int(max_spike_in_frame*E.shape[0]*self.p_local)
        
        # initialize the matrix of output spikes in all channels
        output_spike=np.zeros((len_output_spike,E.shape[1]))

        # compute a slice of LFSR code used for comparison among different frames
        # find the total number of comparsions between all E(t) and LFSR output
        # this is given by the total number of spikes
        num_all_spikes=max_spike_in_frame*E.shape[0]

        # number of LFSR periods needed 
        num_lfsr_period=int(np.ceil(num_all_spikes/(code_lfsr.size)))

        # LFSR output needed in frames and along each frame
        self.code_lfsr.reshape(1,self.code_lfsr.size)

        code_lfsr_frame=np.tile(self.code_lfsr, num_lfsr_period)[:num_all_spikes].reshape(E.shape[0],max_spike_in_frame)

        
        # initalize the IAF_state for further isnpection
        if (record):
            IAF_state_saved=np.zeros((num_all_spikes*self.p_local,self.chan_num),dtype='uint')
            
                
        # perform operation per channel
        for ch in range(E.shape[1]):
            # for each channel
            E_ch=np.copy(E[:,ch])
            M_ch=np.copy(M[:,ch])
            
            # repeat Each E(t) 'max_spike_in_frame' times
            E_ch=E_ch.reshape(E_ch.size,1) # column-vec
            E_ch_rep=np.tile(E_ch,(1,max_spike_in_frame))
        
            # Spike train generate by SG: each row contains spikes generated in a specific frame
            # units of time are 'dt'
            S_sg=np.int_( E_ch_rep>=code_lfsr_frame)
            
            # multiply the frequency of spikes by a factor p_local/2
            # unwrap all frames in time (column vec) -> repeat along column -> wrap again larger frame
            # ->
            # length of expanded frame 
            len_frame_local=int(max_spike_in_frame*self.p_local/2)
            
            # repeat the spikes in each frame
            # note that due to return-to-zero pulse, each spike needs to be zero padded at the output
            S_local_before_pad=np.tile(S_sg.reshape(S_sg.size,1), (1, int(self.p_local/2)) ).reshape(-1,len_frame_local)
            
            S_local=np.zeros((S_local_before_pad.shape[0], 2*S_local_before_pad.shape[1]), dtype='uint')
            S_local[:, :S_local_before_pad.shape[1]]=S_local_before_pad
            
            
            # apply IAF with threshold M(t) at each frame t (each row of S_local)
            # due to surplus from frame t-> t+1, we need to do this frame by frame
            output_spike_ch=np.zeros(S_local.shape, dtype='int64')
            
            # initialize the state of the corresponding counter
            res_from_previous_frame=self.IAF_counter[ch]
            
            
            for t in range(S_local.shape[0]):
                # find the largest integer less than the floating-value threshold M(t)
                # this way of thresholding works because the IAF is implemented by counter
                # so, we need to set the value of threshould to be an integer value
                # some care is needed when M(t)<1 because then IAF fires everytime a 
                # spike comes from the local generator
                # we solve this by simply adding the threshold by 1
                threshold=np.ceil(M_ch[t]).astype('int')+1
                                  
                # compute the cumulative number of firings starting from residual and take mode
                IAF_state=( res_from_previous_frame + np.cumsum(S_local[t,:]) )%threshold
                
                # save if needed
                if (record):
                    IAF_state_saved[t*IAF_state.size: (t+1)*IAF_state.size,ch]=np.copy(IAF_state)
                
                # to find the firing times, we need to find those times for which
                # IAF_state[t]-IAF_state[t+1]<0
                # +1 is needed because of the delay we added
                firing_times_in_frame=np.argwhere((IAF_state[1:]-IAF_state[0:-1]) < 0)+1
                
                # regsiter these firing times
                output_spike_ch[t,firing_times_in_frame]=1
                
                # since we did not cosider the first firing time
                if(t>=1 and IAF_state[0]==0 and res_from_previous_frame>0 ):
                    output_spike_ch[t,0]=1
                
                # compute the residual for the next frame
                res_from_previous_frame=output_spike_ch[t,-1]
            
            # register the state of the IAF counter
            self.IAF_counter[ch]=res_from_previous_frame
            
            # unwrap the spikes and copy it in the output_spike for the channel
            # since we are not worried about modification: no need for copy -> ravel
            output_spike[:,ch]=output_spike_ch.ravel()
            
        
        
        # - Generate state record dictionary
        record_dict = (
            {
                "E": E,
                "M": M,
                "E_frame_counter": self.E_frame_counter,
                "IAF_counter": self.IAF_counter,
                "IAF_state_saved":  IAF_state_saved,

            }
            if record
            else {}
        )


        return output_spike, record_dict
    


# read the LFSR state and build the pseudo-random code
with open('./lfsr_data.txt') as f:
    lines = f.readlines()

# for some reason state 'all-zero' is included in the file but
# LFSR state cannot be all-zero because then the next state also would be
# all-zero
code_lfsr=np.zeros(len(lines), dtype='int')

for i in range(len(lines)-1):
    code_lfsr[i]=int(lines[i],2)
    
# remove the first element corresponding to the all-zero state
code_lfsr=code_lfsr[:-1]+1


    
# build an instant
sim=DivisiveNormalisation(chan_num=1, bits_counter=10, bits_lowpass=16, bits_shift_lowpass=4, fs=10.0e3, frame_dt=50e-3, bits_lfsr=10, code_lfsr=code_lfsr, p_local=12)
    
    
# build a random input
T=1000000
prob=0.1
    
input_spike=np.array( (np.random.rand(T,1)<=prob), dtype='int')

input_spike_supp=np.argwhere(input_spike==1)
input_spike_value=np.zeros(input_spike_supp.shape)

plt.plot(input_spike_supp, input_spike_value, '.')  
plt.xlabel('time')
plt.ylabel('firings')
plt.show()


# print the empirical probability of firing
print('empirical frequency=', np.sum(input_spike==1)/input_spike.size)
# simulate the system

