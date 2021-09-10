from rockpool.nn.modules.module import Module

from rockpool import TSEvent

from rockpool.typehints import P_int, P_float

from rockpool.parameters import Parameter, State, SimulationParameter

import numpy as np

from tqdm.autonotebook import tqdm, trange

import warnings

class DivisiveNormalisation(Module):
    def __init__(
        self,
        shape: tuple,
        chan_num: int = 1,
        bits_counter: int=10,
        bits_lowpass: int = 16,
        bits_shift_lowpass: int = 5,
        fs: float = 10e3,
        frame_dt: float = 50e-3,
        bits_lfsr: int = 10,
        p_local: int = 12,
    ):
        super().__init__(shape, spiking_input=True, spiking_output=True)
        
        # number of input channels
        self.chan_num=chan_num
        
        # initilaize the value of the counter to zero and set its size
        self.counter: P_int = State(np.zeros(self.size_in, "uint"))
        self.bits_counter: P_int=SimulationParameter(bits_counter)
        
        # set the parameters of the low-pass filter (implemented by shifts)
        self.bits_lowpass: P_int = SimulationParameter(bits_lowpass)
        self.bits_shift_lowpass: P_int = SimulationParameter(bits_shift_lowpass)
        
        # set the global clock frequency
        self.fs: P_float = SimulationParameter(fs)
        
        # set the period of the frame
        self.frame_dt: P_float=SimulationParameter(frame_dt)
        
        # set the ratio between the rate of local and global clocks
        # note that due to return-to-zero pulses, the spike rate increases
        # by only p_local/2 -> set p_local to be an even number
        self.p_local=((1+p_local)/2)*2
        if(self.p_local != p_local):
            warnings.warn(f'p_local={p_local} is not an even integer!')


    def evolve(
        self, input_spike: np.ndarray, record: bool = False
    ) -> (np.ndarray, dict):
        # - Convert input spikes with duration 'dt' to frames -> counter output
        # E(t) of duration 'frame_dt'
        # - input : (T, chan_num) -> T is units of 'dt'
        # - E: (T_frames, chan_num) -> units of 'frame_dt'
        
        # counter state
        ts_input = TSEvent.from_raster(input_spike, dt=1 / self.fs)
        E = ts_input.raster(dt=self.frame_dt, add_events=True)
        E = np.clip(E, 0, 2 ** self.bits_counter).astype("uint")

        # - Perform low-pass filter on E(t)-> M(t)
        # - M: (T_frame, Nin) -> units of 'frame_dt'
        #  M(t) = s * E(t) + (1-s) M(t-1)
        M = np.zeros((E.shape[0] + 1, E.shape[1])).astype("uint")
        M[0, :] = self.counter
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
        max_spike_in_frame=np.ceil(self.frame_dt*self.fs)
        
        # total number of spikes at the output
        len_output_spike=max_spike_in_frame*E.shape[0]*self.p_local/2
        
        # initialize the matrix of output spikes in all channels
        output_spike=np.zero(len_output_spike,E.shape[1])
        
        # LFSR noise shared among all channels
        # for the moment we are using pure random noise
        lfsr_noise=np.random.uniform(0, 2**self.nbits_lfsr, E.shape[0], max_spike_in_frame)
            
        # perform operation per channel
        for ch in range(E.shape[1]):
            # for each channel
            E_ch=np.copy(E[:,ch])
            M_ch=np.copy(M[:,ch])
            
            # repeat Each E(t) 'max_spike_in_frame' times
            E_ch=E_ch.reshape(E_ch.size,1) # column-vec
            E_ch_rep=np.repeat(E_ch,max_spike_in_frame, axis=1)
            
            # compare the repeated values with LFSR output
            
            
            # Spike train generate by SG: each row contains spikes generated in a specific frame
            # units of time are 'dt'
            S_sg=(E_ch_rep>=lfsr_noise)
            
            # multiply the frequency of spikes by a factor p_local/2
            # unwrap all frames in time (column vec) -> repeat along column -> wrap again larger frame
            # ->
            # length of expanded frame 
            len_frame_local=max_spike_in_frame*self.p_local/2
            
            # repeat the spikes in each frame
            S_local=np.repeat(S_sg.reshape(S_sg.size,1), self.p_local/2, axis=1).reshape(-1,len_frame_local)
            
            # apply IAF with threshold M(t) at each frame t (each row of S_local)
            # due to surplus from frame t-> t+1, we need to do this frame by frame
            output_spike_ch=np.zero(S_local.shape, dtype='int64')
            
            res_from_previous_frame=0
            
            for t in range(S_local.shape[0]):
                # find the largest integer less than the threshold M(t)
                threshold=np.ceil(M_ch[t]).astype('int')
                                  
                # compute the cumulative number of firings starting from residual
                IAF_state=( res_from_previous_frame + np.cumsum(S_local[t,:]) )%(threshold+1)
                
                # firing times are when IAF_state is equal to threshold
                output_spike_ch[t,IAF_state==threshold]=1
                
                # compute the residual for the next frame
                res_from_previous_frame=output_spike_ch[t,-1]
            
            # unwrap the spikes and copy it in the output_spike for the channel
            output_spike[:,ch]=output_spike_ch.reshape(output_spike_ch.size,1)
            
        

        # - Generate state record dictionary
        record_dict = (
            {
                "E": E,
                "M": M,

            }
            if record
            else {}
        )

        self.counter = E[-1, :]

        return output_spike, record_dict
    
    
    
    # build an instant
    sim=DivisiveNormalisation(p_chan=1, bits_counter=10, bits_lowpass=16, bits_shift_lowpass=5, f_s=10e3, frame_dt=50e-3, bits_lfsr=10, p_local=12)
    
    
    # build a random input
    T=10000
    prob=0.05
    
    in_spike=(np.random.rand(T,1)<=p)
    
