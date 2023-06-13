# -----------------------------------------------------------------------------------------------------------------------
# This module designs a simple IIR low-pass filter + decimation for Xylo-A3 via optimization.
#
# NOTE: This module contains the block-diagram implementation of the filter to make sure that it is compatible with Hardware.
#       To design this filter, we apply optimization to reduce the aliasing nosie due to sampling as much as possible.
#       Also, we use the class of IIR Elliptic filters to get the sharpest transition and smallest aliasing noise.
#
#       For further details on the implementation of this filter and its performance evaluation, please refer to the original design repo
#       https://spinystellate.office.synsense.ai/saeid.haghighatshoar/anti-aliasing-filter-for-xylo-a2
# 
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 13.06.2023
# ------------------------------------------------------------------------------------------------------------------------


#===========================================================================
#*                    Blockdiagram of the Analog Front-end 
#===========================================================================

#---------------------------------------------------------------------------
#     Equivalent ADC module implemented in Analog + Digital domain:
#---------------------------------------------------------------------------
#* Analog Signal --> 
#*          Fixed-Gain Amplifier -->
#*                  Analog Aliasig filter with 20K corner frequency -->
#*                          Programmable-Gain Amplifier (PGA) --> 
#*                                  High Sampling Rate ADC (with oversampling 2 or 4) --> 
#*                                          Anti-Aliasing Low-pass filter + Decimation module (output of this module is the audio signal with target audio sampling rate) -->

#
# followed by
#

#---------------------------------------------------------------------------
#  AGC module to adjust the amplitude of the signal via feedback through PGA
#---------------------------------------------------------------------------   



import numpy as np
from tqdm import tqdm

from dataclasses import dataclass
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.pdm_adc import AUDIO_SAMPLE_RATE

from numpy.linalg import norm

from typing import Tuple
from warnings import warn


#================================================
#*   parameters used in the hardware
#================================================
# 3dB corner frequency of the low-pass filter
LOW_PASS_CORNER = 20_000

# power supply voltage in Xylo-A3
XYLO_VCC = 1.1
XYLO_MAX_AMP_UNIPOLAR = XYLO_VCC/2.0
XYLO_MAX_AMP = XYLO_VCC # this is due to the differential structure of the amplifier






#===========================================================================
#                            SECTION HEADER
#===========================================================================
                           



class FilterOverflowError(Exception):
    pass


#================================================
#*               Block-Diagram Model
#================================================
@dataclass
class BlockDiagram:
    # oversampling factor: the block works corresponds to how much ADC oversampling
    adc_oversampling_factor:int
    
    # clock rate with which the block-diagram should be simulated to be matched with other modules
    fs:float
    
    # number of bits in the input
    B_in:int
    
    # bitwidth of the AR output w[n]
    B_w:int
    
    # filter AR taps
    a_taps:np.ndarray
    
    # number of bits devoted to the AR taps
    B_a:int
    
    # number of bits devoted to the fractional part of AR taps
    B_af:int
    
    # filter MA taps
    b_taps:np.ndarray
    
    # number of bits devoted to b-tap
    B_b:int
    
    # number of bits devoted to the output of MA part
    B_out:int
    
    # surplus factor for adjusting the gain for clipping
    surplus:int
    
    # number of bits devoted to surplus factor
    B_sur:int
    

#===========================================================================
#*         block diagram used for oversampling factor 2 and 4
#===========================================================================
#! Note: we have no implementation for other oversampling factors
bd_oversampling_0 = None
bd_oversampling_1 = None

bd_oversampling_2 = BlockDiagram(
    adc_oversampling_factor=2,
    fs=2*AUDIO_SAMPLE_RATE,
    B_in=10,
    B_w=17,
    a_taps=np.asarray([65536, -76101, 93600, -46155, 15598], dtype=np.int64),
    B_a=18,
    B_af=16,
    b_taps=np.asarray([45, 63, 96, 63, 45], dtype=np.int64),
    B_b=8,
    B_out=22,
    surplus=163,
    B_sur=8,
)

bd_oversampling_3 = None
    
bd_oversampling_4 = BlockDiagram(
    adc_oversampling_factor=4,
    fs=4*AUDIO_SAMPLE_RATE,
    B_in=10,
    B_w=17,
    a_taps=np.asarray([32768, -93468, 113014, -65651, 15547], dtype=np.int64),
    B_a=18,
    B_af=15,
    b_taps=np.asarray([46, -62, 96, -62, 46], dtype=np.int64),
    B_b=8,
    B_out=22,
    surplus=33,
    B_sur=8,
)


bd_list = [
    bd_oversampling_0,
    bd_oversampling_1,
    bd_oversampling_2,
    bd_oversampling_3,
    bd_oversampling_4,
]



class AntiAliasingDecimationFilter:
    def __init__(self, adc_oversampling_factor:int=2):
        """this class simulates the block-diagram model of the decimation anti-aliasing filter
        adc_oversampling_factor (int, optional): oversampling factor of ADC. Defaults to 2.
        """
        self.adc_oversampling_factor = adc_oversampling_factor
        
        try:
            self.bd = bd_list[self.adc_oversampling_factor]
            if self.bd is None: raise Exception
        except:
            raise NotImplementedError(f"decimation filter in block-diagram format is not yet implemented for oversampling factor {self.adc_oversampling_factor}!")
        
        
        #=================================================================================
        #  parameters needed for one-step simulation of the filter (as a state machine)
        #=================================================================================
        self.sample_rate = self.bd.fs
        
        self.reset()
        
        
        
    def reset(self):
        """ reset the states of the filter in one-step simulation """
        # initialize the internal state of the filter: number of bit registers in AR part
        # NOTE: we use an additional dummy register so that the computation of MA part is simplified.
        # This is not needed in the ordinary computation.
        self.state = np.zeros(len(self.bd.a_taps)-1 + 1, dtype=np.int64) 
        
        # set the sampling rate of the filter
        self.time = 0
        
        # output signal and its distortion
        self.sig_out = 0
        self.rel_distortion = 0
                
    
    #----------------------------------------------------------------------------------
    #  functions used for one-step evolution of the filter used in timed simulations.
    #----------------------------------------------------------------------------------
    
    def evolve_onestep(self, sig_in:float, time_in:float):
        #* check the range of input
        if np.max(sig_in) >= 2**(self.bd.B_in-1) or np.min(sig_in) < -2**(self.bd.B_in-1):
            raise ValueError(f"input signal should have {self.bd.B_in} and in the range [{-2**(self.bd.B_in-1)}, {2**(self.bd.B_in-1)-1}]!")
        
        # check if a new sample has arrived and needs to be processed
        if time_in >= self.time:
            #* compute AR part
            # compute the feedback part: each branch individually
            feedback_branches_out = (- self.state[:-1] * self.bd.a_taps[1:]) >> self.bd.B_af

            # add all branches
            feedback_out = np.sum(feedback_branches_out)
            
            # add the input
            next_w_sample = sig_in + feedback_out
            
            # update w
            self.state[1:] = self.state[:-1]
            self.state[0] = next_w_sample
            
            #* compute the MA part
            sig_out_MA = np.sum(self.state * self.bd.b_taps)
            
            # check the overflow in MA part of the filter
            if np.max(sig_out_MA) >= 2**(self.bd.B_out-1) or np.min(sig_out_MA) < -2**(self.bd.B_out-1):
                raise FilterOverflowError(f"Overflow in the MA branch of the filter! the output of MA should fit in {self.bd.B_out} signed bits!")
                #warn(f"Overflow in the MA branch of the filter! the output of MA should fit in {self.bd.B_out} signed bits!")
            
            #* compute the output after surplus scaling
            sig_out_surplus = (sig_out_MA * self.bd.surplus)>>self.bd.B_af
            
            #* compute the output after final truncation
            max_pos_amplitude = 2**(self.bd.B_in-1)-1
            min_neg_amplitude = -2**(self.bd.B_in-1)
            
            sig_out = sig_out_surplus
            
            if sig_out > max_pos_amplitude:
                sig_out = max_pos_amplitude
            
            if sig_out < min_neg_amplitude:
                sig_out = min_neg_amplitude
                
            
            # compute the ralative truncation distortion
            rel_distortion = np.abs(sig_out - sig_out_surplus)/np.max([np.abs(sig_out), np.abs(sig_out_surplus)])

            #* update the output signal and relative distortion
            self.sig_out = sig_out
            self.rel_distortion = rel_distortion
            
            #* update the time to get prepared for the next sample
            self.time += 1/self.sample_rate
        
        # final output sample and its distortion
        return self.sig_out, self.rel_distortion
            


    #---------------------------------------------------------------------------
    #      functions used for ordinary evolution of the filter
    #---------------------------------------------------------------------------
    
    def evolve_AR(self, sig_in:np.ndarray)->np.ndarray:
        """ the auto-regressive part of the filter """
        
        #* check the input validity
        if not isinstance(sig_in, np.ndarray) or not isinstance(sig_in[0], np.int64):
            raise ValueError("input signal should be an array contatining quantized integer-valued version of the signal!")

        #* check the range of input
        if np.max(sig_in) >= 2**(self.bd.B_in-1) or np.min(sig_in) < -2**(self.bd.B_in-1):
            raise ValueError(f"input signal should have {self.bd.B_in} and in the range [{-2**(self.bd.B_in-1)}, {2**(self.bd.B_in-1)-1}]!")
        
        # number of bit registers in AR part
        w = np.zeros(len(self.bd.a_taps)-1, dtype=np.int64)        
        
        sig_out_AR = []
        
        for sig in sig_in:
            # compute the feedback part: each branch individually
            feedback_branches_out = (- w * self.bd.a_taps[1:]) >> self.bd.B_af

            # add all branches
            feedback_out = np.sum(feedback_branches_out)
            
            # add the input
            next_w_sample = sig + feedback_out
            
            # update w
            w[1:] = w[:-1]
            w[0] = next_w_sample
            
            # register the output
            sig_out_AR.append(next_w_sample)
            
        # convert into array and return
        sig_out_AR = np.asarray(sig_out_AR, np.int64)
        
        # check the overflow in the AR part
        if np.max(sig_out_AR) >= 2**(self.bd.B_w-1) or np.min(sig_out_AR) < -2**(self.bd.B_w-1):
            raise FilterOverflowError(f"overflow in the AR branch of the filter! the output of AR should fit in {self.bd.B_w} signed bits!")
        
        
        return sig_out_AR
    
    
    def evolve_MA(self, sig_in:np.ndarray)->np.ndarray:
        """ the moving-average part of the filter """
        
        #* check the input validity
        if not isinstance(sig_in, np.ndarray) or not isinstance(sig_in[0], np.int64):
            raise ValueError("input signal should be an array contatining quantized integer-valued version of the signal!")

        
        # compute the output
        sig_out_MA = (np.convolve(self.bd.b_taps, sig_in, 'full')).astype(np.int64)
        
        # check the overflow in MA part of the filter
        if np.max(sig_out_MA) >= 2**(self.bd.B_out-1) or np.min(sig_out_MA) < -2**(self.bd.B_out-1):
            raise FilterOverflowError(f"Overflow in the MA branch of the filter! the output of MA should fit in {self.bd.B_out} signed bits!")
            #warn(f"Overflow in the MA branch of the filter! the output of MA should fit in {self.bd.B_out} signed bits!")
        
        
        return sig_out_MA
    
    
    def evolve_surplus(self, sig_in:np.ndarray)->np.ndarray:
        """ adding the surplus scaling of the filter """
        
        #* check the input validity
        if not isinstance(sig_in, np.ndarray) or not isinstance(sig_in[0], np.int64):
            raise ValueError("input signal should be an array contatining quantized integer-valued version of the signal!")
        
        sig_out_surplus = (sig_in * self.bd.surplus)>>self.bd.B_af
        
        return sig_out_surplus
    

    def evolve_truncation(self, sig_in:np.ndarray)->Tuple[np.ndarray, float]:
        """ truncating the filter output back to the number of bits in the input """
        
        max_pos_amplitude = 2**(self.bd.B_in-1)-1
        min_neg_amplitude = -2**(self.bd.B_in-1)
        
        sig_out = np.copy(sig_in)
        
        sig_out[sig_out > max_pos_amplitude] = max_pos_amplitude
        sig_out[sig_out < min_neg_amplitude] = min_neg_amplitude
        
        # compute the ralative truncation distortion
        rel_distortion = norm(sig_out - sig_in)/np.sqrt(norm(sig_in) *  norm(sig_out))
        
        return sig_out, rel_distortion
        
    
    
    def evolve_full(self, sig_in:np.ndarray)->Tuple[np.ndarray, float]:
        """ full filtering chain without any decimation at the end """
        
        sig_out_AR = self.evolve_AR(sig_in=sig_in)
        sig_out_MA = self.evolve_MA(sig_in=sig_out_AR)
        sig_out_surplus = self.evolve_surplus(sig_in=sig_out_MA)
        
        sig_out, rel_distortion = self.evolve_truncation(sig_in=sig_out_surplus)
        
        return sig_out, rel_distortion
    
    
    def evolve(self, sig_in:np.ndarray)->Tuple[np.ndarray, float]:
        """ full filtering chain with decimation at the end """
        
        sig_out_full, rel_distortion = self.evolve_full(sig_in=sig_in)
        sig_out_decimated = sig_out_full[::self.adc_oversampling_factor]
        
        return sig_out_decimated, rel_distortion
        
    
    def freq_response(self, freq_vec:np.ndarray, num_periods:int=100)->Tuple[np.ndarray, np.ndarray]:
        """this module computes the frequency response of the quantized filter using direct probing with sinusoid signals.

        Args:
            freq_vec (np.ndarray): array containing frequency vectors.
            num_periods (int, optional): number of periods to be used in freq response estimation. Defaults to 100.
        """
        
        freq_res_vec = []
        rel_distortion_vec = []
        
        
        for freq in tqdm(freq_vec):
            duration = num_periods/freq
            time_vec = np.arange(0, duration, step=1/self.bd.fs)
            
            # input signal and its quantized version
            sig_in = np.sin(2*np.pi*freq*time_vec)
            
            EPS = 0.0001
            sig_in_Q = (2**(self.bd.B_in-1)/(1+EPS) * sig_in).astype(np.int64)
            
            
            # output
            sig_out, rel_distortion = self.evolve_full(sig_in=sig_in_Q)
            
            # remove the transient part
            sig_out_stable = sig_out[len(sig_out)//6:]
            
            # compute the gain
            gain = (np.sqrt(np.mean(sig_out_stable**2)) * np.sqrt(2))/2**(self.bd.B_in-1)
            
            freq_res_vec.append(gain)
            rel_distortion_vec.append(rel_distortion)
            
        # convert into array
        freq_res_vec = np.asarray(freq_res_vec)
        rel_distortion_vec = np.asarray(rel_distortion_vec)
        
        return freq_res_vec, rel_distortion_vec
    
    def __repr__(self)->str:
        """ string representation of the module """
        
        string = "block-diagram representation of the IIR filter:\n"+\
            f"ADC oversampling factor: {self.adc_oversampling_factor}\n"+\
            "Block-diagram representation info:\n"+\
            str(self.bd)
            

        return string