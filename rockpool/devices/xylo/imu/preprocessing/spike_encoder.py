# -----------------------------------------------------------
# This module implements the spike encoding for the signal coming out of filters or IMU sensor directly.
# 
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 28.08.2022
# -----------------------------------------------------------

# required packages
import numpy as np
from imu_preprocessing.util.type_decorator import type_check

class SpikeEncoder:
    @type_check
    def evolve(self, sig_in: np.ndarray):
        """
        this modules takes the multi-channel input signal of dimension `num-channels x T` and concerts it into spikes.
        The number of channels is preserved.

        Args:
            sig_in (np.ndarray): input multi-channel signal of dimnsion `num-channels x T`.
        """
        pass


class ScaleSpikeEncoder(SpikeEncoder):
    def __init__(self, num_scale_bits: int, num_out_bits: int):
        """this module does spike encoding as follows:
                (i)     compute the absolute value of the input sigal (full-wave rectification)
                (ii)    down-scales the input signal by right-bit-shift by `num_scale_bit` (e.g. multiplying with 1/2^num_scale_bits)
                (iii)   truncate the output so that it can fit within `num_out_bits`    

        Args:
            num_scale_bits (int): number of right-bit-shifs needed for down-scaling the iinput signal.
            num_out_bits (int): number of bits devoted to storing the output spike encoding.
        """
        self.num_scale_bits = num_scale_bits
        self.num_out_bits = num_out_bits
        
    def evolve(self, sig_in: np.ndarray):
        # simple type check
        super().evolve(sig_in)
        
        # compute the absolute value of the input signal
        sig_in = np.abs(sig_in)
        
        # scale the signal
        sig_in = sig_in >> self.num_scale_bits
        
        # truncate the signal
        threshold = (1 << self.num_out_bits) - 1
        
        sig_in[sig_in > threshold] = threshold
        
        return sig_in
    
    def __str__(self):
        string = 'scale-and-quantize spike encpder:\n'+\
            f'number of right bit-shifts for down-scaling: {self.num_scale_bits}\n'+\
            f'number of bits used for spike encoding (clipped if less than the max amplitude): {self.num_out_bits}'
        return string
    
    
class IAFSpikeEncoder(SpikeEncoder):
    def __init__(self, iaf_threshold: int):
        self.iaf_threshold = iaf_threshold
    
    def evolve(self, sig_in: np.ndarray):
        # do the type check
        super().evolve(sig_in)
        
        # check the number of channels
        if len(sig_in.shape) == 1:
            sig_in = sig_in.reshape(1, -1)
        
        # compute the absolute value of the signal
        sig_in = np.abs(sig_in)
        
        # compute the cumsum along the time axis
        sig_in = np.cumsum(sig_in, axis=1)
        
        # compute the number of spikes produced so far
        num_spikes = sig_in//self.iaf_threshold
        
        # add a zero column to make sure that the dimensions match
        num_spikes = np.hstack([np.zeros( (num_spikes.shape[0], 1), dtype=object), num_spikes])
        
        # compute the spikes
        spikes = np.diff(num_spikes, axis=1)
        
        # if there are more than one spikes, truncate it to 1
        spikes[spikes > 1] = 1
        
        return spikes
    
    
    # add the call version for further convenience
    def __call__(self, *args, **kwargs):
        """
        This function simply calls the `evolve()` function and is added for further convenience.
        """
        return self.evolve(*args, **kwargs)
        
        
            
    def __str__(self):
        string = 'Integratep-and-Fire (IAF) spike encoder:\n'+\
            f'IAF threshold: {self.iaf_threshold}'
        return string
    
