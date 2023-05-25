# -----------------------------------------------------------
# This function implements the exact block-diagram of the filters
# using bit-shifts and integer multiplication as is done in FPGA.
#
# NOTE: here we have considered a collection of `candidate` bandpass filters that 
# have the potential to be chosen and implemented by the algorithm team.
# Here we make sure that all those filters work properly.
#
#
#
# (C) Karla Burelo, Saeid Haghighatshoar
# email: {karla.burelo, saeid.haghighatshoar}@synsense.ai
#
# last update: 28.08.2022
# -----------------------------------------------------------

# required packages
import numpy as np
from imu_preprocessing.util.type_decorator import type_check

# class containing the parameters of the filter in state-space representation
# This is the block-diagram structure proposed for implementation.
#
# for further details see the proposed filter structure in the report:
# https://paper.dropbox.com/doc/Feasibility-study-for-AFE-with-digital-intensive-approach--BoJoECnIUJvHVe~Htanu~Ee6Ag-b07tQKnwpfDFYrZ5E8seQ

class BlockDiagram:
    B_worst_case    : int   # number of additional bits devoted to storing filter taps such that no over- and under-flow can happen
    B_in            : int   # number of input bits that can be processed with the block diagram
    B_b             : int   # bits needed for scaling b0
    B_a             : int   # total number of bits devoted to storing filter a-taps
    B_af            : int   # bits needed for encoding the fractional parts of taps
    B_wf            : int   # bits needed for fractional part of the filter output
    B_w             : int   # total number of bits devoted to storing the values computed by the AR-filter. It should be equal to `B_in + B_worst_case + B_wf`
    B_out           : int   # total number of bits needed for storing the values computed by the WHOLE filter.
    a1              : int   # integer representation of a1 tap
    a2              : int   # integer representation of a2 tap
    b               : list  # [1, 0 , -1] : special case for normalized Butterworth filters
    scale_out       : int   # surplus scaling due to `b` normalizationsurplus scaling due to `b` normalization. It is always in the range [0.5, 1.0]



class ChipButterworth:
    def __init__(self):
        """
        This class builds the block-diagram version of the filters, which is exactly as it is done in FPGA.
        The propsoed filters are candidates that may be chosen for preprocessing of the IMU data.
        """

        # number of bits needed for quantization
        #self.numQBF_w = 24 # Is this B_A???

        self.numF = 16
        self.bd_list = []

        #========================================#
        # Create block diagram for each filter
        #========================================#
        # Filter 1
        bd_filter_1 = BlockDiagram()
        bd_filter_1.B_worst_case = 9
        bd_filter_1.B_in = 16
        bd_filter_1.B_b = 6
        bd_filter_1.B_a = 17  
        bd_filter_1.B_af = 9
        bd_filter_1.B_wf = 8 #9
        bd_filter_1.B_w = bd_filter_1.B_in + bd_filter_1.B_worst_case + bd_filter_1.B_wf
        bd_filter_1.B_out = bd_filter_1.B_in + bd_filter_1.B_worst_case
        bd_filter_1.a1 = -64700
        bd_filter_1.a2 = 31935
        bd_filter_1.b=[1, 0, -1]
        bd_filter_1.scale_out = 0.8139
        self.bd_list.append(bd_filter_1)

        # Filter 2
        bd_filter_2 = BlockDiagram()
        bd_filter_2.B_worst_case = 5
        bd_filter_2.B_in = 16
        bd_filter_2.B_b = 6
        bd_filter_2.B_a = 17 
        bd_filter_2.B_af = 9
        bd_filter_2.B_wf = 8
        bd_filter_2.B_w = bd_filter_2.B_in + bd_filter_2.B_worst_case + bd_filter_2.B_wf
        bd_filter_2.B_out = bd_filter_2.B_in + bd_filter_2.B_worst_case
        bd_filter_2.a1 = -64458
        bd_filter_2.a2 = 31754
        bd_filter_2.b=[1, 0, -1]
        bd_filter_2.scale_out = 0.9898
        self.bd_list.append(bd_filter_2)
  
        # Filter 3
        bd_filter_3 = BlockDiagram()
        bd_filter_3.B_worst_case = 5
        bd_filter_3.B_in = 16
        bd_filter_3.B_b = 6
        bd_filter_3.B_a = 17
        bd_filter_3.B_af = 9
        bd_filter_3.B_wf = 8
        bd_filter_3.B_w = bd_filter_3.B_in + bd_filter_3.B_worst_case + bd_filter_3.B_wf
        bd_filter_3.B_out = bd_filter_3.B_in + bd_filter_3.B_worst_case
        bd_filter_3.a1 = -64330
        bd_filter_3.a2 = 31754  
        bd_filter_3.b=[1, 0, -1]
        bd_filter_3.scale_out = 0.9898
        self.bd_list.append(bd_filter_3)

        # Filter 4
        bd_filter_4 = BlockDiagram()
        bd_filter_4.B_worst_case = 5
        bd_filter_4.B_in = 16
        bd_filter_4.B_b = 6
        bd_filter_4.B_a = 17
        bd_filter_4.B_af = 9
        bd_filter_4.B_wf = 8
        bd_filter_4.B_w = bd_filter_4.B_in + bd_filter_4.B_worst_case + bd_filter_4.B_wf
        bd_filter_4.B_out = bd_filter_4.B_in + bd_filter_4.B_worst_case
        bd_filter_4.a1 = -64138
        bd_filter_4.a2 = 31754  
        bd_filter_4.b=[1, 0, -1]
        bd_filter_4.scale_out = 0.9898
        self.bd_list.append(bd_filter_4)

        # Filter 5
        bd_filter_5 = BlockDiagram()
        bd_filter_5.B_worst_case = 5
        bd_filter_5.B_in = 16
        bd_filter_5.B_b = 6
        bd_filter_5.B_a = 17
        bd_filter_5.B_af = 9
        bd_filter_5.B_wf = 8
        bd_filter_5.B_w = bd_filter_5.B_in + bd_filter_5.B_worst_case + bd_filter_5.B_wf
        bd_filter_5.B_out = bd_filter_5.B_in + bd_filter_5.B_worst_case
        bd_filter_5.a1 = -63884
        bd_filter_5.a2 = 31754  
        bd_filter_5.b=[1, 0, -1]
        bd_filter_5.scale_out = 0.9898
        self.bd_list.append(bd_filter_5)

        # Filter 6
        bd_filter_6 = BlockDiagram()
        bd_filter_6.B_worst_case = 5
        bd_filter_6.B_in = 16
        bd_filter_6.B_b = 6
        bd_filter_6.B_a = 17
        bd_filter_6.B_af = 9
        bd_filter_6.B_wf = 8
        bd_filter_6.B_w = bd_filter_6.B_in + bd_filter_6.B_worst_case + bd_filter_6.B_wf
        bd_filter_6.B_out = bd_filter_6.B_in + bd_filter_6.B_worst_case
        bd_filter_6.a1 = -63566
        bd_filter_6.a2 = 31754  
        bd_filter_6.b=[1, 0, -1]
        bd_filter_6.scale_out = 0.9898
        self.bd_list.append(bd_filter_6)
        
        # Filter 7
        bd_filter_7 = BlockDiagram()
        bd_filter_7.B_worst_case = 5
        bd_filter_7.B_in = 16
        bd_filter_7.B_b = 6
        bd_filter_7.B_a = 17
        bd_filter_7.B_af = 9
        bd_filter_7.B_wf = 8
        bd_filter_7.B_w = bd_filter_7.B_in + bd_filter_7.B_worst_case + bd_filter_7.B_wf
        bd_filter_7.B_out = bd_filter_7.B_in + bd_filter_7.B_worst_case
        bd_filter_7.a1 = -63185
        bd_filter_7.a2 = 31754  
        bd_filter_7.b=[1, 0, -1]
        bd_filter_7.scale_out = 0.9898
        self.bd_list.append(bd_filter_7)

        # Filter 8
        bd_filter_8 = BlockDiagram()
        bd_filter_8.B_worst_case = 5
        bd_filter_8.B_in = 16
        bd_filter_8.B_b = 6
        bd_filter_8.B_a = 17
        bd_filter_8.B_af = 9
        bd_filter_8.B_wf = 8
        bd_filter_8.B_w = bd_filter_8.B_in + bd_filter_8.B_worst_case + bd_filter_8.B_wf
        bd_filter_8.B_out = bd_filter_8.B_in + bd_filter_8.B_worst_case
        bd_filter_8.a1 = -62743
        bd_filter_8.a2 = 31754  
        bd_filter_8.b=[1, 0, -1]
        bd_filter_8.scale_out = 0.9898
        self.bd_list.append(bd_filter_8)
        
        # Filter 9
        bd_filter_9 = BlockDiagram()
        bd_filter_9.B_worst_case = 5
        bd_filter_9.B_in = 16
        bd_filter_9.B_b = 6
        bd_filter_9.B_a = 17
        bd_filter_9.B_af = 9
        bd_filter_9.B_wf = 8
        bd_filter_9.B_w = bd_filter_9.B_in + bd_filter_9.B_worst_case + bd_filter_9.B_wf
        bd_filter_9.B_out = bd_filter_9.B_in + bd_filter_9.B_worst_case
        bd_filter_9.a1 = -62238
        bd_filter_9.a2 = 31754  
        bd_filter_9.b=[1, 0, -1]
        bd_filter_9.scale_out = 0.9898
        self.bd_list.append(bd_filter_9)

        # Filter 10
        bd_filter_10 = BlockDiagram()
        bd_filter_10.B_worst_case = 5
        bd_filter_10.B_in = 16
        bd_filter_10.B_b = 6
        bd_filter_10.B_a = 17
        bd_filter_10.B_af = 9
        bd_filter_10.B_wf = 8
        bd_filter_10.B_w = bd_filter_10.B_in + bd_filter_10.B_worst_case + bd_filter_10.B_wf
        bd_filter_10.B_out = bd_filter_10.B_in + bd_filter_10.B_worst_case
        bd_filter_10.a1 = -61672
        bd_filter_10.a2 = 31754
        bd_filter_10.b=[1, 0, -1]
        bd_filter_10.scale_out = 0.9898
        self.bd_list.append(bd_filter_10)
        
        # Filter 11
        bd_filter_11 = BlockDiagram()
        bd_filter_11.B_worst_case = 5
        bd_filter_11.B_in = 16
        bd_filter_11.B_b = 6
        bd_filter_11.B_a = 17
        bd_filter_11.B_af = 9
        bd_filter_11.B_wf = 8
        bd_filter_11.B_w = bd_filter_11.B_in + bd_filter_11.B_worst_case + bd_filter_11.B_wf
        bd_filter_11.B_out = bd_filter_11.B_in + bd_filter_11.B_worst_case
        bd_filter_11.a1 = -61045
        bd_filter_11.a2 = 31754  
        bd_filter_11.b=[1, 0, -1]
        bd_filter_11.scale_out = 0.9898
        self.bd_list.append(bd_filter_11)        

        # Filter 12
        bd_filter_12 = BlockDiagram()
        bd_filter_12.B_worst_case = 5
        bd_filter_12.B_in = 16
        bd_filter_12.B_b = 6
        bd_filter_12.B_a = 17
        bd_filter_12.B_af = 9
        bd_filter_12.B_wf = 8
        bd_filter_12.B_w = bd_filter_12.B_in + bd_filter_12.B_worst_case + bd_filter_12.B_wf
        bd_filter_12.B_out = bd_filter_12.B_in + bd_filter_12.B_worst_case
        bd_filter_12.a1 = -60357
        bd_filter_12.a2 = 31754 
        bd_filter_12.b=[1, 0, -1]
        bd_filter_12.scale_out = 0.9898
        self.bd_list.append(bd_filter_12)

        # Filter 13
        bd_filter_13 = BlockDiagram()
        bd_filter_13.B_worst_case = 5
        bd_filter_13.B_in = 16
        bd_filter_13.B_b = 6
        bd_filter_13.B_a = 17
        bd_filter_13.B_af = 9
        bd_filter_13.B_wf = 8
        bd_filter_13.B_w = bd_filter_13.B_in + bd_filter_13.B_worst_case + bd_filter_13.B_wf
        bd_filter_13.B_out = bd_filter_13.B_in + bd_filter_13.B_worst_case
        bd_filter_13.a1 = -59611
        bd_filter_13.a2 = 31754  
        bd_filter_13.b=[1, 0, -1]
        bd_filter_13.scale_out = 0.9898
        self.bd_list.append(bd_filter_13)

        # Filter 14
        bd_filter_14 = BlockDiagram()
        bd_filter_14.B_worst_case = 5
        bd_filter_14.B_in = 16
        bd_filter_14.B_b = 6
        bd_filter_14.B_a = 17
        bd_filter_14.B_af = 9
        bd_filter_14.B_wf = 8
        bd_filter_14.B_w = bd_filter_14.B_in + bd_filter_14.B_worst_case + bd_filter_14.B_wf
        bd_filter_14.B_out = bd_filter_14.B_in + bd_filter_14.B_worst_case
        bd_filter_14.a1 = -58805
        bd_filter_14.a2 = 31754  
        bd_filter_14.b=[1, 0, -1]
        bd_filter_14.scale_out = 0.9898
        self.bd_list.append(bd_filter_14)

        # Filter 15
        bd_filter_15 = BlockDiagram()
        bd_filter_15.B_worst_case = 5
        bd_filter_15.B_in = 16
        bd_filter_15.B_b = 6
        bd_filter_15.B_a = 17
        bd_filter_15.B_af = 9
        bd_filter_15.B_wf = 8
        bd_filter_15.B_w = bd_filter_15.B_in + bd_filter_15.B_worst_case + bd_filter_15.B_wf
        bd_filter_15.B_out = bd_filter_15.B_in + bd_filter_15.B_worst_case
        bd_filter_15.a1 = -57941
        bd_filter_15.a2 = 31754  
        bd_filter_15.b=[1, 0, -1]
        bd_filter_15.scale_out = 0.9898
        self.bd_list.append(bd_filter_15)

        # Filter 16
        bd_filter_16 = BlockDiagram()
        bd_filter_16.B_worst_case = 5
        bd_filter_16.B_in = 16
        bd_filter_16.B_b = 6
        bd_filter_16.B_a = 17
        bd_filter_16.B_af = 9
        bd_filter_16.B_wf = 8
        bd_filter_16.B_w = bd_filter_16.B_in + bd_filter_16.B_worst_case + bd_filter_16.B_wf
        bd_filter_16.B_out = bd_filter_16.B_in + bd_filter_16.B_worst_case
        bd_filter_16.a1 = -57020
        bd_filter_16.a2 = 31754  
        bd_filter_16.b=[1, 0, -1]
        bd_filter_16.scale_out = 0.9898
        self.bd_list.append(bd_filter_16)    


    @type_check
    def _filter_AR(self, bd: BlockDiagram, sig_in: np.ndarray):
        """
        This function computes the AR part of the filter in the block-diagram with the given parameters.
        
        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): the quantized input signal in python-object integer format.
            
        Raises:
            OverflowError: if any overflow happens during the filter computation.
            
        Returns:
            np.ndarray: the output signal of the AR filter.
        """
        
        # check that the input is within the valid range of block-diagram

        if np.max(np.abs(sig_in)) >= 2**(bd.B_in - 1):
            raise ValueError(f'The input signal values can be in the range [-2^{bd.B_in-1}, +2^{bd.B_in-1}]!')

        
        output= []
        
        # w[n], w[n-1], w[n-2]
        w = [0,0,0]
        
        for sig in sig_in:
            # computation after the clock
            #w_new = sig * 2**bd.B_wf + np.floor( (-bd.a2*w[2] - bd.a1 * w[1])/ 2**bd.B_af )
            #w_new = np.floor(w_new / 2**bd.B_b)
            
            w_new = (sig << bd.B_wf) + ( (-bd.a2 * w[2] - bd.a1 * w[1]) >> bd.B_af )
            w_new = w_new >> bd.B_b
            
            w[0] = w_new
            
            # register shift at the rising edge of the clock
            w[1], w[2] = w[0], w[1]

            output.append(w[0])
        
            # check the overflow: here we have the integer version

        if np.max(np.abs(output)) >= 2**(bd.B_w - 1):
            raise ValueError(f'output signal is beyond the valid output range of AR branch [-2^{bd.B_w-1}, +2^{bd.B_w-1}]!')

        # convert into numpy
        return np.asarray(output, dtype=object)
    
    
    @type_check
    def _filter_MA(self, bd : BlockDiagram, sig_in: np.ndarray):
        """
        This function computes the MA part of the filter in the block-diagram representation.
        
        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): input signal (in this case output of AR part) of datatype `pyton.object`.
            
        Raises:
            OverflowError: if any overflow happens during the filter computation.
            
        Returns:
            np.ndarray: quantized filtered output signal.
        """
                
        # check dimension
        if sig_in.ndim > 1:
            raise ValueError('input signal should be 1-dim.')
    
        
        sig_out = bd.b[0] * sig_in 
        sig_out[2:] = sig_out[2:] + bd.b[2] * sig_in[:-2]

        # apply the last B_wf bitshift to get rid of additional scaling needed to avoid dead-zone in the AR part
        #sig_out = np.floor(sig_out/2**bd.B_wf)
        sig_out = sig_out >> bd.B_wf

        # check the validity of the computed output
        if np.max(np.abs(sig_out)) >= 2**(bd.B_out - 1):
            raise OverflowError(f'overflow or underflow: computed filter output is beyond the valid range [-2^{bd.B_out-1}, +2^{bd.B_out-1}]!')
        
        
        return sig_out
    
    
    @type_check
    def _filter(self, bd: BlockDiagram, sig_in: np.ndarray):
        """
        This filter combines the filtering done in the AR and MA part of the block-diagram representation.
        
        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): quantized input signal of python.object integer type.
            
        Raises:
            OverflowError: if any overflow happens during the filter computation.
            
        Returns:
            np.nadarray: quantized filtered output signal.
        """
        
        # AR branch
        w = self._filter_AR(bd, sig_in)
        
        # followed by MA branch
        out = self._filter_MA(bd, w)

        return out

    @type_check
    def evolve(self, sig_in: np.ndarray, scale_out : bool = False):
        """
        This function computes the output of all filters for an input signal.
        
        Args:
            sig_in (np.ndarray): the quantized input signal of datatype python.object integer.
            scale_out (bool, optional)  : add the surplus scaling due to `b` normalization. Defaults to True.
        """
        
        if scale_out:
            raise ValueError(
                'In this version, we work with just integer version of the filters.'+\
                'The surplus scaling in the range [0.5, 1.0] can be applied later.'
            )
        
        output = []
        
        for filt_num in range(self.numF):
            # check the parameters as block diagram
            bd = self.bd_list[filt_num]
            
            # apply the filter to the input signal
            sig_out = self._filter(bd, sig_in)
            
            # apply output scaling if requested
            if scale_out:
                sig_out = bd.scale_out * sig_out

            output.append(sig_out)

        # convert into numpy
        output = np.asarray(output, dtype=object)
        
        return output
    
    
    # add the call version for further convenience
    def __call__(self, *args, **kwargs):
        """
        This function simply calls the `evolve()` function and is added for further convenience.
        """
        return self.evolve(*args, **kwargs)
        
        
    
    # utility functions
    def print_parameters(self):
        print('*'*60)
        for filt_num in range(self.numF):
            bd = self.bd_list[filt_num]
            print(f'filter {filt_num}:')
            print(f'B_worst_case (worst case amplitude)= {bd.B_worst_case}')
            print(f'B_b = {bd.B_b}')
            print(f'B_in = {bd.B_in}')
            print(f'B_a = {bd.B_a}')
            print(f'B_w = {bd.B_w}')
            print(f'B_out = {bd.B_out}')
            print(f'B_wf = {bd.B_wf}')
            print(f'B_af = {bd.B_af}')
            print(f'output surplus scale = {bd.scale_out: 0.4f}')

            print('\n')

