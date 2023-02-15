# -----------------------------------------------------------
# This module implements the digital filterbank in Xylo-A3 chip.
# This is the first version of Xylo chip in which the analog filters
# have been replaced with the digital ones.
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 19.01.2023
# -----------------------------------------------------------

# FIXME:      
# (i) at the moment in MA part of the filter I dot get rid of B_wf bits that was added at the start of the filter to get rid of dead zone.
# It would be better to keep these B_wf bits to have a better bit resolution especially for the first filter.
# I need to finalize this with Sunil.


# FIXME: I need to take into account 4 additional bits in PDM ADC.

# FIXME: what should we do with removing right bitshift in the filters due to B_wf left bitshift we did to avoid dead zone.
#        I need to add this to all the filters. And also adjust the filters parameters.

# FIXME: In the jax version, I dod not do any over- and under-flow check according to the filter params designed according to the worst case analysis.
#        I need to add this although it would be useless because filters are implemented in the float32 version.


# required packages
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from functools import wraps

from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter, ParameterBase

from functools import partial

from typing import Union, Tuple, List, Dict
P_int = Union[int, ParameterBase]
P_float = Union[float, ParameterBase]
P_array = Union[np.array, ParameterBase]





# list of modules exported
__all__ = ["ChipButterworth", "NUM_FILETRS"]


# number of filters in Xylo-A3
NUM_FILETRS = 16


# a simple decorator to make sure that the input to the filters has type `int` or `np.int64`
# this is needed to avoid any over- and under-flow in filter implementation.

def type_check(func):
    """
    this function is a type-check decorator for make sure that all the input data are of type `int` or `np.int64`.
    This assures that the hardware and software will behave the same for the register sizes we have in mind.

    Args:
        func (Callable): the function to be decorated.
    """
    
    valid_types = [int, np.int64]
    
    # function for checking the type
    def verify(input):
        if isinstance(input, list):
            if len(input) == 0:
                return
            for el in input:
                type_check(el)
        
        if isinstance(input, np.ndarray):
            if (input.dtype not in valid_types) or (type(input.ravel()[0]) not in valid_types):
                raise ValueError(
                    f'The elements of the following variable are not of type {valid_types}. This may cause mismatch between hardware and python implementation.\n'+\
                    f'problem with the follpowing variable:\n{input}\n'+\
                    f'To solve the problem make sure that all the arrays have `dtype in {valid_types}`.'
                )

        return
    
    # original function implementation
    @wraps(func)
    def inner_func(*args, **kwargs):
        # verification phase
        for arg in args:
            verify(arg)
        
        for key in kwargs:
            verify(kwargs[key])
        
        # main functionality
        return func(*args, **kwargs)
    
    # return an instance of the inner function
    return inner_func


# class for capturing over-flow and under-flow in computation
class OverflowError(Exception):
    pass



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



class ChipButterworth(Module):
    def __init__(self):
        """
        This class builds the block-diagram version of the filters, which is exactly as it is done in FPGA.
        The propsoed filters are candidates that may be chosen for preprocessing of the IMU data.
        """
        super().__init__()

        # number of bits needed for quantization
        #self.numQBF_w = 24 # Is this B_A???

        self.numF : P_int = Parameter(NUM_FILETRS)
        self.bd_list = []

        #========================================#
        # Create block diagram for each filter
        #========================================#
        # Filter 1
        bd_filter_1 = BlockDiagram()
        bd_filter_1.B_worst_case = 7
        bd_filter_1.B_in = 14
        bd_filter_1.B_b = 8
        bd_filter_1.B_a = 16  
        bd_filter_1.B_af = 6
        bd_filter_1.B_wf = 8
        bd_filter_1.B_w = bd_filter_1.B_in + bd_filter_1.B_worst_case + bd_filter_1.B_wf
        bd_filter_1.B_out = bd_filter_1.B_in + bd_filter_1.B_worst_case + bd_filter_1.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_1.a1 = -32694
        bd_filter_1.a2 = 16313
        bd_filter_1.b=[1, 0, -1]
        bd_filter_1.scale_out = 0.5573
        self.bd_list.append(bd_filter_1)

        # Filter 2
        bd_filter_2 = BlockDiagram()
        bd_filter_2.B_worst_case = 6
        bd_filter_2.B_in = 14
        bd_filter_2.B_b = 8
        bd_filter_2.B_a = 16
        bd_filter_2.B_af = 6
        bd_filter_2.B_wf = 8
        bd_filter_2.B_w = bd_filter_2.B_in + bd_filter_2.B_worst_case + bd_filter_2.B_wf
        bd_filter_2.B_out = bd_filter_2.B_in + bd_filter_2.B_worst_case + bd_filter_2.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_2.a1 = -32663
        bd_filter_2.a2 = 16284
        bd_filter_2.b=[1, 0, -1]
        bd_filter_2.scale_out = 0.7810
        self.bd_list.append(bd_filter_2)
  
        # Filter 3
        bd_filter_3 = BlockDiagram()
        bd_filter_3.B_worst_case = 6
        bd_filter_3.B_in = 14
        bd_filter_3.B_b = 7
        bd_filter_3.B_a = 16
        bd_filter_3.B_af = 7
        bd_filter_3.B_wf = 8
        bd_filter_3.B_w = bd_filter_3.B_in + bd_filter_3.B_worst_case + bd_filter_3.B_wf
        bd_filter_3.B_out = bd_filter_3.B_in + bd_filter_3.B_worst_case + bd_filter_3.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_3.a1 = -32617
        bd_filter_3.a2 = 16244  
        bd_filter_3.b=[1, 0, -1]
        bd_filter_3.scale_out = 0.5470
        self.bd_list.append(bd_filter_3)

        # Filter 4
        bd_filter_4 = BlockDiagram()
        bd_filter_4.B_worst_case = 5
        bd_filter_4.B_in = 14
        bd_filter_4.B_b = 7
        bd_filter_4.B_a = 16
        bd_filter_4.B_af = 7
        bd_filter_4.B_wf = 8
        bd_filter_4.B_w = bd_filter_4.B_in + bd_filter_4.B_worst_case + bd_filter_4.B_wf
        bd_filter_4.B_out = bd_filter_4.B_in + bd_filter_4.B_worst_case + bd_filter_4.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_4.a1 = -32551
        bd_filter_4.a2 = 16188  
        bd_filter_4.b=[1, 0, -1]
        bd_filter_4.scale_out = 0.7660
        self.bd_list.append(bd_filter_4)

        # Filter 5
        bd_filter_5 = BlockDiagram()
        bd_filter_5.B_worst_case = 5
        bd_filter_5.B_in = 14
        bd_filter_5.B_b = 6
        bd_filter_5.B_a = 16
        bd_filter_5.B_af = 8
        bd_filter_5.B_wf = 8
        bd_filter_5.B_w = bd_filter_5.B_in + bd_filter_5.B_worst_case + bd_filter_5.B_wf
        bd_filter_5.B_out = bd_filter_5.B_in + bd_filter_5.B_worst_case + bd_filter_5.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_5.a1 = -32453
        bd_filter_5.a2 = 16110  
        bd_filter_5.b=[1, 0, -1]
        bd_filter_5.scale_out = 0.5359
        self.bd_list.append(bd_filter_5)

        # Filter 6
        bd_filter_6 = BlockDiagram()
        bd_filter_6.B_worst_case = 4
        bd_filter_6.B_in = 14
        bd_filter_6.B_b = 6
        bd_filter_6.B_a = 16
        bd_filter_6.B_af = 8
        bd_filter_6.B_wf = 8
        bd_filter_6.B_w = bd_filter_6.B_in + bd_filter_6.B_worst_case + bd_filter_6.B_wf
        bd_filter_6.B_out = bd_filter_6.B_in + bd_filter_6.B_worst_case + bd_filter_6.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_6.a1 = -32305
        bd_filter_6.a2 = 16000  
        bd_filter_6.b=[1, 0, -1]
        bd_filter_6.scale_out = 0.7492
        self.bd_list.append(bd_filter_6)
        
        # Filter 7
        bd_filter_7 = BlockDiagram()
        bd_filter_7.B_worst_case = 4
        bd_filter_7.B_in = 14
        bd_filter_7.B_b = 5
        bd_filter_7.B_a = 16
        bd_filter_7.B_af = 9
        bd_filter_7.B_wf = 8
        bd_filter_7.B_w = bd_filter_7.B_in + bd_filter_7.B_worst_case + bd_filter_7.B_wf
        bd_filter_7.B_out = bd_filter_7.B_in + bd_filter_7.B_worst_case + bd_filter_7.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_7.a1 = -32077
        bd_filter_7.a2 = 15848  
        bd_filter_7.b=[1, 0, -1]
        bd_filter_7.scale_out = 0.5230
        self.bd_list.append(bd_filter_7)

        # Filter 8
        bd_filter_8 = BlockDiagram()
        bd_filter_8.B_worst_case = 3
        bd_filter_8.B_in = 14
        bd_filter_8.B_b = 5
        bd_filter_8.B_a = 16
        bd_filter_8.B_af = 9
        bd_filter_8.B_wf = 8
        bd_filter_8.B_w = bd_filter_8.B_in + bd_filter_8.B_worst_case + bd_filter_8.B_wf
        bd_filter_8.B_out = bd_filter_8.B_in + bd_filter_8.B_worst_case + bd_filter_8.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_8.a1 = -31718
        bd_filter_8.a2 = 15638  
        bd_filter_8.b=[1, 0, -1]
        bd_filter_8.scale_out = 0.7288
        self.bd_list.append(bd_filter_8)
        
        # Filter 9
        bd_filter_9 = BlockDiagram()
        bd_filter_9.B_worst_case = 3
        bd_filter_9.B_in = 14
        bd_filter_9.B_b = 4
        bd_filter_9.B_a = 16
        bd_filter_9.B_af = 10
        bd_filter_9.B_wf = 8
        bd_filter_9.B_w = bd_filter_9.B_in + bd_filter_9.B_worst_case + bd_filter_9.B_wf
        bd_filter_9.B_out = bd_filter_9.B_in + bd_filter_9.B_worst_case + bd_filter_9.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_9.a1 = -31139
        bd_filter_9.a2 = 15347  
        bd_filter_9.b=[1, 0, -1]
        bd_filter_9.scale_out = 0.5065
        self.bd_list.append(bd_filter_9)

        # Filter 10
        bd_filter_10 = BlockDiagram()
        bd_filter_10.B_worst_case = 2
        bd_filter_10.B_in = 14
        bd_filter_10.B_b = 4
        bd_filter_10.B_a = 16
        bd_filter_10.B_af = 10
        bd_filter_10.B_wf = 8
        bd_filter_10.B_w = bd_filter_10.B_in + bd_filter_10.B_worst_case + bd_filter_10.B_wf
        bd_filter_10.B_out = bd_filter_10.B_in + bd_filter_10.B_worst_case + bd_filter_10.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_10.a1 = -30185
        bd_filter_10.a2 = 14947
        bd_filter_10.b=[1, 0, -1]
        bd_filter_10.scale_out = 0.7018
        self.bd_list.append(bd_filter_10)
        
        # Filter 11
        bd_filter_11 = BlockDiagram()
        bd_filter_11.B_worst_case = 2
        bd_filter_11.B_in = 14
        bd_filter_11.B_b = 4
        bd_filter_11.B_a = 16
        bd_filter_11.B_af = 10
        bd_filter_11.B_wf = 8
        bd_filter_11.B_w = bd_filter_11.B_in + bd_filter_11.B_worst_case + bd_filter_11.B_wf
        bd_filter_11.B_out = bd_filter_11.B_in + bd_filter_11.B_worst_case + bd_filter_11.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_11.a1 = -28582
        bd_filter_11.a2 = 14402  
        bd_filter_11.b=[1, 0, -1]
        bd_filter_11.scale_out = 0.9679
        self.bd_list.append(bd_filter_11)        

        # Filter 12
        bd_filter_12 = BlockDiagram()
        bd_filter_12.B_worst_case = 2
        bd_filter_12.B_in = 14
        bd_filter_12.B_b = 3
        bd_filter_12.B_a = 16
        bd_filter_12.B_af = 11
        bd_filter_12.B_wf = 8
        bd_filter_12.B_w = bd_filter_12.B_in + bd_filter_12.B_worst_case + bd_filter_12.B_wf
        bd_filter_12.B_out = bd_filter_12.B_in + bd_filter_12.B_worst_case + bd_filter_12.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_12.a1 = -25862
        bd_filter_12.a2 = 13666 
        bd_filter_12.b=[1, 0, -1]
        bd_filter_12.scale_out = 0.6635
        self.bd_list.append(bd_filter_12)

        # Filter 13
        bd_filter_13 = BlockDiagram()
        bd_filter_13.B_worst_case = 2
        bd_filter_13.B_in = 14
        bd_filter_13.B_b = 3
        bd_filter_13.B_a = 16
        bd_filter_13.B_af = 11
        bd_filter_13.B_wf = 8
        bd_filter_13.B_w = bd_filter_13.B_in + bd_filter_13.B_worst_case + bd_filter_13.B_wf
        bd_filter_13.B_out = bd_filter_13.B_in + bd_filter_13.B_worst_case + bd_filter_13.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_13.a1 = -21262
        bd_filter_13.a2 = 12687  
        bd_filter_13.b=[1, 0, -1]
        bd_filter_13.scale_out = 0.9026
        self.bd_list.append(bd_filter_13)

        # Filter 14
        bd_filter_14 = BlockDiagram()
        bd_filter_14.B_worst_case = 2
        bd_filter_14.B_in = 14
        bd_filter_14.B_b = 2
        bd_filter_14.B_a = 16
        bd_filter_14.B_af = 13
        bd_filter_14.B_wf = 8
        bd_filter_14.B_w = bd_filter_14.B_in + bd_filter_14.B_worst_case + bd_filter_14.B_wf
        bd_filter_14.B_out = bd_filter_14.B_in + bd_filter_14.B_worst_case + bd_filter_14.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_14.a1 = -27375
        bd_filter_14.a2 = 22803  
        bd_filter_14.b=[1, 0, -1]
        bd_filter_14.scale_out = 0.6082
        self.bd_list.append(bd_filter_14)

        # Filter 15
        bd_filter_15 = BlockDiagram()
        bd_filter_15.B_worst_case = 2
        bd_filter_15.B_in = 14
        bd_filter_15.B_b = 2
        bd_filter_15.B_a = 16
        bd_filter_15.B_af = 13
        bd_filter_15.B_wf = 8
        bd_filter_15.B_w = bd_filter_15.B_in + bd_filter_15.B_worst_case + bd_filter_15.B_wf
        bd_filter_15.B_out = bd_filter_15.B_in + bd_filter_15.B_worst_case + bd_filter_15.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_15.a1 = -4180
        bd_filter_15.a2 = 19488  
        bd_filter_15.b=[1, 0, -1]
        bd_filter_15.scale_out = 0.8105
        self.bd_list.append(bd_filter_15)

        # Filter 16
        bd_filter_16 = BlockDiagram()
        bd_filter_16.B_worst_case = 2
        bd_filter_16.B_in = 14
        bd_filter_16.B_b = 1
        bd_filter_16.B_a = 16
        bd_filter_16.B_af = 14
        bd_filter_16.B_wf = 8
        bd_filter_16.B_w = bd_filter_16.B_in + bd_filter_16.B_worst_case + bd_filter_16.B_wf
        bd_filter_16.B_out = bd_filter_16.B_in + bd_filter_16.B_worst_case + bd_filter_16.B_wf  # NOTE: these additional 8 bits were added in the final version.
        bd_filter_16.a1 = 25566
        bd_filter_16.a2 = 15280  
        bd_filter_16.b=[1, 0, -1]
        bd_filter_16.scale_out = 0.5337
        self.bd_list.append(bd_filter_16)    
        
        
        self.bd_list : P_array = Parameter(self.bd_list, shape=(NUM_FILETRS,))
        

    @type_check
    def _filter_AR(self, bd: BlockDiagram, sig_in: np.ndarray)->np.ndarray:
        """
        This function computes the AR (auto-regressive) part of the filter in the block-diagram with the given parameters.
        
        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): the input signal.
            
        Raises:
            OverflowError: if any overflow or underflow happens during the filter computation.
            
        Returns:
            np.ndarray: the output signal of the AR filter.
        """
        
        # check that the input is within the valid range of block-diagram

        if np.max(np.abs(sig_in)) >= 2**(bd.B_in - 1):
            raise OverflowError(f'The input signal values can be in the range [-2^{bd.B_in-1}, +2^{bd.B_in-1}]!')

        
        output= []
        
        # w[n], w[n-1], w[n-2]
        w = [0,0,0]
        
        for sig in sig_in:
            # NOTE: Here we assume that AR part uses the latched/gated version of the filter at its ouput.
            # we have used the same convention for jax version
            output.append(w[0])

            # computation after the clock
            w_new = (sig << bd.B_wf) + ( (-bd.a2 * w[2] - bd.a1 * w[1]) >> bd.B_af )
            w_new = w_new >> bd.B_b
            
            w[0] = w_new
            
            # register shift at the rising edge of the clock
            w[1], w[2] = w[0], w[1]

            
        # convert into numpy array
        output = np.asarray(output, dtype=np.int64)
        
        # check the overflow: here we have the integer version
        if np.max(np.abs(output)) >= 2**(bd.B_w - 1):
            raise OverflowError(f'output signal is beyond the valid output range of AR branch [-2^{bd.B_w-1}, +2^{bd.B_w-1}]!')

        return output
    
    
    @type_check
    def _filter_MA(self, bd : BlockDiagram, sig_in: np.ndarray)->np.ndarray:
        """
        This function computes the MA (moving-average) part of the filter in the block-diagram representation.
        
        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): integer-valued input signal.
            
        Raises:
            OverflowError: if any overflow happens during the filter computation.
            
        Returns:
            np.ndarray: the output of the MA part of the filter.
        """
                
        # check dimension
        if sig_in.ndim > 1:
            raise ValueError('input signal should be 1-dim.')
    
        
        sig_out = bd.b[0] * sig_in 
        sig_out[2:] = sig_out[2:] + bd.b[2] * sig_in[:-2]

        # apply the last B_wf bitshift to get rid of additional scaling needed to avoid dead-zone in the AR part
        # NOTE: we need to bitshift by B_wf (=8 bits here for all the filters) to get rid of the left-bitshift that we had added to avoid dead-zone in the filter
        #       After discussing with the hardware team we decided to not do this to have a better resolution at the output.
        #       This implies that the low-pass filter, thresholds, etc. in the DN and spike generation part also needs to be increased by 8 bits.
        #
        # sig_out = sig_out >> bd.B_wf

        # check the validity of the computed output
        if np.max(np.abs(sig_out)) >= 2**(bd.B_out - 1):
            raise OverflowError(f'overflow or underflow: computed filter output is beyond the valid range [-2^{bd.B_out-1}, +2^{bd.B_out-1}]!')
        
        return sig_out
    
    
    @type_check
    def _filter(self, bd: BlockDiagram, sig_in: np.ndarray)->np.ndarray:
        """
        This filter combines the filtering done in the AR and MA part of the block-diagram representation.
        
        Args:
            bd (BlockDiagram): block diagram representation of the filter.
            sig_in (np.ndarray): integer-valued input signal.
            
        Raises:
            OverflowError: if any overflow happens during the filter computation.
            
        Returns:
            np.nadarray: the output of the whole filter.
        """
        
        # AR branch
        w = self._filter_AR(bd, sig_in)
        
        # followed by MA branch
        out = self._filter_MA(bd, w)

        return out
    
    def _filter_iter(self, args_tuple):
        """ 
        this is the the same `_filter` function with the difference that it maps on a single argument which contains a tuple of other arguments.
        """
        return self._filter(*args_tuple)

    @type_check
    def evolve(self, sig_in: np.ndarray, num_workers:int=4, scale_out:bool = False, python_version:bool=False, record:bool=False):
        """
        This function computes the output of all filters for an input signal.
        
        Args:
            sig_in (np.ndarray): integer-valued input signal.
            num_workers (int): number of independent processes (noth threads) used to compute the filte faster. Defaults to 1 (4 workers).
            record (bool, optional): record the state of the filter (AR part). Defaults to False.
            scale_out (bool, optional): add the surplus scaling due to `b` normalization.
            python_version (bool, optional): force computing the filters with python version. Defaults to False: use jax if available.
            NOTE: this is due to the fact that after integer-quantization, one may still need to scale
            the filter output by some value in the range [0.5, 1.0] to obtain the final output. Defaults to False.
        """
        
        if scale_out:
            scale_out_list = np.asarray([bd.scale_out for bd in self.bd_list])
        else:
            scale_out_list = np.ones(self.numF)
            
        # check if jax version is available
        if JAX_Filter and not python_version:
            #===========================================================================
            #                            Jax Version
            #===========================================================================
                        
            Bwf_list = np.asarray([bd.B_wf for bd in self.bd_list])
            Bb_list = np.asarray([bd.B_b for bd in self.bd_list])
            Baf_list = np.asarray([bd.B_af for bd in self.bd_list])
            a_list = np.asarray([[bd.a1, bd.a2] for bd in self.bd_list])
            b_list = np.asarray([bd.b for bd in self.bd_list])
            
            
            sig_out, recording = jax_filter(
                sig_in=sig_in,
                Bwf_list=Bwf_list,
                Bb_list=Bb_list,
                Baf_list=Baf_list,
                a_list=a_list,
                b_list=b_list,
                scale_out_list=scale_out_list,
                record=record,
            )
            
        else:
            #===========================================================================
            #                            Python Version
            #===========================================================================
            
            # to avoid issue with multiprocessing when num_workers = 1 => switch to single-cpu version
            
            # create an iterator of the arguments
            args_iter = ((bd, sig_in) for bd in self.bd_list)
            
            if num_workers > 1:
                #===========================
                #     Multi-CPU Version
                #===========================
                # use the multi-processing version
                # NOTE: this is unstable sometimes: processes start but they do not return any output
        
    
                # create an executer
                with ProcessPoolExecutor(max_workers=num_workers) as PPE:
                    # obtain the results
                    results = PPE.map(self._filter_iter, args_iter)
                    
                # convert the results into numpy array: a transpose is needed to be compatible with rockpool T x C format where time is the first index.
                sig_out = np.asarray([result for result in results], dtype=np.int64).T
                
            else:
                #===========================
                #     Single-CPU Version
                #===========================
                sig_out = []
                
                for args_tuple in args_iter:
                    output = self._filter_iter(args_tuple=args_tuple)
                    sig_out.append(output)
                    
                # convert the results into numpy array: a transpose is needed to be compatible with rockpool T x C format where time is the first index.
                sig_out = np.asarray(sig_out, dtype=np.int64).T
            
            # add the surplus scaling factor
            if scale_out:
                sig_out = np.einsum("ij, j -> ij", sig_out, scale_out_list)
            
            # in the python version state is empty: for performance reasons
            recording = {}
            
        
        
        return sig_out, recording

    
    # add the call version for further convenience
    def __call__(self, *args, **kwargs):
        """
        This function simply calls the `evolve()` function and is added for further convenience.
        """
        return self.evolve(*args, **kwargs)
        
        
    
    # utility functions
    def __repr__(self)->str:
        string = '*'*60
        for filt_num in range(self.numF):
            bd = self.bd_list[filt_num]
            string += f'filter {filt_num}:\n'+\
                    f'B_worst_case (worst case amplitude)= {bd.B_worst_case}\n'+\
                    f'B_b = {bd.B_b}\n'+\
                    f'B_in = {bd.B_in}\n'+\
                    f'B_a = {bd.B_a}\n'+\
                    f'B_w = {bd.B_w}\n'+\
                    f'B_out = {bd.B_out}\n'+\
                    f'B_wf = {bd.B_wf}\n'+\
                    f'B_af = {bd.B_af}\n'+\
                    f'output surplus scale = {bd.scale_out: 0.4f}\n\n'
                    
        return string



# implement the jax version as well
try:
    import jax
    import jax.numpy as jnp 
    
    # only jax.float32 version implemented as jax.int32 will not work in the filters due to their large number of bits.
    def jax_filter(sig_in: np.ndarray, Bwf_list: np.ndarray, Bb_list: np.ndarray, Baf_list: np.ndarray, a_list: np.ndarray, b_list: np.ndarray, scale_out_list: np.ndarray, record:bool=False)->Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """this function implements the filters in jax with float32 format.
        
        NOTE: To have the exact chip version, one need to implement the filters in integer format. 
        But with jax int32 there might be high chance for over-flow and under-flow. So we decided to use only float32 version.
        
        The problem with flost32 is that there might be a slight difference between software and hardware version.
        
        

        Args:
            sig_in (np.ndarray): quantized input signal.
            Bwf_list (np.ndarray): an array containing the list of B_wf (bitshifts used to avoid dead zone).
            Bb_list (np.ndarary): an array containing the list of Bb values used for proper scaling and quantizing the filter coefficients.
            Baf_list (np.ndarray): an array containing the bistshift the feedback part of the AR part of the filter.
            a_list (np.ndarray): an array containing the a-params of the filter (AR part).
            b_list (np.ndarray): an array containing the b-params of the filter (MA part).
            scale_out_list (np.ndarray): an array containing surplus scaling factors at the filter outputs. 
            NOTE: This is due to bit-wise quantization, since all scaling is implemented as division or multiplication by 2. And scale_out is the left-over scaling.
            record (bool, optional): record the states of the filter (here the taps of AR part of the filter). Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: output containing the filtered signal in all filters + states.
        """
        # number of filters
        num_filter = len(a_list)
        
        # convert the parameters into jax.numpy
        sig_in = jnp.asarray(sig_in, dtype=jnp.float32)
        Bwf_list = jnp.asarray(Bwf_list, dtype=jnp.float32)
        Bb_list = jnp.asarray(Bb_list, dtype=jnp.float32)
        Baf_list = jnp.asarray(Baf_list, dtype=jnp.float32)
        a_list = jnp.asarray(a_list, dtype=jnp.float32)
        b_list = jnp.asarray(b_list, dtype=jnp.float32)
        scale_out_list = jnp.asarray(scale_out_list, dtype=jnp.float32)
        
        # number of states in AR part of the filter: 2 due to order 1 filters + 1 (assuming that we latch the state at the output of AR filter -> then feed to AR)
        NUM_STATE_AR = 2 + 1
        
        
        @partial(jax.jit, static_argnums=(6,))
        def _compile_filter(sig_in:jnp.ndarray, Bwf_list:np.ndarray, Bb_list:jnp.ndarray, a_list:jnp.ndarray, b_list:jnp.ndarray, scale_out_list: jnp.ndarray, record:bool=False):
            # initial state: all the registers of the AR filters
            init_state = jnp.zeros((num_filter, NUM_STATE_AR), dtype=jnp.float32)
            
            def forward(state_in, input):
                # compute the feedback part in AR part of the filters
                ar_feedback = -jnp.sum(state_in[:,0:-1] * a_list, axis=1)/2**Baf_list
                
                # combine scaled input (to avoid dead zone) with feedback coming from AR part
                merged_input = input * 2**Bwf_list + ar_feedback
                
                # find the next state after bitshift by Bb -> the next state is going to be loaded in the next clock
                next_state = merged_input/2**Bb_list
                
                ## compute the next state
                state_out = jnp.zeros_like(state_in)
                
                # first shift right when the clock comes -> then replace the new value into the state
                state_out = state_out.at[:,1:].set(state_in[:,0:-1])
                state_out = state_out.at[:,0].set(next_state)
                
                # compute the output based on the gated version of the state (for more signal stability)
                # so the past state is used in computation
                # NOTE: we have not done any truncation by B_wf because of the bits added for avoiding dead-zone
                # this was due to the fact that, low-freq filters get badly truncated.
                sig_out = jnp.sum(b_list * state_in, axis=1)
                
                # apply the final surplus scaling
                sig_out = sig_out * scale_out_list
                
                # if states needed to be recorded -> consider states as part of the output
                output = (sig_out,state_out) if record else (sig_out,)
                
                return (state_out, output)
            
            # apply the forward to compute 
            final_state, output = jax.lax.scan(forward, init_state, sig_in)
            
            return final_state, output
        
        
        final_state, output = _compile_filter(
            sig_in=sig_in,
            Bwf_list=Bwf_list,
            Bb_list=Bb_list,
            a_list=a_list,
            b_list=b_list,
            scale_out_list=scale_out_list,
            record=record,
        )
        
        if record:
            sig_out, state = output
            
            sig_out = np.asarray(sig_out, dtype=np.int64)
            recording = {"filter_AR_state": np.asarray(state, dtype=np.int64)}
            
        else:
            (sig_out, ) = output
            
            sig_out = np.asarray(sig_out, dtype=np.int64)
            recording = {}
        
        return sig_out, recording
                
    # set the  flag for jax version
    JAX_Filter = True

    print("\n\nJax version was found! Filterbank will be computed using jax speedup!\n\n")
    
except ModuleNotFoundError as e:
    print("No jax module was found for filter implementation! Filterbank will use python version (multi-procesing version).\n" + str(e))
    
    # set flag for jax 
    JAX_Filter = False