# -----------------------------------------------------------
# This module contains the PDM-based ADC for sampling the input audion signal. 
# It consists of two main parts:
#   (i)     a sigma-delta based PDM microphone that converts the input analog audio into a PDM bit stream where the 
#           relative frequency of 1-vs-0 depends on the amplitude of the signal
#   (ii)    a low-pass filter follows by decimation stage that processes the PDM bit stream and recovers the sampled
#           audio upto a given bit precision.
#
#
# In brief, PDM microphone with its internal sigma-delta modulation followed by low-pass filtering + decimation
# module implemented here yield an ADC for the input analog audio signal.
#
# The low-pass filtering is implemented as a **polyphase** filter structure to consume as less power as possible.
#
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 18.01.2023
# -----------------------------------------------------------

# FIXME: 
# (i)   I need to check if the final decimation factor was decided to be 32 or 64?
# (ii)  If 32, did hardware team decided to reduce PDM clock accordingly?
# (iii) What was the final decision for the filter length?
# I need to finalize this with Sunil?

# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter, ParameterBase
from rockpool.timeseries import TSContinuous

# - Other imports
import deltasigma as ds
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal as sp

import warnings

from typing import Union, Tuple, List, Dict
P_int = Union[int, ParameterBase]
P_float = Union[float, ParameterBase]
P_array = Union[np.array, ParameterBase]


# list of modules exported
__all__ = ["PDM_ADC", "PDM_Microphone", "PolyPhaseFIR_DecimationFilter"]


##------------------------------------------------------##
## design parameters currently used in the Xylo-A3 chip ##
##------------------------------------------------------##
SYSTEM_CLOCK_RATE = 50_000_000  # 50 MHz

AUDIO_SAMPLING_RATE = SYSTEM_CLOCK_RATE / (64 * 16)
AUDIO_CUTOFF_FREQUENCY = 20_000
AUDIO_CUTOFF_FREQUENCY_WIDTH = 0.2 * AUDIO_CUTOFF_FREQUENCY

PDM_FILTER_DECIMATION_FACTOR = 32
PDM_SAMPLING_RATE = AUDIO_SAMPLING_RATE * PDM_FILTER_DECIMATION_FACTOR

SIGMA_DELTA_ORDER = 2

DECIMATION_FILTER_LENGTH = 512
NUM_BITS_FILTER_Q = 16
NUM_BITS_ADC = 14

##---------------------------------------------------##


class PDM_Microphone(Module):
    def __init__(
        self, 
        sdm_order: int = SIGMA_DELTA_ORDER, 
        sdm_OSR: int = PDM_FILTER_DECIMATION_FACTOR, 
        fs: float = PDM_SAMPLING_RATE
    ):
        """this class simulates the PDM microphone which applies sigma-delta modulation on the input audio signal.
        The input to microphone is an analog audio signal and the output is a PDM bit-stream in which the relative
        frequency of 1-vs-0 depends on the instanteneous amplitude of the signal.

        Args:
            sdm_order (int): order of the sigma-delta modulator (conventional ones are 2 or 3). Defaults to SIGMA_DELTA_ORDER.
            sdm_OSR (int): oversampling rate in sigma-delta modulator. Defaults to PDM_FILTER_DECIMATION_FACTOR.
            fs (int): rate of the clock used for deriving the PDM microphone. 
                    NOTE: PDM microphone can be derived by various clock rates. By changing the clock rate and sdm_OSR we can
                    keep the sampling rate of the audio fixed.
        """

        # sigma-delta modulator parameters
        self._sdm_order: P_int = Parameter(sdm_order)
        self._sdm_OSR : P_int = Parameter(sdm_OSR)
        self._fs : P_float = Parameter(fs)
        
        # build the sigmal-delta module
        self._sdm_module = ds.synthesizeNTF(self.sdm_order, self.sdm_OSR, opt=1)
        
        
    def evolve(self, audio:np.ndarray, audio_sampling_rate:float)->np.ndarray:
        """This function takes the input audio signal and produces the PDM bit-stream.
        NOTE: the audio signal should be normalized to the valid range of sigma-delat modulator [-1.0, 1.0].
              If not in this range, clipoing should be applied manually to limit the signal into this range.
        

        Args:
            audio (np.ndarray): input audio signal.
            audio_sampling_rate (float): sampling rate of the audio signal.
            NOTE: In reality the input signal to sigma-delta modulator in PDM microphone is the analog audio signal.
            In simulation, however, we have to still use a sampled version of this analog signal as representative.

        Raises:
            ValueError: if the amplitude is not scaled properly and is not in the valid range [-1.0, 1.0]

        Returns:
            np.ndarray: array containing PDM bit-stream at the output of the microphone.
        """
        
        if audio.ndim != 1:
            raise ValueError("only single-channel audio signals can be processed by the sigma-delta modulator in PDM microphone!")

        if np.max(np.abs(audio)) > 1.0:
            raise ValueError(
                "Some of the signal samples have an amplitude larger than 1.0.\n"+\
                "Sigma-delta modulator is designed to work with signal values normalized in the range [-1.0, 1.0].\n"+\
                "Normalize the signal or clip it to the range [-1.0, 1.0] manually before applying it to PDM microhpne."
            )

        if audio_sampling_rate < self.fs:
            warnings.warn("\n\n" + " warnings ".center(120, "+") + "\n" +\
                f"In practice, the input to the PDM microphone (fed by a clock of rate:{self.fs}) is the analog audio signal.\n"+\
                "In simulations, however, we have to use sampled audio signal at the input to mimic this analog signal.\n"+\
                f"Here we resample the input auido (of course artificially) to the higher sample rate of PDM mcirophone ({self.fs}).\n"+\
                "For a more realistic simulation, it is better to provide an audio signal which is originally sampled with a higher rate.\n"+
                "+"*120 + "\n\n"
            )
        
        # resample the input audio signal to the sampling rate of the sigma-delta modulator
        duration = len(audio)/audio_sampling_rate
        time_resample = np.arange(0, duration, step=1/self.fs)
        
        audio_resampled = TSContinuous.from_clocked(samples=audio.ravel(), dt=1/audio_sampling_rate)(time_resample).ravel()
        
        # compute the sigma-delta modulation
        audio_pdm, *_ = ds.simulateDSM(audio_resampled, self._sdm_module)

        # use the integer format for the final {-1,+1}-valued binary PDM data
        if audio_pdm.dtype != np.int64:
            audio_pdm = audio_pdm.astype(np.int64)
            
        unique_vals = set(np.unique(audio_pdm))
        if unique_vals != {-1,1} and unique_vals !={1} and unique_vals!={-1}:
            raise ValueError(
                "The output of sigma-delta modulator should be a {-1,+1}-valued signal.\n"+\
                "This problem may arise when the sigma-delta simulator is unstable!\n"
            )
        
        return audio_pdm


    def __call__(self, *args, **kwargs)->np.ndarray:
        """This is the same as `evolve` function.

        Returns:
            np.ndarray: binary PDM signal resulted from sigma-delta modulator.
        """
        return self.evolve(*args, **kwargs)

    def __repr__(self) -> str:
        string = "This is the module for simulating PDM microphone which uses sigma-delta modulation.\n"+\
                "The input analog audio signal is mapped to a binary stream of modulated data which is them interpolated to recover a sampled version of the analog input.\n"+\
                "Parameters:\n"+\
                f"Sigma-Delta modulation order: {self.sdm_order}\n"+\
                f"Sigma-Delta oversampling rate (ratio between the rate of PDM clock and target audio sampling rate): {self.sdm_OSR}\n"+\
                f"Sigma-Delta clock rate: {self.fs}\n"
        return string


    # collection of attributes
    @property
    def fs(self):
        return self._fs
    
    @fs.setter
    def fs(self, fs:float):
        self._fs: P_float = Parameter(fs)
        self._sdm_module = ds.synthesizeNTF(self.sdm_order, self.sdm_OSR, opt=1)
    
    @property
    def sdm_order(self):
        return self._sdm_order
        
    
    @sdm_order.setter
    def sdm_order(self, sdm_order: int):
        self._sdm_order: P_int = Parameter(sdm_order)
        self._sdm_module = ds.synthesizeNTF(self.sdm_order, self.sdm_OSR, opt=1)

    @property
    def sdm_OSR(self):
        return self._sdm_OSR
    
    @sdm_OSR.setter
    def sdm_OSR(self, sdm_OSR):
        self._sdm_OSR: P_int = Parameter(sdm_OSR)
        self._sdm_module = ds.synthesizeNTF(self.sdm_order, self.sdm_OSR, opt=1)




class PolyPhaseFIR_DecimationFilter(Module):
    def __init__(
        self,
        decimation_factor: int = PDM_FILTER_DECIMATION_FACTOR,
        cutoff: float = AUDIO_CUTOFF_FREQUENCY,
        cutoff_width: float = AUDIO_CUTOFF_FREQUENCY_WIDTH,
        filt_length: int = DECIMATION_FILTER_LENGTH,
        num_bits_filter_Q: int = NUM_BITS_FILTER_Q,
        num_bits_output: int = NUM_BITS_ADC,
        fs: float = PDM_SAMPLING_RATE,
    ):
        """this class implements the low-pass decimation filter for PDM binary data implemented based on polyphase FIR filters.
        This has several advantages:
            - filter does not need any multiplication as the input signal is binary (this is not the case in IIR filters),

            - for FIR filters, one can use polyphase structure such that only those samples that are kept after decimation of filter output are computed.
            This is not the case in IIR filters since one needs to compute all the samples of the signal and then decimate them, thus, throwing away a
            majority of the computed samples.

            - FIR filters have linear phase and group delay and this can be advantageous in terms of feature extraction.

        Args:
            decimation_factor (int, optional): how much the signal needs to be decimated or subsampled. Defaults to PDM_FILTER_DECIMATION_FACTOR.
            cutoff (float, optional): cutoff frequency of the filter. Defaults to AUDIO_CUTOFF_FREQUENCY.
            cutoff_width (float, optional): the transition width of the filter. Defaults to AUDIO_CUTOFF_FREQUENCY_WIDTH (is set to 20% of the cutoff frequency).
            filt_length (int, optional): length of the designed FIR filter. Defaults to DECIMATION_FILTER_LENGTH.
            num_bits_filter_Q (int, optional): number of bits used for quantzing the filter coefficients. Defaults to NUM_BITS_FILTER_Q bits.
            num_bits_output (int, optional): number of bits devoted to the final sampled audio obtained after low-pass filtering + decimation. 
                                            Officially this corresponds to number of quantization bits in a conventional SAR ADC. Defaults to NUM_BITS_ADC bits.
            fs (float, optional): sampling frequency or bit rate of PDM bit-stream. Defaults to PDM_SAMPLING_RATE.
        """
        
        # ratio between the clock rate of PDM microphone and the target sampling rate of the final sampled audio obatined after low-pass filtering + decimation
        self._decimation_factor : P_float = Parameter(decimation_factor)

        # cutoff frequency of the low-pass filter used for audio recovery
        self._cutoff : P_float = Parameter(cutoff)

        # the transition width of the low-pass filter
        self._cutoff_width : P_float = Parameter(cutoff_width)

        # length of the FIR low-pass filter 
        self._filt_length: P_int = Parameter(filt_length)

        # number of bits used for quantizing the FIR filter taps
        self._num_bits_filter_Q: P_int = Parameter(num_bits_filter_Q)

        # number of bits in the final output sampled audio
        self._num_bits_output: P_int = Parameter(num_bits_output)

        # clock rate of the PDM microphone / clock rate of the binary PDM signal received from the microphone
        self._fs: P_float = Parameter(fs)


        # filter quantized impulse response and the number of bits at the output before final quantization (bit truncation)
        # NOTE: 
        # After low-pass filteirng the PDM bit-stream, one obtains the pre-quantization/pre-truncation signal.
        # Enough number of bits are needed to avoid any over and underflow in the filter.
        # Also, some of the LSB's in the pre-truncation output need to be dropped to obtain the final sampled audio with
        # targetted number of bits.
        h, num_bits_output_pre_Q = self._build_filter()
        self._h : P_array = Parameter(h)
        self._num_bits_output_pre_Q: P_int = Parameter(num_bits_output_pre_Q)


    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        """this function processes the {-1,+1}-valued PDM signal and produces the final sampled audio signal.

        Args:
            sig_in (np.ndarray): input PDM signal.

        Returns:
            np.ndarray: final sampled audio signal.
        """
        self._verify_pdm_signal(sig_in)
        

        # convert the signal into 0-1 representtaion to have a simpler implementation in hardware with less addition
        # this efficiency is, however, not considered in the simulation implemented here.
        sig_in = (1 + sig_in) // 2
        dc_offset = self.h.sum()

        # convert the signal and the filter into polyphase representation
        h_pp = self._filter_polyphase_array(self.h)
        sig_in_pp = self._signal_polyphase_array(sig_in)

        # compute output signal before final bit-truncation
        sig_out_pre_Q = 0

        for sig_p, h_p in zip(sig_in_pp, h_pp):
            sig_out_pre_Q += np.convolve(h_p, sig_p, mode="full")

        # apply the dc-level adjustment
        sig_out_pre_Q = (sig_out_pre_Q << 1) - dc_offset

        # truncate the LSBs for the output signal
        num_right_bit_shifts = self.num_bits_output_pre_Q - self.num_bits_output

        if num_right_bit_shifts > 0:
            sig_out = sig_out_pre_Q >> num_right_bit_shifts
        else:
            sig_out = sig_out_pre_Q << (-num_right_bit_shifts)

        return sig_out
    

    def __call__(self, *args, **kwargs):
        """
        this is the same as `evolve` function.
        """
        return self.evolve(*args, **kwargs)

    
    def print_hardware_specs(self):
        """
        this function prints the details needed for implementing the filter in hardware.
        """
        # print the parameters
        print(self)

        # print the filter impulse response
        print_width = int(np.log10(2**self._num_bits_filter_Q) + 6)
        print("filter impulse response is:")

        for i, h_tap in enumerate(self._h):
            print(f"{h_tap}".ljust(print_width), end="")

            if (i + 1) % 16 == 0:
                print()


    def __repr__(self) -> str:
        string = (
            "Polyphase FIR filter:\n"
            + f"PDM sampling rate: {self.fs}\n"
            + f"filter length: {self.filt_length}\n"
            + f"filter cutoff frequency: {self.cutoff}\n"
            + f"filter transition width: {self.cutoff_width}\n"
            + f"decimation factor: {self.decimation_factor}\n"
            + f"number of quantization bits for filter taps: {self.num_bits_filter_Q}\n"
            + f"number of bits at the output of the filter before final bit-truncation: {self._num_bits_output_pre_Q}\n"
            + f"number of bits at the final output: {self._num_bits_output}\n"
        )

        return string


    # functions for converting the input PDM signal and filter taps into Polyphase representation
    def _filter_polyphase_array(self, filt:np.ndarray)->np.ndarray:
        """this function converts the filter into its polyphase array representation.
        For more details, see the documentation in https://spinystellate.office.synsense.ai/saeid.haghighatshoar/pdm-microphone-analysis

        Args:
            filt (np.ndarray): input signal representing the impulse response of the FIR filter.

        Returns:
            np.ndarray: polyphase representation of the filter.
        """
        if filt.ndim > 1:
            raise ValueError("the filter should be a 1-dim array!")

        num_zero_samples = (self.decimation_factor - (len(filt) % self.decimation_factor)) % self.decimation_factor

        if num_zero_samples > 0:
            # append zeros to the end in order not to lose the 0-phase in polyphase
            filt = np.concatenate([filt.ravel(), [0] * num_zero_samples])

        filt = np.concatenate([[0] * (self._decimation_factor - 1), filt, [0]])
        filt = filt.reshape(-1, self._decimation_factor).T[::-1, :]

        return filt


    def _signal_polyphase_array(self, sig_in:np.ndarray)->np.ndarray:
        """this function converts the input signal into its polyphase array representation.

        Args:
            sig_in (np.ndarray): input signal (here {-1,+1}-valued PDM signal).

        Returns:
            np.ndarray: polyphase representation of the signal.
        """
        if sig_in.ndim > 1:
            raise ValueError("input signal should be 1-dim!")

        num_zero_samples = (self._decimation_factor - (len(sig_in) % self._decimation_factor)) % self._decimation_factor

        if num_zero_samples > 0:
            # append zeros to the end in order not to lose the 0-phase in polyphase
            sig_in = np.concatenate([sig_in.ravel(), [0] * num_zero_samples])

        # reshape the signal
        sig_in = sig_in.reshape(-1,self._decimation_factor).T

        return sig_in
    

    # collection of getter and setter for the attributes
    # this is needed since changing one of the parameters requires adjusting the other parameters as well.
    @property
    def decimation_factor(self):
        return self._decimation_factor

    @decimation_factor.setter
    def decimation_factor(self, decimation_factor: int):
        self._decimation_factor: P_int = Parameter(decimation_factor)
        h, num_bits_output_pre_Q = self._build_filter()
        self._h : P_array = Parameter(h)
        self._num_bits_output_pre_Q: P_int = Parameter(num_bits_output_pre_Q)


    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, cutoff: int):
        self._cutoff = cutoff

        # filter quantized impulse response and the number of bits at the output before final quantization (bit truncation)
        h, num_bits_output_pre_Q = self._build_filter()
        self._h : P_array = Parameter(h)
        self._num_bits_output_pre_Q: P_int = Parameter(num_bits_output_pre_Q)

    @property
    def cutoff_width(self):
        return self._cutoff_width

    @cutoff_width.setter
    def cutoff_width(self, cutoff_width: float):
        self._cutoff_width = cutoff_width

        # filter quantized impulse response and the number of bits at the output before final quantization (bit truncation)
        h, num_bits_output_pre_Q = self._build_filter()
        self._h : P_array = Parameter(h)
        self._num_bits_output_pre_Q: P_int = Parameter(num_bits_output_pre_Q)

    @property
    def filt_length(self):
        return self._filt_length

    @filt_length.setter
    def filt_length(self, filt_length: int):
        self._filt_length = filt_length

        # filter quantized impulse response and the number of bits at the output before final quantization (bit truncation)
        h, num_bits_output_pre_Q = self._build_filter()
        self._h : P_array = Parameter(h)
        self._num_bits_output_pre_Q: P_int = Parameter(num_bits_output_pre_Q)

    @property
    def num_bits_filter_Q(self):
        return self._num_bits_filter_Q

    @num_bits_filter_Q.setter
    def num_bits_filter_Q(self, num_bits_filter_Q: int):
        self._num_bits_filter_Q = num_bits_filter_Q

        # filter quantized impulse response and the number of bits at the output before final quantization (bit truncation)
        h, num_bits_output_pre_Q = self._build_filter()
        self._h : P_array = Parameter(h)
        self._num_bits_output_pre_Q: P_int = Parameter(num_bits_output_pre_Q)

    @property
    def num_bits_output_pre_Q(self):
        return self._num_bits_output_pre_Q

    @property
    def num_bits_output(self):
        return self._num_bits_output

    @num_bits_output.setter
    def num_bits_output(self, num_bits_output):
        # no filter recomputation is needed
        self._num_bits_output = num_bits_output

    @property
    def num_right_bit_shifts_output(self):
        return self._num_bits_output_pre_Q - self._num_bits_output

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, fs: float):
        self._fs = fs

        # filter quantized impulse response and the number of bits at the output before final quantization (bit truncation)
        h, num_bits_output_pre_Q = self._build_filter()
        self._h : P_array = Parameter(h)
        self._num_bits_output_pre_Q: P_int = Parameter(num_bits_output_pre_Q)

    @property
    def h(self):
        return self._h


    # initialization part
    def _build_filter(self) -> np.ndarray:
        """
        this function computes the impulse response of the FIR filter.
        """
        numtaps = self._filt_length
        cutoff = self._cutoff
        width = self._cutoff_width
        pass_zero = True
        scale = True
        fs = self._fs

        warnings.warn(
            "filter cutoff-width was unset in the implementation to futher lower-down the tail response!"
        )

        # scipy.signal.firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, nyq=None, fs=None)
        h_vec = sp.firwin(
            numtaps=numtaps,
            cutoff=cutoff,
            # width=width,
            pass_zero=pass_zero,
            scale=scale,
            fs=fs,
        )

        # quantize the filter:
        # NOTE: we have two objective in quantizing the filter:
        # (i)   just quantize to get the maximum precision in the given number of quantization bits
        # (ii)  try to quantize the filter in a way that worst-case amplitude of the filter is as close
        #       to a power of two as possible -> this simplifies the bit-truncation after filter computation and decimation

        EPS = 1e-10
        h_vec_peak_1 = h_vec / (np.max(np.abs(h_vec)) * (1 + EPS))
        h_vec_Q = (h_vec_peak_1 * 2 ** (self._num_bits_filter_Q - 1)).astype(np.int64)

        ### find the worst-case amplitude of the filter and push it to be a power of two

        # maximum amplitude of the filter with the quantized taps and with {+1,-1} PDM input
        max_val = np.sum(np.abs(h_vec_Q))

        ## closest number of bits needed for the output
        # additional one bit for the sign bit
        num_bits_output_pre_Q = int(np.floor(np.log2(max_val))) + 1

        # maximum level that we can cover while keeping the filter coefficients to the devoted quantization bits
        max_val_pow2_target = 2 ** (num_bits_output_pre_Q - 1) - 1

        # amount of scaling needed to lower down the filter taps while keeping them in the devoted quantization bits
        scale = max_val_pow2_target / max_val

        # scale the filter taps
        h_vec_scaled = scale * h_vec_peak_1

        # quantize the scaled version
        h_vec_Q = (h_vec_scaled * 2 ** (self._num_bits_filter_Q - 1)).astype(np.int64)

        return h_vec_Q, num_bits_output_pre_Q

    def _verify_pdm_signal(self, sig_in: np.ndarray):
        """ input PDM signal : {-1,+1}-valued """
        if sig_in.ndim > 1:
            raise ValueError("input signal should be 1-dim.")

        if sig_in.dtype != np.int64:
            raise ValueError(
                "the components of the input signal should be of type np.int64! Ideally +1 and -1 for PDM!"
            )
        # check if the input siganl is binary
        unique_values = set(np.unique(sig_in))
        if unique_values != {1, -1} and unique_values != {1} and unique_values != {-1}:
            raise ValueError(
                "Input signal should be binary-valued (PDM signal) in anti-podal format in {-1, +1}!\n"+\
                "Convert to integer in case needed to make sure that the signal is indeed binary with -1 or +1 values!"
            )

    # some utility functions needed for testing and verifiaction

    def fullevolve(self, sig_in: np.ndarray) -> np.ndarray:
        """this function returns the output of the filter before decimation.
        This is not needed in the implementation but is necessary in performance analysis to see what was the 
        high-precision signal before applying the decimation to lower-down its sampling rate to that of target audio sampling rate.

        Args:
            sig_in (np.ndarray): input PDM signal.

        Returns:
            np.ndarray: output sampled audio signal before decimation.
        """

        self._verify_pdm_signal(sig_in)

        # convert the input signal into 0-1 to reduce the number of additions
        sig_in = (1 + sig_in) // 2
        dc_offset = self.h.sum()

        sig_out_pre_Q = np.convolve(self.h, sig_in, mode="full")

        # apply the dc-offset removal
        sig_out_pre_Q = (sig_out_pre_Q << 1) - dc_offset

        # truncate the LSBs for the output signal
        num_right_bit_shifts = self.num_bits_output_pre_Q - self.num_bits_output

        if num_right_bit_shifts > 0:
            sig_out = sig_out_pre_Q >> num_right_bit_shifts
        else:
            sig_out = sig_out_pre_Q << (-num_right_bit_shifts)

        return sig_out

    def plot_filter(self):
        """
        this function plots the time and frequency response of the filter.
        """
        # scipy.signal.freqz(b, a=1, worN=512, whole=False, plot=None, fs=6.283185307179586, include_nyquist=False)
        num_samples = 40_000

        # normalize the filter for having gain 0dB
        h_vec = self._h
        h_vec /= np.sum(h_vec)

        f_vec, f_response = sp.freqz(
            b=h_vec,
            a=1,
            worN=num_samples,
            fs=self._fs,
        )

        plt.figure(figsize=(12, 12))
        plt.subplot(211)
        plt.semilogx(f_vec, 20 * np.log10(f_response))
        plt.grid(True, which="both")
        plt.xlim([0, self.fs / 2])
        plt.ylim([-90, 3])
        plt.xlabel("frequency (Hz)")
        plt.ylabel("frequency response [ 20 x log10 |H(f)| ]")
        plt.title("specifications of the FIR filter used in polyphase implementtaion")

        plt.subplot(212)
        plt.plot(self._h)
        plt.ylabel(
            f"Filter impulse response (Quantized to {self.num_bits_filter_Q} bits)"
        )
        plt.xlabel("filter taps")
        plt.grid(True)

        plt.show()

    

class PDM_ADC(Module):
    def __init__(
        self,
        sdm_order: int = SIGMA_DELTA_ORDER, 
        sdm_OSR: int = PDM_FILTER_DECIMATION_FACTOR, 
        fs: int = PDM_SAMPLING_RATE,
        cutoff: float = AUDIO_CUTOFF_FREQUENCY,
        cutoff_width: float = AUDIO_CUTOFF_FREQUENCY_WIDTH,
        filt_length: int = DECIMATION_FILTER_LENGTH,
        num_bits_filter_Q: int = NUM_BITS_FILTER_Q,
        num_bits_output: int = NUM_BITS_ADC,
    ):
        """
        Analog-to-Digital (ADC) module for Xylo-A3 chip consisting of 
            (i)  PDM microphone converting the input analog audio signal into PDM bit-stream.
            (ii) low-pass filtering and decimation module converting the binary PDM stream into the target sampled audio signal. 


        Args:
            sdm_order (int, optional): order of sigma-delta modulator. Defaults to SIGMA_DELTA_ORDER.
            sdm_OSR (int, optional): oversampling rate of PDM microphone : ratio between PDM clock and rate of target audio signal. Defaults to PDM_FILTER_DECIMATION_FACTOR.
            This is the same as decimation factor in PDM low-pass filter. Defaults to PDM_FILTER_DECIMATION_FACTOR.
            fs (int, optional): sampling rate/clock rate of PDM module. Defaults to PDM_SAMPLING_RATE.
            cutoff (float, optional): cutoff frequency of the low-pass filter used for recovery of target sampled audio from PDM bit-stream. Defaults to AUDIO_CUTOFF_FREQUENCY.
            cutoff_width (float, optional): transition withs of the low-pass filter. Defaults to AUDIO_CUTOFF_FREQUENCY_WIDTH (is set to 20% of the cutoff frequency).
            filt_length (int, optional): length of the FIR low-pass filter. Defaults to DECIMATION_FILTER_LENGTH.
            num_bits_filter_Q (int, optional): number of bits used for quantizing the filter taps. Defaults to NUM_BITS_FILTER_Q.
            num_bits_output (int, optional): target number of bits in the final sampled audio obtained after low-pass filtering and decimation.
            This is equivalent to the number of bits in the equivalent ADC. Defaults to NUM_BITS_ADC.
        """
        super().__init__()
        
        # two modules of equivalent ADC
        self.pdm_microphone = PDM_Microphone(
            sdm_order=sdm_order,
            sdm_OSR=sdm_OSR,
            fs=fs,
        )

        self.low_pass_decimation = PolyPhaseFIR_DecimationFilter(
            decimation_factor=sdm_OSR,
            cutoff=cutoff,
            cutoff_width=cutoff_width,
            filt_length=filt_length,
            num_bits_filter_Q=num_bits_filter_Q,
            num_bits_output=num_bits_output,
            fs=fs,
        )
        
    def parameters(self, family: Union[str, Tuple, List] = None) -> Dict:
        """ return the parameters as a dictionary. """
        return {
            **super().parameters(family), 
            "pdm_microphone": self.pdm_microphone.parameters(family),
            "low_pass_decimation": self.low_pass_decimation.parameters(family),
        }
        
    def simulation_parameters(self, family: Union[str, Tuple, List] = None) -> Dict:
        return {
            **super().parameters(family), 
            "pdm_microphone": self.pdm_microphone.simulation_parameters(family),
            "low_pass_decimation": self.low_pass_decimation.simulation_parameters(family),
        }

    def evolve(self, audio:np.ndarray, audio_sampling_rate:float)->np.ndarray:
        """This function takes the input audio signal and produces the 1-bit PDM modulation.
        Then it recovers the sampled audio from binary PDM signal with low-pass filtering and decimation.
        NOTE: the audio signal should be normalized to the valid range of sigma-delat modulator (-1.0, 1.0).
        

        Args:
            audio (np.ndarray): input audio signal.
            audio_sampling_rate (float): sampling rate of the audio signal.

        Raises:
            ValueError: if the amplitude is not scaled properly to the valid range [-1.0, 1.0].

        Returns:
            np.ndarray: array containing the recovered sampled audio : output of the equivalent ADC 
        """

        # obtain the PDM binary signal
        audio_pdm = self.pdm_microphone(audio=audio, audio_sampling_rate=audio_sampling_rate)
        
        # recover the sampled audio from the PDM signal
        adc_pdm_output = self.low_pass_decimation.evolve(sig_in=audio_pdm)

        return adc_pdm_output

    def __call__(self, *args, **kwargs)->np.ndarray:
        """this module is the same as `evolve` function.

        Returns:
            np.ndarray: output sampled audio.
        """
        return self.evolve(*args, **kwargs)

    def __repr__(self) -> str:
        string = "".center(120, "+")+"\n"+\
            "PDM ADC for sampling the audio signal in Xylo-A3 consisting of the following modules:\n\n"+\
            "-> PDM Microphone:\n"+\
            self.pdm_microphone.__repr__()+\
            "\n-> Low-pass filter + decimation:\n"+\
            self.low_pass_decimation.__repr__() +\
            "".center(120, "+")+"\n"
        
        return string
