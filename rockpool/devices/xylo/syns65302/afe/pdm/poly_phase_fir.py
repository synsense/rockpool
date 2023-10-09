"""
A poly phase low-pass filter follows by decimation stage that processes the PDM bit stream 
and recovers the sampled audio upto a given bit precision.
"""

from typing import Any, Tuple

import numpy as np
import scipy.signal as sp

from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_CUTOFF_FREQUENCY,
    AUDIO_CUTOFF_FREQUENCY_WIDTH,
    DECIMATION_FILTER_LENGTH,
    NUM_BITS_FILTER_Q,
    NUM_BITS_PDM_ADC,
    PDM_FILTER_DECIMATION_FACTOR,
    PDM_SAMPLING_RATE,
)
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from rockpool.typehints import P_float, P_int

__all__ = ["PolyPhaseFIR"]


class PolyPhaseFIR(Module):
    """
    This class implements the low-pass decimation filter for PDM binary data implemented based on polyphase FIR filters

    This has several advantages:
        - filter does not need any multiplication as the input signal is binary (this is not the case in IIR filters),

        - for FIR filters, one can use polyphase structure such that only those samples that are kept after decimation of filter output are computed.
        This is not the case in IIR filters since one needs to compute all the samples of the signal and then decimate them, thus, throwing away a
        majority of the computed samples.

        - FIR filters have linear phase and group delay and this can be advantageous in terms of feature extraction.
    """

    def __init__(
        self,
        decimation_factor: int = PDM_FILTER_DECIMATION_FACTOR,
        cutoff: float = AUDIO_CUTOFF_FREQUENCY,
        cutoff_width: float = AUDIO_CUTOFF_FREQUENCY_WIDTH,
        filt_length: int = DECIMATION_FILTER_LENGTH,
        num_bits_filter_Q: int = NUM_BITS_FILTER_Q,
        num_bits_output: int = NUM_BITS_PDM_ADC,
        fs: float = PDM_SAMPLING_RATE,
    ):
        """
        Initialise a PolyPhaseFIR module

        Args:
            decimation_factor (int, optional): how much the signal needs to be decimated or subsampled. Defaults to PDM_FILTER_DECIMATION_FACTOR.
            cutoff (float, optional): cutoff frequency of the filter. Defaults to AUDIO_CUTOFF_FREQUENCY.
            cutoff_width (float, optional): the transition width of the filter. Defaults to AUDIO_CUTOFF_FREQUENCY_WIDTH (is set to 20% of the cutoff frequency).
            filt_length (int, optional): length of the designed FIR filter. Defaults to DECIMATION_FILTER_LENGTH.
            num_bits_filter_Q (int, optional): number of bits used for quantizing the filter coefficients. Defaults to NUM_BITS_FILTER_Q bits.
            num_bits_output (int, optional): number of bits devoted to the final sampled audio obtained after low-pass filtering + decimation.
                                            Officially this corresponds to number of quantization bits in a conventional SAR ADC. Defaults to NUM_BITS_PDM_ADC bits.
            fs (float, optional): sampling frequency or bit rate of PDM bit-stream. Defaults to PDM_SAMPLING_RATE.
        """
        super().__init__(shape=1, spiking_input=True, spiking_output=True)

        # ratio between the clock rate of PDM microphone and the target sampling rate of the final sampled audio obatined after low-pass filtering + decimation
        self.decimation_factor: P_float = SimulationParameter(decimation_factor)
        """ float: ratio between the clock rate of PDM microphone and the target sampling rate of the final sampled audio obatined after low-pass filtering + decimation """

        # cutoff frequency of the low-pass filter used for audio recovery
        self.cutoff: P_float = SimulationParameter(cutoff)
        """ float: cutoff frequency of the low-pass filter used for audio recovery """

        # the transition width of the low-pass filter
        self.cutoff_width: P_float = SimulationParameter(cutoff_width)
        """ float: the transition width of the low-pass filter """

        # length of the FIR low-pass filter
        self.filt_length: P_int = SimulationParameter(filt_length)
        """ int: length of the FIR low-pass filter"""

        # number of bits used for quantizing the FIR filter taps
        self.num_bits_filter_Q: P_int = SimulationParameter(num_bits_filter_Q)
        """ int: number of bits used for quantizing the FIR filter taps """

        # number of bits in the final output sampled audio
        self.num_bits_output: P_int = SimulationParameter(num_bits_output)
        """ int: number of bits in the final output sampled audio """

        # clock rate of the PDM microphone / clock rate of the binary PDM signal received from the microphone
        self.fs: P_float = SimulationParameter(fs)
        """ float: clock rate of the PDM microphone / clock rate of the binary PDM signal received from the microphone """

        # filter quantized impulse response and the number of bits at the output before final quantization (bit truncation)
        # NOTE:
        # After low-pass filtering the PDM bit-stream, one obtains the pre-quantization/pre-truncation signal.
        # Enough number of bits are needed to avoid any over and underflow in the filter.
        # Also, some of the LSB's in the pre-truncation output need to be dropped to obtain the final sampled audio with
        # target number of bits.
        self._h, self._num_bits_output_pre_Q = self._build_filter()

    def evolve(
        self, sig_in: np.ndarray, record: bool = False, *args, **kwargs
    ) -> Tuple[np.ndarray, dict, dict]:
        """this function processes the {-1,+1}-valued PDM signal and produces the final sampled audio signal.

        Args:
            sig_in (np.ndarray): input PDM signal.

        Returns:
            np.ndarray: final sampled audio signal.
        """
        self._verify_pdm_signal(sig_in)

        # convert the signal into 0-1 representation to have a simpler implementation in hardware with less addition
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

        return sig_out, self.state(), {}

    def print_hardware_specs(self):
        """
        this function prints the details needed for implementing the filter in hardware.
        """
        # print the parameters
        print(self)

        # print the filter impulse response
        print_width = int(np.log10(2**self.num_bits_filter_Q) + 6)
        print("filter impulse response is:")

        for i, h_tap in enumerate(self._h):
            print(f"{h_tap}".ljust(print_width), end="")

            if (i + 1) % 16 == 0:
                print()

    # functions for converting the input PDM signal and filter taps into Polyphase representation
    def _filter_polyphase_array(self, filt: np.ndarray) -> np.ndarray:
        """this function converts the filter into its polyphase array representation.
        For more details, see the documentation in https://spinystellate.office.synsense.ai/saeid.haghighatshoar/pdm-microphone-analysis

        Args:
            filt (np.ndarray): input signal representing the impulse response of the FIR filter.

        Returns:
            np.ndarray: polyphase representation of the filter.
        """
        if filt.ndim > 1:
            raise ValueError("the filter should be a 1-dim array!")

        num_zero_samples = (
            self.decimation_factor - (len(filt) % self.decimation_factor)
        ) % self.decimation_factor

        if num_zero_samples > 0:
            # append zeros to the end in order not to lose the 0-phase in polyphase
            filt = np.concatenate([filt.ravel(), [0] * num_zero_samples])

        filt = np.concatenate([[0] * (self.decimation_factor - 1), filt, [0]])
        filt = filt.reshape(-1, self.decimation_factor).T[::-1, :]

        return filt

    def _signal_polyphase_array(self, sig_in: np.ndarray) -> np.ndarray:
        """this function converts the input signal into its polyphase array representation.

        Args:
            sig_in (np.ndarray): input signal (here {-1,+1}-valued PDM signal).

        Returns:
            np.ndarray: polyphase representation of the signal.
        """
        if sig_in.ndim > 1:
            raise ValueError("input signal should be 1-dim!")

        num_zero_samples = (
            self.decimation_factor - (len(sig_in) % self.decimation_factor)
        ) % self.decimation_factor

        if num_zero_samples > 0:
            # append zeros to the end in order not to lose the 0-phase in polyphase
            sig_in = np.concatenate([sig_in.ravel(), [0] * num_zero_samples])

        # reshape the signal
        sig_in = sig_in.reshape(-1, self.decimation_factor).T

        return sig_in

    def __setattr__(self, name: str, val: Any):
        # - Use super-class setattr to assign attribute
        super().__setattr__(name, val)

        # - Re-generate filter
        if (
            not self._in_Module_init
            and name != "_h"
            and name != "_num_bits_output_pre_Q"
        ):
            self._h, self._num_bits_output_pre_Q = self._build_filter()

    @property
    def h(self):
        return self._h

    @property
    def num_bits_output_pre_Q(self):
        return self._num_bits_output_pre_Q

    # initialization part
    def _build_filter(self) -> np.ndarray:
        """
        this function computes the impulse response of the FIR filter.
        """
        numtaps = self.filt_length
        cutoff = self.cutoff
        width = self.cutoff_width
        pass_zero = True
        scale = True
        fs = self.fs

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
        h_vec_Q = (h_vec_peak_1 * 2 ** (self.num_bits_filter_Q - 1)).astype(np.int64)

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
        h_vec_Q = (h_vec_scaled * 2 ** (self.num_bits_filter_Q - 1)).astype(np.int64)

        return h_vec_Q, num_bits_output_pre_Q

    def _verify_pdm_signal(self, sig_in: np.ndarray):
        """input PDM signal : {-1,+1}-valued"""
        if sig_in.ndim > 1:
            raise ValueError("input signal should be 1-dim.")

        if sig_in.dtype != np.int64:
            raise ValueError(
                "the components of the input signal should be of type np.int64! Ideally +1 and -1 for PDM!"
            )
        # check if the input signal is binary
        unique_values = set(np.unique(sig_in))
        if unique_values != {1, -1} and unique_values != {1} and unique_values != {-1}:
            raise ValueError(
                "Input signal should be binary-valued (PDM signal) in anti-podal format in {-1, +1}!\n"
                + "Convert to integer in case needed to make sure that the signal is indeed binary with -1 or +1 values!"
            )

    # some utility functions needed for testing and verification

    def _evolve_no_decimation(self, sig_in: np.ndarray) -> np.ndarray:
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

    def _plot_filter(self):
        """
        this function plots the time and frequency response of the filter.
        """
        import matplotlib.pyplot as plt

        num_samples = 40_000

        # normalize the filter for having gain 0dB
        h_vec = self._h.astype(float)
        h_vec /= np.sum(h_vec)

        f_vec, f_response = sp.freqz(
            b=h_vec,
            a=1,
            worN=num_samples,
            fs=self.fs,
        )

        plt.figure(figsize=(12, 12))
        plt.subplot(211)
        plt.semilogx(f_vec, 20 * np.log10(f_response))
        plt.grid(True, which="both")
        plt.xlim([0, self.fs / 2])
        plt.ylim([-90, 3])
        plt.xlabel("frequency (Hz)")
        plt.ylabel("frequency response [ 20 x log10 |H(f)| ]")
        plt.title("specifications of the FIR filter used in polyphase implementation")

        plt.subplot(212)
        plt.plot(self._h)
        plt.ylabel(
            f"Filter impulse response (Quantized to {self.num_bits_filter_Q} bits)"
        )
        plt.xlabel("filter taps")
        plt.grid(True)

        plt.show()

    def _info(self) -> str:
        string = (
            "Polyphase FIR filter:\n"
            + f"PDM sampling rate: {self.fs}\n"
            + f"filter length: {self.filt_length}\n"
            + f"filter cutoff frequency: {self.cutoff}\n"
            + f"filter transition width: {self.cutoff_width}\n"
            + f"decimation factor: {self.decimation_factor}\n"
            + f"number of quantization bits for filter taps: {self.num_bits_filter_Q}\n"
            + f"number of bits at the output of the filter before final bit-truncation: {self._num_bits_output_pre_Q}\n"
            + f"number of bits at the final output: {self.num_bits_output}\n"
        )

        return string
