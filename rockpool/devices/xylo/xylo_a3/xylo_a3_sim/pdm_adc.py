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
#
# last update: 27.06.2023
# -----------------------------------------------------------


# - Rockpool imports
from multiprocessing.sharedctypes import Value
from rockpool.nn.modules.module import Module
from rockpool.parameters import SimulationParameter
from rockpool.nn.combinators import Sequential
from rockpool.timeseries import TSContinuous

# - Other imports
# import deltasigma as ds

import numpy as np
import scipy.signal as sp


import warnings
from logging import info

from typing import Union, Tuple, Any
from numbers import Number

from rockpool.typehints import P_int, P_float


# list of modules exported
__all__ = ["PDM_ADC", "PDM_Microphone", "PolyPhaseFIR_DecimationFilter", "DeltaSigma"]


##------------------------------------------------------##
## design parameters currently used in the Xylo-A3 chip ##
##------------------------------------------------------##
SYSTEM_CLOCK_RATE = 50_000_000  # 50 MHz

AUDIO_SAMPLING_RATE = SYSTEM_CLOCK_RATE / (64 * 16)
AUDIO_CUTOFF_FREQUENCY = 20_000
AUDIO_CUTOFF_FREQUENCY_WIDTH = 0.2 * AUDIO_CUTOFF_FREQUENCY

PDM_FILTER_DECIMATION_FACTOR = 32
PDM_SAMPLING_RATE = AUDIO_SAMPLING_RATE * PDM_FILTER_DECIMATION_FACTOR

DELTA_SIGMA_ORDER = 4

DECIMATION_FILTER_LENGTH = 256
NUM_BITS_FILTER_Q = 16
NUM_BITS_ADC = 14


##---------------------------------------------------##


class DeltaSigma:
    def __init__(
        self,
        amplitude: float = 1.0,
        bandwidth: float = AUDIO_CUTOFF_FREQUENCY,
        order: int = DELTA_SIGMA_ORDER,
        fs: float = PDM_SAMPLING_RATE,
    ):
        """this class implements a simple deltasigma modulation module.
        NOTE: this class is going to replace the `deltasigma` library which is not supported anymore.

        Args:
            amplitude (float, optional): maximum amplitude of the input signal. Defaults to 1.0 to obtain +1, -1 as the output.
            bandwidth (float, optional): target bandwidth of the input signal. Defaults to AUDIO_CUTOFF_FREQUENCY = 20K in Xylo-A3.
            order (int, optional): order of deltasigma module. Defaults to DELTA_SIGMA_ORDER = 4 in Xylo-A3.
            fs (float, optional): sampling rate of deltasigma module. Defaults to PDM_SAMPLING_RATE = 1.6 M in Xylo-A3.
        """
        self.amplitude = amplitude
        self.bandwidth = bandwidth

        if fs <= 2 * bandwidth:
            raise ValueError(
                "sampling rate of deltasigma module should be much larger than the bandwidth of the signal!"
            )

        self.fs = fs
        self.order = order

        # * build a low-pass filter with the given order
        # butterworth filter
        filter_order = self.order
        filter_cutoff = self.bandwidth
        filter_type = "lowpass"

        b, a = sp.butter(
            N=filter_order,
            Wn=2 * np.pi * filter_cutoff,
            btype=filter_type,
            analog=True,
            output="ba",
        )

        # reverse the coefficients to convert them into recursive feedback format
        b = b[::-1]
        a = a[::-1]

        # dimension of the state
        self.dim_state = len(a) - 1

        # * convert this into block-diagram format
        # compute the state space representation
        A = np.diag(np.ones(self.dim_state - 1), -1)
        A_Q = -a[:-1]

        B = np.zeros(self.dim_state)
        B[0] = b[0]

        C = np.zeros(self.dim_state)
        C[-1] = 1

        # * apply state normalization for better numerical stability
        # NOTE: this normalization is needed since for a signal at frequency f0, its first and second derivative are scaled by f0 and f0^2 and so on.
        # So without proper normalization the simulation may be numerically unstable!
        self.norm_factor = self.fs
        N = np.diag(1 / self.norm_factor ** np.arange(self.dim_state - 1, -1, step=-1))

        self.A = N @ A @ np.linalg.inv(N)
        self.A_Q = N @ A_Q
        self.B = N @ B
        self.C = np.linalg.inv(N) @ C

    def evolve(
        self,
        sig_in: np.ndarray,
        sample_rate: float = None,
        python_version: bool = False,
        record: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """this module processes the input signal and produces its corresponding sigmadelta modulation via simulating the corrresponding ODE.

        Args:
            sig_in (np.ndarray): input signal.
            sample_rate (float): sample rate of the input signal. Defaults to None.
            python_version (bool, optional): ttthis flag forces deltasigma computation to be done in Python without any Jax speedup. Defaults to False.
            record (bool, optional): record the states of the filter. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: a tuple containing arrays corresponding to deltasigma +1, -1 signal, pre-quantization signal,
                                                                    possibly interpolated high-rate signal used for simuation, and states of the filter.

        """

        # * validate the amplitude
        if np.max(np.abs(sig_in)) > self.amplitude:
            raise ValueError(
                "the amplitude of the input signal is larger than the target amplitude for sigmadelta module! This may results in wrong modulation or divergence of the module!"
            )

        # * convert the sample rate of the signal if needed
        if sample_rate is None:
            sample_rate = self.fs

        # signal needs resampling?
        if sample_rate != self.fs:
            time_in = np.arange(len(sig_in)) / sample_rate

            duartion = (len(sig_in) - 1) / sample_rate
            time_target = np.arange(0, duartion, step=1 / self.fs)
            sig_in_resampled = np.interp(time_target, time_in, sig_in)

            # replace the original signal
            sig_in = sig_in_resampled

        # check if Jax is available for speedup and if it is permitted
        if JAX_DeltaSigma and not python_version:
            sig_out_Q, sig_out, state_list = jax_deltasigma_evolve(
                sig_in=sig_in,
                fs=self.fs,
                A=self.A,
                A_Q=self.A_Q,
                B=self.B,
                C=self.C,
                amplitude=self.amplitude,
                record=record,
            )

            return sig_out_Q, sig_out, sig_in, state_list

        # otherwise: proceed with the python versiooon
        state_list = []

        # states for the simulation
        state = np.zeros(self.dim_state)

        sig_out = []
        sig_out_Q = []

        sigmadelta_out = 0
        sigmadelta_out_Q = 0

        for sig in sig_in:
            # differential of the state
            d_state = (
                np.einsum("ij, j -> i", self.A, state)
                + self.B * sig
                + self.A_Q * sigmadelta_out_Q
            )

            # update the state
            state += d_state * 1 / self.fs

            if record:
                state_list.append(np.copy(state))

            # compute the output
            sigmadelta_out = np.sum(self.C * state)
            sigmadelta_out_Q = self.amplitude * (
                2 * np.heaviside(sigmadelta_out, 0) - 1
            )

            sig_out.append(sigmadelta_out)
            sig_out_Q.append(sigmadelta_out_Q)

        sig_out = np.asarray(sig_out)
        sig_out_Q = np.asarray(sig_out_Q)
        state_list = np.asarray(state_list)

        return sig_out_Q, sig_out, sig_in, state_list

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # this is the same as `evolve` function.
        return self.evolve(*args, **kwargs)

    def recover(self, bin_in: np.ndarray) -> np.ndarray:
        """this module takes the binary encoded signal and recovers the original signal via low-pass filtering and decimation.

        NOTE: here we are applying a simple low-pass filter to recover the signal.
        This is used only for sanity check and in practice more involved filters may be applied to obtain a better recovery.

        Args:
            bin_in (np.ndarray): binary input signal obtained from deltasigma modulation.

        Returns:
            np.ndarray: array containing the recovered signal from binary deltasigma output.
        """

        # * build a low-pass filter with the given order
        filter_order = self.order
        filter_cutoff = self.bandwidth
        filter_type = "lowpass"

        b, a = sp.butter(
            N=filter_order,
            Wn=filter_cutoff,
            btype=filter_type,
            analog=False,
            output="ba",
            fs=self.fs,
        )

        # * filter the binary signal and decimate it
        sig_rec = sp.lfilter(b, a, bin_in)

        oversampling = int(self.fs / self.bandwidth)
        sig_rec_dec = sig_rec[::oversampling]

        # return the high-res and low-res (decimated) version of the recovered signal
        return sig_rec, sig_rec_dec

    def validate(self, sig_in: np.ndarray, bin_in: np.ndarray) -> bool:
        """
        this module investigates if the generated deltasigma encoding is valid or not.

        NOTE: this is used as a sanity check since in practice the simulation of deltasigma in block-diagram format may diverge!

        Args:
            sig_in (np.ndarray): input signal.
            bin_in (np.ndarray): binary bitstream obtain from deltasigma modulation of the input signal.

        Returns:
            bool: returns True if the deltasigma modulation is not diverged.
        """

        # compute the mean values of the signam
        mean_sig_pos = np.mean(sig_in + np.abs(sig_in)) / 2
        mean_sig_neg = np.mean(np.abs(sig_in) - sig_in) / 2

        mean_sig_sum = mean_sig_pos + mean_sig_neg

        mean_sig_neg /= mean_sig_sum
        mean_sig_pos /= mean_sig_sum

        mean_bin_pos = np.mean(bin_in + np.abs(bin_in)) / 2
        mean_bin_neg = np.mean(np.abs(bin_in) + bin_in) / 2

        mean_bin_sum = mean_bin_pos + mean_bin_neg

        mean_bin_pos /= mean_bin_sum
        mean_bin_neg /= mean_bin_sum

        # measure the relative error
        rel_err = np.max(
            [abs(mean_sig_pos - mean_bin_pos), abs(mean_sig_neg - mean_bin_neg)]
        )

        # threshold for the relative error
        EPS = 0.2

        return rel_err < EPS


class PDM_Microphone(Module):
    """
    This class simulates a PDM microphone which applies sigma-delta modulation on the input audio signal

    The input to microphone is an analog audio signal and the output is a PDM bit-stream in which the relative
    frequency of 1-vs-0 depends on the instanteneous amplitude of the signal.
    """

    def __init__(
        self,
        sdm_order: int = DELTA_SIGMA_ORDER,
        sdm_OSR: int = PDM_FILTER_DECIMATION_FACTOR,
        bandwidth: float = AUDIO_CUTOFF_FREQUENCY,
        fs: float = PDM_SAMPLING_RATE,
    ):
        """
        Initialise a PDM_Microphone module

        Args:
            sdm_order (int): order of the sigma-delta modulator (conventional ones are 2 or 3). Defaults to DELTA_SIGMA_ORDER.
            sdm_OSR (int): oversampling rate in sigma-delta modulator. Defaults to PDM_FILTER_DECIMATION_FACTOR.
            fs (int): rate of the clock used for deriving the PDM microphone.
                    NOTE: PDM microphone can be derived by various clock rates. By changing the clock rate and sdm_OSR we can
                    keep the sampling rate of the audio fixed.
        """
        super().__init__(shape=1, spiking_output=True)

        # sigma-delta modulator parameters
        self.sdm_order: P_int = SimulationParameter(sdm_order)
        """ int: order of the sigma-delta modulator """

        self.sdm_OSR: P_int = SimulationParameter(sdm_OSR)
        """ int: oversampling rate in sigma-delta modulator """

        self.fs: P_float = SimulationParameter(fs)
        """ float: Sampling frequency of the module in Hz """

        target_audio_sample_rate = self.fs / self.sdm_OSR
        if bandwidth > target_audio_sample_rate / 2.0:
            raise ValueError(
                f"PDM microphone with clock rate {self.fs} and oversampling factor {self.sdm_OSR} is targeted to process audio with sample rate {target_audio_sample_rate}.\n"
                + f"Therefore its deltasigma analog circuits should have a badnwidth less than half target audio sample rate, i.e., < {target_audio_sample_rate/2} Hz.\n"
            )

        self.bandwidth: P_float = SimulationParameter(bandwidth)

        # build the sigmal-delta module
        # self._sdm_module = ds.synthesizeNTF(self.sdm_order, self.sdm_OSR, opt=1)
        self._sdm_module = DeltaSigma(
            amplitude=1.0,
            bandwidth=self.bandwidth,
            order=self.sdm_order,
            fs=self.fs,
        )

    def evolve(
        self,
        audio_in: Tuple[np.ndarray, float],
        record: bool = False,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
            This function takes the input audio signal and produces a PDM bit-stream

            NOTE: In reality the input signal to sigma-delta modulator in PDM microphone is the analog audio signal. In simulation, however, we have to still use a sampled version of this analog signal as representative.

            NOTE: the audio signal should be normalized to the valid range of sigma-delta modulator [-1.0, 1.0]. If not in this range, clipping should be applied manually to limit the signal into this range.
            If the signal amplitue is very close to +1 or -1, there is a higher chance that the block-diagram version we use to simulate the signal diverges. In such cases, it is better to reduce the signal amplitude slightly.

            Resampling is performed if the sample rate of the input signal is less that the clock rate of PDM bitstream.

        Args:
            audio_in (Tuple[np.ndarray), float]: a tuple containing the input audio signal and its sampling rate.
            record (bool): record the inner states of the deltasigma module used for PDM modulation.

        Raises:
            ValueError: if the amplitude is not scaled properly and is not in the valid range [-1.0, 1.0]

        Returns:
            np.ndarray: array containing PDM bit-stream at the output of the microphone.
        """

        try:
            audio, sample_rate = audio_in

            if isinstance(audio, np.ndarray) and isinstance(sample_rate, Number):
                pass
        except:
            raise TypeError(
                "`audio_in` should be a tuple consisting of a numpy array containing the audio and its sample rate!"
            )

        if audio.ndim != 1:
            raise ValueError(
                "only single-channel audio signals can be processed by the sigma-delta modulator in PDM microphone!"
            )

        if np.max(np.abs(audio)) > 1.0:
            raise ValueError(
                "Some of the signal samples have an amplitude larger than 1.0.\n"
                + "Sigma-delta modulator is designed to work with signal values normalized in the range [-1.0, 1.0].\n"
                + "Normalize the signal or clip it to the range [-1.0, 1.0] manually before applying it to PDM microhpne."
            )

        if sample_rate != self.fs:
            warnings.warn(
                "\n\n"
                + " warnings ".center(120, "+")
                + "\n"
                + f"In practice, the input to the PDM microphone (fed by a clock of rate:{self.fs}) is the analog audio signal.\n"
                + "In simulations, however, we have to use sampled audio signal at the input to mimic this analog signal.\n"
                + f"Here we resample the input auido (of course artificially) to the higher sample rate of PDM mcirophone ({self.fs}).\n"
                + "For a more realistic simulation, it is better to provide an audio signal which is originally sampled with a higher rate.\n"
                + "+" * 120
                + "\n\n"
            )

        # compute the sigma-delta modulation
        (
            audio_pdm,
            audio_pdm_pre_Q,
            sig_resampled,
            deltasigma_filter_states,
        ) = self._sdm_module.evolve(
            sig_in=audio, sample_rate=sample_rate, record=record
        )

        if record:
            recording = {
                "deltasigma_signal_pre_Q": audio_pdm_pre_Q,
                "deltasigma_filter_states": deltasigma_filter_states,
            }
        else:
            recording = {}

        # validate the signal to make sure that it is reasonably converged
        if not self._sdm_module.validate(sig_in=sig_resampled, bin_in=audio_pdm):
            warnings.warn(
                "It seems that the simulator used for deltasigma modulation has not converged properly!\n"
                + "To solve this issue, try reducing the amplitude of the signal. Also try feeding the input with a little bit of noise, which may help the convergence!\n"
            )

        # use the integer format for the final {-1,+1}-valued binary PDM data
        if audio_pdm.dtype != np.int64:
            audio_pdm = audio_pdm.astype(np.int64)

        unique_vals = set(np.unique(audio_pdm))
        if unique_vals != {-1, 1} and unique_vals != {1} and unique_vals != {-1}:
            raise ValueError(
                "The output of sigma-delta modulator should be a {-1,+1}-valued signal.\n"
                + "This problem may arise when the sigma-delta simulator is unstable!\n"
                + "To solve this issue, try reducing the amplitude of the signal. Also try feeding the input with a little bit of noise, which may help the convergence!\n"
            )

        return audio_pdm, self.state(), recording

    def __setattr__(self, name: str, val: Any):
        # - Use super-class setattr to assign attribute
        super().__setattr__(name, val)

        # - Re-generate SDM module
        if not self._in_Module_init and name != "_sdm_module":
            self._sdm_module = DeltaSigma(
                amplitude=1.0,
                bandwidth=self.bandwidth,
                order=self.sdm_order,
                fs=self.fs,
            )

    def _info(self) -> str:
        string = (
            "This is the module for simulating PDM microphone which uses sigma-delta modulation.\n"
            + "The input analog audio signal is mapped to a binary stream of modulated data which is them interpolated to recover a sampled version of the analog input.\n"
            + "Parameters:\n"
            + f"Sigma-Delta modulation order: {self.sdm_order}\n"
            + f"Sigma-Delta oversampling rate (ratio between the rate of PDM clock and target audio sampling rate): {self.sdm_OSR}\n"
            + f"Sigma-Delta clock rate: {self.fs}\n"
        )
        return string


class PolyPhaseFIR_DecimationFilter(Module):
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
        num_bits_output: int = NUM_BITS_ADC,
        fs: float = PDM_SAMPLING_RATE,
    ):
        """
        Initialise a PolyPhaseFIR_DecimationFilter module

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
        # After low-pass filteirng the PDM bit-stream, one obtains the pre-quantization/pre-truncation signal.
        # Enough number of bits are needed to avoid any over and underflow in the filter.
        # Also, some of the LSB's in the pre-truncation output need to be dropped to obtain the final sampled audio with
        # targetted number of bits.
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

        # warnings.warn(
        #     "filter cutoff-width was unset in the implementation to futher lower-down the tail response!"
        # )

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
        # check if the input siganl is binary
        unique_values = set(np.unique(sig_in))
        if unique_values != {1, -1} and unique_values != {1} and unique_values != {-1}:
            raise ValueError(
                "Input signal should be binary-valued (PDM signal) in anti-podal format in {-1, +1}!\n"
                + "Convert to integer in case needed to make sure that the signal is indeed binary with -1 or +1 values!"
            )

    # some utility functions needed for testing and verifiaction

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

        # scipy.signal.freqz(b, a=1, worN=512, whole=False, plot=None, fs=6.283185307179586, include_nyquist=False)
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
        plt.title("specifications of the FIR filter used in polyphase implementtaion")

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


def PDM_ADC(
    sdm_order: int = DELTA_SIGMA_ORDER,
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
        sdm_order (int, optional): order of sigma-delta modulator. Defaults to DELTA_SIGMA_ORDER.
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
    # two modules of equivalent ADC
    return Sequential(
        PDM_Microphone(
            sdm_order=sdm_order,
            sdm_OSR=sdm_OSR,
            bandwidth=cutoff,
            fs=fs,
        ),
        PolyPhaseFIR_DecimationFilter(
            decimation_factor=sdm_OSR,
            cutoff=cutoff,
            cutoff_width=cutoff_width,
            filt_length=filt_length,
            num_bits_filter_Q=num_bits_filter_Q,
            num_bits_output=num_bits_output,
            fs=fs,
        ),
    )


# ===========================================================================
# *  Jax implementation for further speedup of deltasigma modulation module
# ===========================================================================
try:
    import jax
    import jax.numpy as jnp

    # only jax.float32 version implemented as jax.int32 will not work in the filters due to their large number of bits.
    def jax_deltasigma_evolve(
        sig_in: np.ndarray,
        A: np.ndarray,
        A_Q: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        amplitude: np.ndarray,
        fs: float,
        record: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """this module implements the jax version of deltasigma modulator for further speedup.

        Args:
            sig_in (np.ndarray): input signal
            A (np.ndarray): A matrix in block-diagram version.
            A_Q (np.ndarray): A_Q vector in block-diagram version.
            B (np.ndarray): B vector in block-diagramm version.
            C (np.ndarray): C vector in block-diagram version.
            amplitude (np.ndarray): amplitude of the antipodal output of deltasigma modulator.
            record (bool, optional): record the inner states of deltasigma. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: a tuple containing the deltasigma output after quantizatio and before quenatization, and the inner states of the deltasigma module.
        """
        # dimension of the state
        state_dim = A.shape[0]

        state_init = jnp.zeros(state_dim + 1, dtype=jnp.float32)
        A = jnp.asarray(A, dtype=jnp.float32)
        A_Q = jnp.asarray(A_Q, dtype=jnp.float32)
        B = jnp.asarray(B, dtype=jnp.float32)
        C = jnp.asarray(C, dtype=jnp.float32)

        # define the forward function of the dynamics
        def forward(state_in, input):
            # decompse the state and the quantized output
            state, sig_out_Q = state_in[:-1], state_in[-1]

            # compute diffrential of the state
            d_state = A @ state + B * input + A_Q * sig_out_Q

            # update the state
            state += d_state * 1 / fs

            # produce the quantized output
            sig_out = state[-1]
            sig_out_Q = amplitude * (2 * jnp.heaviside(sig_out, 0) - 1)

            # build the output state and output
            state_out = jnp.zeros_like(state_in)
            state_out = state_out.at[:-1].set(state)
            state_out = state_out.at[-1].set(sig_out_Q)

            output = state_out if record else state_out[-2:]

            return state_out, output

        # apply the forward dynamics to compute the deltasigma output
        final_state, output = jax.lax.scan(forward, state_init, sig_in)

        # convert into numpy format for return
        sig_out = np.asarray(output[:, -2], dtype=np.float64)
        sig_out_Q = np.asarray(output[:, -1], dtype=np.float64)

        state = (
            np.asarray(output[:, :-1], dtype=np.float64)
            if record
            else np.asarray([], dtype=np.float64)
        )

        return sig_out_Q, sig_out, state

    # set the  flag for jax version
    JAX_DeltaSigma = True

    info(
        "Jax version was found. DeltaSigma module will be computed using jax speedup.\n"
    )

except ModuleNotFoundError as e:
    info(
        "No jax module was found for DeltaSigma implementation. DeltaSigma module will use python version!\n"
        + str(e)
    )

    # set flag for jax
    JAX_DeltaSigma = False


# - In debug mode deactivate accelerated version
__DEBUG_MODE__ = False

if __DEBUG_MODE__:
    JAX_DeltaSigma = False


if JAX_DeltaSigma:
    # Jax version is active: use jax since it is slightly faster than CPP if all dependencies are ok!
    # apply simple embedding in Python
    info(
        f"JAX_DeltaSigma: {JAX_DeltaSigma}: Using Jax-JIT version of DeltaSigma modulation."
    )

else:
    # use the Python version
    info(f"No Jax version: Using Python native for DeltaSigma modulation.")
