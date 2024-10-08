"""
This module implements a new controller for adjusting the gain based on the envelope detection of the input signal.
"""

import warnings
from typing import Any, Optional

import numpy as np

from rockpool.devices.xylo.syns65302.afe.params import (
    AMPLITUDE_THRESHOLDS,
    AUDIO_SAMPLING_RATE_AGC,
    FALL_TIME_CONSTANT,
    MAX_WAITING_TIME_BEFORE_GAIN_CHANGE,
    NUM_BITS_AGC_ADC,
    NUM_BITS_COMMAND,
    PGA_GAIN_INDEX_VARIATION,
    RELIABLE_MAX_HYSTERESIS,
    RISE_TIME_CONSTANT,
    WAITING_TIME_VEC,
)
from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter, State


class EnvelopeController(Module):
    """
    Estimate the envelope of the input signal and uses it to send gain-change commands to PGA.
    """

    def __init__(
        self,
        num_bits: int = NUM_BITS_AGC_ADC,
        amplitude_thresholds: Optional[np.ndarray] = None,
        rise_time_constant: float = RISE_TIME_CONSTANT,
        fall_time_constant: float = FALL_TIME_CONSTANT,
        reliable_max_hysteresis: int = RELIABLE_MAX_HYSTERESIS,
        waiting_time_vec: Optional[np.ndarray] = None,
        max_waiting_time_before_gain_change: float = MAX_WAITING_TIME_BEFORE_GAIN_CHANGE,
        pga_gain_index_variation: Optional[np.ndarray] = None,
        num_bits_command: int = NUM_BITS_COMMAND,
        fs: float = AUDIO_SAMPLING_RATE_AGC,
    ):
        """
        Args:
            num_bits (int): number of bits in the input signal (input signal should be in signed format). Defaults to 14.
            amplitude_thresholds (np.ndarray, optional): sequence of amplitude thresholds specifying the amplitude regions of the signal envelope. Defaults to AMPLITUDE_THRESHOLDS

            rise_time_constant (float, optional): reaction time-constant of the envelope detection when the signal is rising. Defaults to RISE_TIME_CONSTANT = 0.1e-3.
            fall_time_constant (float, optional): reaction time-constant of the envelope detection when the signal is falling. Defaults to FALL_TIME_CONSTANT = 300e-3.
            NOTE: a rise time-constant of `T = 0.1e-3` means that the rise bandwidth is `1/(2 pi x T) = 1600 Hz`. Please pay attention to factor `2 pi` in this equation.

            reliable_max_hysteresis (int, optional): this parameter specify how much rise in maximum envelope is needed before a new maximum (thus, a new context) is identified.
            waiting_time_vec (np.ndarray, optional): this parameter specifies how much waiting time (in seconds) is needed before varying the gain. Defaults to a square-root pattern with higher gains/amplitudes having larger waiting times. Defaults to WAITING_TIME_VEC
            pga_gain_index_variation (np.ndarray, optional): how much the pga_gain_index should be varied (increases, decreased, kept fixed) to track the signal amplitude. Defaults to -1: saturation, 0,0: 2 levels below saturation, +1: remaining regions. Defaults to PGA_GAIN_INDEX_VARIATION
            num_bits_command (int, optional): number of bits used for sending commands to PGA. This also sets number of various gains.
            fs (float): sampling or clock rate of the module.
        """
        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)
        # number of bits in the input signal and also the command sent to PGA
        self.num_bits = SimulationParameter(num_bits, shape=())
        self.num_bits_command = SimulationParameter(num_bits_command, shape=())

        # * set the amplitude levels
        # largest one   -> saturation level
        # smallest one  -> noise level

        # check the validity
        if amplitude_thresholds is None:
            amplitude_thresholds = np.asarray(AMPLITUDE_THRESHOLDS)

        if amplitude_thresholds.dtype != np.int64:
            raise ValueError(
                "all the elements of the amplitude thresholds should be `np.int64` integers!"
            )

        if (
            np.any(amplitude_thresholds[1:] - amplitude_thresholds[:-1] <= 0)
            or np.any(amplitude_thresholds < 0)
            or np.any(amplitude_thresholds >= 2 ** (self.num_bits - 1))
        ):
            raise ValueError(
                f"Amplitude thresholds should be an increasing sequence of length {2**num_bits_command} with elements in the range [0, {2**(self.num_bits-1)-1}]!"
            )

        self.amp_thresholds = SimulationParameter(
            amplitude_thresholds, shape=(2**num_bits_command,)
        )

        # clock rate
        self.fs = SimulationParameter(fs, shape=())

        # * set the rise and fall time constants of the envelope estimator
        # validity check
        if rise_time_constant > fall_time_constant:
            raise ValueError(
                "An invalid design! Envelope estimator circuit needed to have a much smaller rise time constant to allow tracking the signal amplitude faster!\n"
                + "the ratio between fall and rise time constant should be at least 10 in a good design!"
            )

        # number of signal samples received during rise period -> quantize it into a power of two
        # NOTE: we changed it slightly so that the number of fall samples does not change with gain ratio
        # num_rise_samples = rise_time_constant * fs * np.log(gain_ratio)
        num_rise_samples = rise_time_constant * fs
        self.num_rise_samples = int(2 ** np.ceil(np.log2(num_rise_samples)))

        self.rise_time_constant = self.num_rise_samples / fs
        self.rise_avg_bitshift = SimulationParameter(
            int(np.log2(self.num_rise_samples)), shape=()
        )

        # number of signal samples received during fall period -> quantize it into a power of two
        # NOTE: we changed this so that the number of fall samples does not change with gain ratio
        # num_fall_samples = fall_time_constant * fs * np.log(gain_ratio)
        num_fall_samples = fall_time_constant * fs
        self.num_fall_samples = int(2 ** np.ceil(np.log2(num_fall_samples)))
        self.fall_time_constant = self.num_fall_samples / fs
        self.fall_avg_bitshift = SimulationParameter(
            int(np.log2(self.num_fall_samples)), shape=()
        )

        warnings.warn(
            "\n\n"
            + " WARNING ".center(100, "+")
            + "\n"
            + "Number of samples in rise and fall time-constant for envelope controller was set to a power 2 to simplify the implementation!\n"
            + "Please make sure this does not change the AGC performance considerably!\n\n"
            + f"original rise time-constant: {rise_time_constant} sec\n"
            + f"current rise time-constant: {self.rise_time_constant} sec\n"
            + f"window length of the rise window low-pass filter: {int(self.rise_time_constant * self.fs)}\n\n"
            + f"original fall time-constant: {fall_time_constant} sec\n"
            + f"current fall time-constant: {self.fall_time_constant} sec\n"
            + f"window length of the rise window low-pass filter: {int(self.fall_time_constant * self.fs)}.\n"
            + "".center(100, "+")
            + "\n\n"
        )

        # number of additional bits needed to implement the low-pass filters with a good resolution
        self.deadzone_bitshift = max([self.fall_avg_bitshift, self.rise_avg_bitshift])

        # how much rise in maximum is needed before a new envelope sample is announced as the maximum
        self.reliable_max_hysteresis = SimulationParameter(
            reliable_max_hysteresis, shape=()
        )

        # waiting times in sec
        if waiting_time_vec is None:
            waiting_time_vec = WAITING_TIME_VEC

        # waiting times in terms of number of samples
        self.waiting_time_length_vec = SimulationParameter(
            np.array(waiting_time_vec * self.fs, dtype=np.int64),
            shape=(2**self.num_bits_command,),
        )

        # maximum waiting time before gain change
        self.max_num_samples_in_waiting_time = SimulationParameter(
            int(max_waiting_time_before_gain_change * self.fs), shape=()
        )

        # ===========================================================================
        #         Specify how much pga_gain_index needs to be varied
        # ===========================================================================
        # NOTE: this variation simply opportunistic in the sense that it specifies how much jump in amplitude is desired but it may not be fulfilled.
        # For example, suppose signal is very strong and the command is already set to the lowest value b0000 (for 4-bit command).
        # In such a case, if the signal is in saturation mode, we may request a gain decrease by setting `pga_gain_index_variation = -1` but this is not
        # of course fulfilled since the gain is already in its lowest level.
        # * check the validity

        if pga_gain_index_variation is None:
            pga_gain_index_variation = PGA_GAIN_INDEX_VARIATION

        # * check the validity: in the saturation mode we should definitely decrease the gain so the last component should be negative
        if pga_gain_index_variation[-1] >= 0:
            raise ValueError(
                "the last component of the `pga_gain_index_variation` which shows the jump policy between amplitude regions belong to the saturation region\n"
                + "So it should be always negative to decrease the gain immediately!"
            )

        # * check validity: after jump we should be in a valid region
        start_region_index = np.arange(len(pga_gain_index_variation))
        final_region_index = start_region_index + pga_gain_index_variation

        if np.min(final_region_index) < 0 or np.max(final_region_index) > len(
            pga_gain_index_variation
        ):
            raise ValueError(
                f"`pga_gain_index_variation` is not a valid jump vector between amplitude regions since it may trigger jumps to an invalid amplitude index (valid range: [0, {len(pga_gain_index_variation)})!"
            )

        if np.any(np.abs(pga_gain_index_variation) > 3):
            idx = np.where(np.abs(pga_gain_index_variation) > 3)
            raise ValueError(
                f"3 bits (signed) are allocated for each `pga_gain_index_variation` element! "
                + f"idx = {idx[0]} is out of limits! pga_gain_index_variation[{idx[0]}] = {pga_gain_index_variation[idx]} "
            )

        self.pga_gain_index_variation = SimulationParameter(
            pga_gain_index_variation, shape=(2**self.num_bits_command + 1,)
        )

        # ===========================================================================
        #                 create and initialize state variables
        # ===========================================================================

        # envelope estimation parameters
        self.high_res_envelope = State(0, init_func=lambda _: 0, shape=())
        self.envelope = State(0, init_func=lambda _: 0, shape=())
        self.num_processed_samples = State(0, init_func=lambda _: 0, shape=())

        # maximum envelope values
        self.max_envelope = State(0, init_func=lambda _: 0, shape=())
        self.registered_max_envelope = State(0, init_func=lambda _: 0, shape=())

        # parameters of the command transferred to PGA
        # NOTE: we use the same name `pga_gain_index == command` since sometimes pga_gain_index is easy to recall
        self.pga_gain_index = State(0, init_func=lambda _: 0, shape=())
        self.command = self.pga_gain_index

        # amplitude level of the envelope signal
        self.amp_index = State(0, init_func=lambda _: 0, shape=())

        # waiting time until the next gain adjustment
        self.waiting_time = State(
            self.waiting_time_length_vec[0],
            init_func=lambda _: self.waiting_time_length_vec[0],
            shape=(),
        )

        # number of samples processed during a waiting time
        # NOTE: this is needed since the waiting time may keep extending
        self.num_samples_in_waiting_time = State(0, init_func=lambda _: 0, shape=())

    def evolve(self, sig_in: int, record: bool = False):
        """this module updates the state of envelope estimator based on the input signal.

        Args:
            sig_in (int): input signal.
            record (bool, optional): record the state during the simulation. Defaults to False.
        """

        # NOTE: if PGA is not settled after gain change, its output signal will be unstable.
        # In such a case, we model the output of PGA by None.
        # ADC and other modules should stop working until a valid sample arrives.
        # We model this by generating None for the ADC output so that the next layers also skip this unstable period.

        if sig_in is None:
            # this should not happen as the ADC should repeat the samples and not propagate None received from amplifier!
            raise ValueError(
                "this should not happen as the ADC should repeat the samples when a None is received from the amplifier in an unstable mode!"
            )

        # add to the number of processed sampled
        self.num_processed_samples += 1

        # check the amplitude of the input signal
        if type(sig_in) != int and type(sig_in) != np.int64:
            raise ValueError("input quantized sample should be of integer type!")

        if sig_in >= 2 ** (self.num_bits - 1) or sig_in < -(2 ** (self.num_bits - 1)):
            raise ValueError(
                f"Overflow Error: Input signal does not fit in the input number of bits: {self.num_bits}!"
            )

        # compute the absolute value of the signal for envelope estimation
        sig_in = abs(sig_in)

        # add additional bitshifts to have a better precision
        sig_in_high_res = sig_in << self.deadzone_bitshift

        # ===========================================================================
        #     see if the diode gets activated and choose rise or fall mode
        # ===========================================================================
        if sig_in_high_res >= self.high_res_envelope:
            # rise mode
            self.high_res_envelope = (
                self.high_res_envelope
                - (self.high_res_envelope >> self.rise_avg_bitshift)
                + (sig_in_high_res >> self.rise_avg_bitshift)
            )

        else:
            # fall mode
            self.high_res_envelope = (
                self.high_res_envelope
                - (self.high_res_envelope >> self.fall_avg_bitshift)
                + (sig_in_high_res >> self.fall_avg_bitshift)
            )

        # check possible overflow
        if self.high_res_envelope >= 2 ** (self.num_bits - 1 + self.deadzone_bitshift):
            raise ValueError("Overflow Error: In envelope estimation!")

        # compute the envelope of the signal
        self.envelope = self.high_res_envelope >> self.deadzone_bitshift

        # ===========================================================================
        # *        compute the max envelope as new envelopes are calculated
        # ===========================================================================
        if self.envelope > self.max_envelope:
            self.max_envelope = self.envelope

        # compute the index of the max envelope at this point of time
        max_envelope_index = np.sum(self.amp_thresholds < self.max_envelope)

        # ===========================================================================
        # *                            record the state
        # ===========================================================================
        # register the state if needed:
        # NOTE: that we register the state one-clock earlier since the new states with be calculated in this clock
        # and will appear in the next clock
        if record:
            __rec = {"sig_in_high_res": sig_in_high_res}
        else:
            __rec = {}

        # * because of the one-clock delay to the output we record the past gain and gain index values
        pga_gain_index_to_be_sent_out = self.pga_gain_index

        # NOTE: from this point on we need to check saturation and non-saturation regions

        # ===========================================================================
        # *                                Urgent
        # *                    first check the saturation region
        # *                                 or
        # *       the regions that have a negative `pga_gain_index_variation`
        # ===========================================================================
        # NOTE: the first condition is not needed because we have verified that the last component of `pga_gain_index_variation`
        # corresponding to the saturation region is always negative!
        if (
            max_envelope_index == len(self.amp_thresholds)
            or self.pga_gain_index_variation[max_envelope_index] < 0
        ):
            # step 1: reduce the gain according to the jump pattern set in the design
            # * new version:
            # * jump the amplitude region depending on the jump size set in the design
            # * default: go down by -1 in saturation mode

            # update the PGA gain index
            self.pga_gain_index += self.pga_gain_index_variation[max_envelope_index]

            if self.pga_gain_index >= 2**self.num_bits_command:
                self.pga_gain_index = 2**self.num_bits_command - 1

            if self.pga_gain_index < 0:
                self.pga_gain_index = 0

            self.command = self.pga_gain_index

            # update max_envelope_index
            # NOTE: we use the same jump pattern as in pga_gain_index with the difference that in `updated_max_envelope_index` after jump we should be in a valid region
            updated_max_envelope_index = (
                max_envelope_index + self.pga_gain_index_variation[max_envelope_index]
            )

            if (
                updated_max_envelope_index >= (len(self.amp_thresholds) + 1)
                or updated_max_envelope_index < 0
            ):
                raise ValueError(
                    "the maximum envelope index after jump should be valid index! This might be caused by a flow in the AGC jump pattern!"
                )

            # step 2: since envelope drops very slowly, update the envelope by setting it to a lower value than the satureation level
            self.high_res_envelope = (
                self.amp_thresholds[updated_max_envelope_index]
                - self.reliable_max_hysteresis
            ) << self.deadzone_bitshift
            self.max_envelope = 0
            self.registered_max_envelope = 0

            # setp 3: adjust the waiting time and reset the number of samples processed so far
            self.waiting_time = self.waiting_time_length_vec[updated_max_envelope_index]
            self.num_samples_in_waiting_time = 0

            # send the gain command to the PGA
            return pga_gain_index_to_be_sent_out, self.state(), __rec

        # ===========================================================================
        # *                               Otherwise:
        # *                        if not in saturation region
        # *                                   and
        # *                   no immediate pga gain reduction needed
        # ===========================================================================
        # * register the reliable maximum and extend the waiting time accordingly
        if (
            self.max_envelope
            >= self.registered_max_envelope + self.reliable_max_hysteresis
        ):
            # register the reliable max envelope
            self.registered_max_envelope = self.max_envelope

            # extend the waiting time further to search more to find higher max envelope values
            # NOTE (1): it seems there was a bug in old version where we used `max_envelope_index - 1` whereas we should have used the true index of the region, i.e., `max_enevelope_index`
            # NOTE (2): we don't have any waiting time for saturation region since as soon as we detect it we take actions to reduce the gain.
            #           As a result of this policy, `max_envelope_index` can never be equal to `len(self.amp_thresholds) = len(self.waiting_time_length_vec) = #amplitudd regions - 1`
            #           So the following indexing is valid.
            self.waiting_time = self.waiting_time_length_vec[max_envelope_index]

        else:
            # just reduce the waiting time since the envelope/amplitude has not increases significantly
            self.waiting_time -= 1

        # total number of samples processed during waiting time starting from the previous gain change
        self.num_samples_in_waiting_time += 1

        # ===========================================================================
        # *  check if the waiting time is over and it is the time to adjust the gain
        # ===========================================================================

        # we have waited too long and we need to make a decision on gain adjustemt
        if (
            self.num_samples_in_waiting_time == self.max_num_samples_in_waiting_time
            or self.waiting_time == 0
        ):
            # * new version
            # * find out what is the next amplitude region to be in
            # NOTE: this condition is checked/arrived only when the `pga_gain_index_variation` is positive, i.e., gain increase is needed
            # In such a case, the envelope estimation method is quite fast so we don't need
            self.pga_gain_index += self.pga_gain_index_variation[max_envelope_index]

            if self.pga_gain_index >= 2**self.num_bits_command:
                self.pga_gain_index = 2**self.num_bits_command - 1

            if self.pga_gain_index < 0:
                self.pga_gain_index = 0

            self.command = self.pga_gain_index
            # * END new version

            # set the waiting time
            self.waiting_time = self.waiting_time_length_vec[max_envelope_index]

            # reset the number of samples processed within the past waiting times
            self.num_samples_in_waiting_time = 0

            # adjust the value of maximum envelope to re-estimate it according to the new envelope that is going to be computed soon
            self.max_envelope = 0
            self.registered_max_envelope = 0

        # send the gain command to the PGA
        return pga_gain_index_to_be_sent_out, self.state(), __rec
