# -----------------------------------------------------------
# This module implements a new controller for adjusting the gain based on the
# envelope detection of the input signal.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 14.06.2023
# -----------------------------------------------------------


import numpy as np
from typing import Any
import warnings


# ===========================================================================
# *    some constants defined according to Xylo-A3 specficiations
# ===========================================================================
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.xylo_a3_agc_specs import (
    NUM_BITS_ADC,
    AUDIO_SAMPLING_RATE,
    AMPLITUDE_THRESHOLDS,
    NUM_BITS_COMMAND,
    RISE_TIME_CONSTANT,
    FALL_TIME_CONSTANT,
    WAITING_TIME_VEC,
    RELIABLE_MAX_HYSTERESIS,
    PGA_GAIN_INDEX_VARIATION,
    MAX_WAITING_TIME_BEFORE_GAIN_CHANGE
)


class EnvelopeController:
    def __init__(
        self,
        num_bits: int = NUM_BITS_ADC,
        amplitude_thresholds: np.ndarray = AMPLITUDE_THRESHOLDS,
        rise_time_constant: float = RISE_TIME_CONSTANT,
        fall_time_constant: float = FALL_TIME_CONSTANT,
        reliable_max_hysteresis: int = RELIABLE_MAX_HYSTERESIS,
        waiting_time_vec: np.ndarray = WAITING_TIME_VEC,
        max_waiting_time_before_gain_change = MAX_WAITING_TIME_BEFORE_GAIN_CHANGE,
        pga_gain_index_variation: np.ndarray = PGA_GAIN_INDEX_VARIATION,
        num_bits_command: int = NUM_BITS_COMMAND,
        fs: float = AUDIO_SAMPLING_RATE,
    ):
        """this module estimates the envelope of the input signal and uses it to send gain-change commands to PGA.

        Args:
            num_bits (int): number of bits in the input signal (input signal should be in signed format). Defaults to 14.
            amplitude_thresholds (np.ndarray, optional): sequence of amplitude thresholds specifying the amplitude regions of the signal envelope.

            rise_time_constant (float, optional): reaction time-constant of the envelope detection when the signal is rising. Defaults to RISE_TIME_CONSTANT = 0.1e-3.
            fall_time_constant (float, optional): reaction time-constant of the envelope detection when the signal is falling. Defaults to FALL_TIME_CONSTANT = 300e-3.
            NOTE: a rise time-constant of `T = 0.1e-3` means that the rise bandwidth is `1/(2 pi x T) = 1600 Hz`. Please pay attention to factor `2 pi` in this equation.

            reliable_max_hysteresis (int, optional): this parameter specify how much rise in maximum envelope is needed before a new maximum (thus, a new context) is identified.
            waiting_time_vec (np.ndarray, oprional): this parameter specifies how much waiting time (in seconds) is needed before varying the gain. Defaults to a square-root pattern with higher gains/amplitudes having larger waiting times.
            pga_gain_index_variation (np.ndarray, optional): how much the pga_gain_index should be varied (increases, decreased, kept fixed) to track the signal amplitude. Defaults to -1: saturation, 0,0: 2 levels below saturation, +1: remaining regions.
            num_bits_command (int, optional): number of bits used for sending commands to PGA. This also sets number of various gains.
            fs (float): sampling or clock rate of the module.
        """
        # number of bits in the input signal and also the command sent to PGA
        self.num_bits = num_bits
        self.num_bits_command = num_bits_command

        # * set the amplitude levels
        # largest one   -> saturation level
        # smallest one  -> noise level

        # check the validity
        amplitude_thresholds = np.asarray(amplitude_thresholds)

        if amplitude_thresholds.dtype != np.int64:
            raise ValueError(
                "all the elements of the amplitude thresholds should be integers!"
            )

        if (
            len(amplitude_thresholds) != 2**num_bits_command
            or np.any(amplitude_thresholds[1:] - amplitude_thresholds[:-1] <= 0)
            or np.any(amplitude_thresholds < 0)
            or np.any(amplitude_thresholds >= 2 ** (self.num_bits - 1))
        ):
            raise ValueError(
                f"Amplitude thresholds should be an increasing sequenceue of length {2**num_bits_command} with elements in the range [0, {2**(self.num_bits-1)-1}]!"
            )

        self.amp_thresholds = amplitude_thresholds

        # clock rate
        self.fs = fs

        # * set the rise and fall time constants of the envelope estimator
        # validity check
        if rise_time_constant > fall_time_constant:
            raise ValueError(
                "An invalid design! Envelope estimator circuit needed to have a much smaller rise time constant to allow tracking the signal amplitude faster!\n"
                + "the ratio between fall and rise time constant should be at least 10 in a good design!"
            )

        # number of signal samples received during rise period -> quantize it into a power of two
        # NOTE: we changed it slightly so that the number of fall samples does not chnage with gain ratio
        # num_rise_samples = rise_time_constant * fs * np.log(gain_ratio)
        num_rise_samples = rise_time_constant * fs
        self.num_rise_samples = int(2 ** np.ceil(np.log2(num_rise_samples)))

        self.rise_time_constant = self.num_rise_samples / fs
        self.rise_avg_bitshift = int(np.log2(self.num_rise_samples))

        # number of signal samples received during fall period -> quantize it into a power of two
        # NOTE: we changed this so that the number of fall samples does not change with gain ratio
        # num_fall_samples = fall_time_constant * fs * np.log(gain_ratio)
        num_fall_samples = fall_time_constant * fs
        self.num_fall_samples = int(2 ** np.ceil(np.log2(num_fall_samples)))

        self.fall_time_constant = self.num_fall_samples / fs
        self.fall_avg_bitshift = int(np.log2(self.num_fall_samples))

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
        self.reliable_max_hysteresis = reliable_max_hysteresis

        # ===========================================================================
        #           specify the waiting times at various gain levels
        # ===========================================================================
        # * check the validity
        if len(waiting_time_vec) != 2**self.num_bits_command:
            raise ValueError(
                f"length of the waiting-time sequence should be {2**self.num_bits_command}!\nNOTE: in saturation amplitude level, we have no waiting time."
                + "So there are {2**self.num_bits_command} amplitude levels left!"
            )

        # waiting times in sec
        self.waiting_time_vec = waiting_time_vec

        # waiting times in terms of number of samples
        self.waiting_time_length_vec = (self.waiting_time_vec * self.fs).astype(
            np.int64
        )
        
        # maximum waiting time before gain change
        self.max_num_samples_in_waiting_time = int(max_waiting_time_before_gain_change * self.fs)

        # ===========================================================================
        #         Specify how much pga_gain_index needs to be varied
        # ===========================================================================
        # NOTE: this variation simply oppurtunistic in the sense that it sepecifies how much jump in amplitude is desired but it may not be fulfilled.
        # For example, suppose signal is very strong and the command is already set to the lowest value b0000 (for 4-bit command).
        # In such a case, if the signal is in saturation mode, we may request a gain decrease by setting `pga_gain_index_variation = -1` but this is not
        # of course fulfilled since the gain is already in its lowest level.
        # * check the validity
        if len(pga_gain_index_variation) != 2**self.num_bits_command + 1:
            raise ValueError(
                f"Since there are {2**self.num_bits_command + 1} amplitude level regions (including the saturation region), {2**self.num_bits_command + 1} PGA gain index variations are needed!\n"
                + "Note that these parameters specify the jump pattern between various amplitude regions!\n"
                + "For example, the defaults mode [+1, +1, ..., 0, 0, -1] implies that in saturation mode the region should go down by 1 step whereas in low-amplitude regions it should go up by 1.\n"
                + "Also, we set 0 for the two amplitude regions below the saturation to stop signal entering the saturation immediately after it starts increasing!"
            )

        # * check the validity: in the saturation mode we should definitely decrease the gain so the last component should be negative
        if pga_gain_index_variation[-1] >= 0:
            raise ValueError(
                "the last component of the `pga_gain_index_variation` which shows the jump policy betweeb amplitude regions belong to the saturation region\n"
                + "So it should be always negative to decrease the gain immediately!"
            )

        # * check validity: after jump we should be in a valid region
        start_region_index = np.arange(len(pga_gain_index_variation))
        final_region_index = start_region_index + pga_gain_index_variation

        if np.min(final_region_index) < 0 or np.max(final_region_index) > len(
            pga_gain_index_variation
        ):
            raise ValueError(
                f"`pga_gain_index_variation` is not a valid jump vector beteween amplitude regions since it may trigger jumps to an invalid amplitude index (valid range: [0, {len(pga_gain_index_variation)})!"
            )

        self.pga_gain_index_variation = np.copy(pga_gain_index_variation)

        # ===========================================================================
        #                 create and initialize simulation variables
        # ===========================================================================
        self.reset()

    def reset(self):
        # envelope estimation parameters
        self.high_res_envelope = 0
        self.envelope = 0
        self.num_processed_samples = 0

        # maximum envelope values
        self.max_envelope = 0
        self.registered_max_envelope = 0

        # parameters of the command transferred to PGA
        # NOTE: we use the same name `pga_gain_index == command` since sometimes pga_gain_index is easy to recall
        self.pga_gain_index = 0
        self.command = self.pga_gain_index

        # amplitude level of the envelope signal
        self.amp_index = 0

        # waiting time until the next gain adjustment
        self.waiting_time = self.waiting_time_length_vec[0]

        # number of samples processed during a waiting time
        # NOTE: this is needed since the waiting time may keep extending
        self.num_samples_in_waiting_time = 0

        # reset the state
        self.state = {}

    def evolve(self, sig_in: int, time_in: float, record: bool = False):
        """this module updates the state of envelope estimator based on the input signal.

        Args:
            sig_in (int): input signal.
            time_in (float): time of the input signal.
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

        # check if it is the start of the simulation and set the state
        if self.num_processed_samples == 0:
            if record:
                self.state = {
                    "num_processed_samples": [],
                    "time_in": [],
                    "sig_in": [],
                    "sig_in_high_res": [],
                    "high_res_envelope": [],
                    "envelope": [],
                    "max_envelope": [],
                    "max_envelope_index": [],
                    "registered_max_envelope": [],
                    "registered_max_envelope_index": [],
                    "waiting_time": [],
                    "total_waiting_after_latest_change": [],
                    "pga_gain_index": [],
                }
            else:
                self.state = {}

        # add to the number of processed sampled
        self.num_processed_samples += 1

        # check the amplitude of the input signal
        if type(sig_in) != int and type(sig_in)!= np.int64:
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
        # regiser the state if needed:
        # NOTE: that we register the state one-clock earlier since the new states with be calculated in this clock
        # and will appear in the next clock
        if record:
            self.state["num_processed_samples"].append(self.num_processed_samples)
            self.state["time_in"].append(time_in)
            self.state["sig_in"].append(sig_in)
            self.state["sig_in_high_res"].append(sig_in_high_res)
            self.state["high_res_envelope"].append(self.high_res_envelope)
            self.state["envelope"].append(self.envelope)
            self.state["max_envelope"].append(self.max_envelope)
            self.state["max_envelope_index"].append(max_envelope_index)
            self.state["registered_max_envelope"].append(self.registered_max_envelope)
            self.state["registered_max_envelope_index"].append(
                np.sum(self.amp_thresholds <= self.registered_max_envelope)
            )
            self.state["waiting_time"].append(self.waiting_time)
            self.state["total_waiting_after_latest_change"].append(
                self.num_samples_in_waiting_time
            )

            # record the gain at this clock
            # NOTE: if there is any change to gain, it will be applied in the next clock after latch to make sure that a clean
            # gain-index goes to the PGA
            self.state["pga_gain_index"].append(self.pga_gain_index)

        # * becuase of the one-clock delay to the output we record the past gain and gain index values
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
                    "the maximum enevlope index after jump should be valid index! This might be caused by a flow in the AGC jump pattern!"
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
            return (
                pga_gain_index_to_be_sent_out,
                self.envelope,
            )

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
        return (pga_gain_index_to_be_sent_out, self.envelope)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        this is the same as evolve function.
        """
        return self.evolve(*args, **kwargs)

    def __repr__(self) -> str:
        # string representation of the envelope controller module
        string = (
            "+" * 100
            + "\n"
            + "Envelope Controller module:\n"
            + f"clock rate: {self.fs}\n"
            + f"number of bits in the input coming from ADC: {self.num_bits}\n"
            + f"number of bits used for sending gain-change command to PGA: {self.num_bits_command}\n"
            + f"rise time-constant: {self.rise_time_constant}\n"
            + f"number of samples in rise time-constant: {self.num_rise_samples}\n"
            + f"fall time-constant: {self.fall_time_constant}\n"
            + f"number of samples in fall time-constant: {self.num_fall_samples}\n"
            + f"amplitude thresholds: {self.amp_thresholds}\n"
            + f"waiting times in various amplitude levels: {self.waiting_time_length_vec}\n"
            + f"maximum waiting time: {self.max_num_samples_in_waiting_time}\n"
        )

        return string
