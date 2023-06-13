# -----------------------------------------------------------
# This module implements a simple model of the amplifier as a high-pass module
# with possibility to switch the amplitudes abruptly.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 11.04.2023
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from typing import Any
import warnings


# ===========================================================================
# *    some constants defined according to Xylo-A3 specficiations
# ===========================================================================s
from agc.xylo_a3_agc_specs import (
    AUDIO_SAMPLING_RATE,
    XYLO_MAX_AMP,
    AUDIO_SAMPLING_RATE,
    SETTLING_TIME_PGA,
    HIGH_PASS_CORNER,
    LOW_PASS_CORNER,
    EXP_PGA_GAIN_VEC,
    DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
)


class Amplifier:
    def __init__(
        self,
        high_pass_corner: float = HIGH_PASS_CORNER,
        low_pass_corner: float = LOW_PASS_CORNER,
        max_audio_amplitude: float = XYLO_MAX_AMP,
        pga_gain_vec: np.ndarray = EXP_PGA_GAIN_VEC,
        settling_time: float = SETTLING_TIME_PGA,
        fixed_gain_for_PGA_mode: bool = False,
        pga_command_in_fixed_gain_for_PGA_mode: int = DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
        fs: float = AUDIO_SAMPLING_RATE,
    ):
        """this class simulates the effect of amplifier on the audio signal when there is an abrupt gain change.
        NOTE: We assume that amplifier is like a high-pass module due to AC coupling.
        Args:
            high_pass_corner (float, optional): the corner frequency of the high-pass DC coupling.
            low_pass_corner (float, optional): the corner frequency of the low-pass response of the amplifier.
            max_audio_amplitude (float, optional): maximum possible amplitude of the signal within the chip.
            pga_gain_vec (np.ndarray, optional): a set of gains that are possible in amplifier. Defaults to array of size 16 as designed in envelope controller.
            settling_time (float, opt): how much time does it take for PGA gain to settle to its final value after the gain change command is received.
            fixed_gain_for_PGA_mode (bool, optional): flag showing if the gain of pga needs to be frozen. Defaults to False in AGC mode.
            pga_command_in_fixed_gain_for_PGA_mode (int, optional): which gain index should be used as the default one in the fixed gain mode of PGA.
            fs (float): sampling rate of the audio signal.
        """
        # make sure that the simulation rate of the analog filter part of the amplifier is correct
        if fs < 4.0 * AUDIO_SAMPLING_RATE:
            warnings.warn(
                "\n\n"
                + " WARNING ".center(100, "+")
                + "\n"
                + "The fixed amplifier within the amplifier module uses a 2nd order ODE to model the high-pass AC coupling and low-pass anti-aliasing filtering.\n"
                + "To make sure that this simulation is done precisely, the sampling rate (here, the simulation rate), should be at least 4 times\n"
                + "larger than the audio sampling rate that is fed into the amplifier.\n\n"
                + f"The target audio sampling rate is {AUDIO_SAMPLING_RATE} sample/sec.\n\n"
                + "If you are using the amplifier module for another signal with less sampling rate such that `fs > 4 x signal sample rate`, `fs` is sufficient enough\n"
                + "and the simulation will be done quite precisely.\n"
                + "So, in such cases, you can ignore this warning!\n"
                + "".center(100, "+")
                + "\n\n"
            )

        self.high_pass_corner = high_pass_corner
        self.low_pass_corner = low_pass_corner
        self.max_audio_amplitude = max_audio_amplitude
        self.settling_time = settling_time
        self.fs = fs

        # NOTE: for precision reason it is always better to run the amplifier with a higher clock rate
        EPS = 0.000001
        oversampling_factor = fs / AUDIO_SAMPLING_RATE

        if (
            np.abs(oversampling_factor - np.round(oversampling_factor))
            / np.min([oversampling_factor, np.round(oversampling_factor)])
            > EPS
        ):
            warnings.warn(
                """
                The oversampling factor (ratio between simulation clock and Xylo-A3 audio sampling rate) of the amplifier is better to be an integer factor!
                
                NOTE: This is not obligatory and the amplifier module may be run with an arbitrary clock as user wishes! 
                But then the user should be careful to match the clock rate between various modules. 
                Otherwise, the signal may go under frequency scaling (e.g., higher frequencies shifted to lower ones and vice versa) while passing through the
                modules.
                """
            )

        self.oversampling_factor = np.round(oversampling_factor)

        self.pga_gain_vec = np.copy(pga_gain_vec)

        # * see if PGA gain is frozen
        self.fixed_gain_for_PGA_mode = fixed_gain_for_PGA_mode

        # check the validity of the command in fixed-gain mode
        if not (
            0 <= pga_command_in_fixed_gain_for_PGA_mode <= len(self.pga_gain_vec) - 1
        ):
            raise ValueError(
                f"Invalid PGA command in the fixed-gain mode: it should be in the range [0, {len(self.pga_gain_vec)-1}]!"
            )

        self.pga_command_in_fixed_gain_for_PGA_mode = (
            pga_command_in_fixed_gain_for_PGA_mode
        )

        # ==================================================
        # * build a simple filter model of the amplifier:
        #   low-pass module followed by the high-pass one
        # ==================================================
        #
        #               `low-pass cutoff`    `high-pass AC coupling`
        #
        #                                             | |
        #  input audio --------/\/\/\-----------------| |-----------   amplifier output
        #                                |            | |          |
        #                                |                         |
        #                              -----                       \
        #                              -----                       /
        #                                |                         \
        #                                |                         /
        #                                |                         |
        #                               ---                       ---
        #                                -                         -

        # low-pass and high-pass filter resistors
        self.r_low = 1
        self.r_high = self.r_low * self.low_pass_corner / self.high_pass_corner

        self.c_high = 1 / (2 * np.pi * self.r_high * self.high_pass_corner)
        self.c_low = 1 / (2 * np.pi * self.r_low * self.low_pass_corner)

        # reset to initialize the state variables
        self.reset()

    def reset(self):
        self.v_c_low = 0.0
        self.v_c_high = 0.0

        # adjust the timing and initialize the module
        self.time_stamp = 0.0
        self.last_pga_command = 0
        self.last_gain_switch_time = -self.settling_time

        # reset the simulation state
        self.state = {}

        # number of input samples recieved since the last reset
        self.num_processed_samples = 0

    def evolve(
        self, audio: float, time_in: float, pga_command: int = 0, record: bool = False
    ):
        """this module takes the input auido signal and also signal from AGC and simulates the behavior of amplifier.

        Args:
            audio (float): input audio sample.
            time_in (float): time instant of the input audio.
            pga_command (int, optional): command for adjusting the gain.
            record (bool, optional): record the state during the simulation. Defaults to False.
        """

        # check if PGA is in frozen-gain mode and if yes ignore the comand received from envelope-controller module.
        if self.fixed_gain_for_PGA_mode:
            pga_command = self.pga_command_in_fixed_gain_for_PGA_mode

        # check the start of the simulation and set the gain values in PGA
        if self.num_processed_samples == 0:
            if record:
                self.state = {
                    "num_processed_samples": [],
                    "time_in": [],
                    "audio_in": [],
                    "fixed_amp_output": [],
                    "pga_output": [],
                    "pga_gain": [],
                    "pga_gain_index": [],
                }
            else:
                self.state = {}

        # increase the number of processed samples
        self.num_processed_samples += 1

        # register the times
        dt, self.time_stamp = time_in - self.time_stamp, time_in

        # check if any gain variation has happened
        if pga_command != self.last_pga_command:
            self.last_pga_command = pga_command
            self.last_gain_switch_time = time_in

        # ================================================
        # *      simulate the first/fixed amplifier
        # ================================================
        # change the order of two circuits
        q_low_res = (audio - self.v_c_low) / self.r_low * dt
        q_high_res = (self.v_c_low - self.v_c_high) / self.r_high * dt

        # variation in voltage due to charge variation
        dv_c_high = q_high_res / self.c_high
        dv_c_low = (q_low_res - q_high_res) / self.c_low

        self.v_c_high += dv_c_high
        self.v_c_low += dv_c_low

        # check the saturation bound due to voltage power supply
        self.v_c_low = (
            self.v_c_low
            if np.abs(self.v_c_low) <= self.max_audio_amplitude
            else np.sign(self.v_c_low) * self.max_audio_amplitude
        )

        fixed_amp_output = self.v_c_low - self.v_c_high
        fixed_amp_output = (
            fixed_amp_output
            if np.abs(fixed_amp_output) <= self.max_audio_amplitude
            else np.sign(fixed_amp_output) * self.max_audio_amplitude
        )

        # ================================================
        # *  simulate the effect of PGA and gain switch
        # ================================================

        # check the saturation due to power supply voltage at the output of PGA
        pga_gain = self.pga_gain_vec[self.last_pga_command]
        pga_output = pga_gain * fixed_amp_output
        pga_output = (
            pga_output
            if np.abs(pga_output) <= self.max_audio_amplitude
            else np.sign(pga_output) * self.max_audio_amplitude
        )

        # return None if still in the transition mode and not settled down
        pga_output = (
            pga_output
            if (time_in - self.last_gain_switch_time) >= self.settling_time
            else None
        )

        if pga_output is None:
            # return immediately: do not record the state in the unstable phase.
            return pga_output

        # record the state
        if record:
            self.state["num_processed_samples"].append(self.num_processed_samples)
            self.state["time_in"].append(time_in)
            self.state["audio_in"].append(audio)
            self.state["fixed_amp_output"].append(fixed_amp_output)
            self.state["pga_output"].append(pga_output)
            self.state["pga_gain"].append(pga_gain)
            self.state["pga_gain_index"].append(pga_command)

        return pga_output

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        this function is the same as evolve.
        """
        return self.evolve(*args, **kwargs)

    def __repr__(self) -> str:
        # string representation of the amplifier module
        string = (
            "+" * 100
            + "\n"
            + "Amplifier module consisting of fixed and programmable amplifier (PGA):\n"
            + f"simulation clock rate: {self.fs}\n"
            + f"target audio sample rate: {AUDIO_SAMPLING_RATE} Hz\n"
            + f"over sampling factor used for more precise simulation: {self.oversampling_factor}\n"
            + f"fixed amplifier with normalized gain 1 and bandwidth: [{self.high_pass_corner}, {self.low_pass_corner}] Hz\n"
            + f"PGA with gain vector:\n{self.pga_gain_vec}\n"
        )

        return string
