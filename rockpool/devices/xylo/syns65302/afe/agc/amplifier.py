"""
This module implements a simple model of the amplifier as a high-pass module with possibility 
to switch the amplitudes abruptly if its gain changes.
"""

import numpy as np
from typing import Tuple
from rockpool.nn.modules import Module
from rockpool.parameters import State, SimulationParameter

from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_SAMPLING_RATE,
    XYLO_MAX_AMP,
    AUDIO_SAMPLING_RATE,
    SETTLING_TIME_PGA,
    HIGH_PASS_CORNER,
    LOW_PASS_CORNER,
    EXP_PGA_GAIN_VEC,
    DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
    NUM_BITS_COMMAND,
)

__all__ = ["Amplifier"]


class Amplifier(Module):
    """
    Simulate the effect of amplifier on the audio signal when there is an abrupt gain change.
    NOTE: We assume that amplifier is like a high-pass module due to AC coupling.

    A simple filter model of the amplifier:
    low-pass module followed by the high-pass one
    
                   `low-pass cutoff`    `high-pass AC coupling`
    
                                                 | |
      input audio --------/\/\/\-----------------| |-----------   amplifier output
                                    |            | |          |
                                    |                         |
                                  -----                       \
                                  -----                       /
                                    |                         \
                                    |                         /
                                    |                         |
                                   ---                       ---
                                    -                         -
    """

    def __init__(
        self,
        high_pass_corner: float = HIGH_PASS_CORNER,
        low_pass_corner: float = LOW_PASS_CORNER,
        max_audio_amplitude: float = XYLO_MAX_AMP,
        pga_gain_vec: np.ndarray = None,
        settling_time: float = SETTLING_TIME_PGA,
        fixed_gain_for_PGA_mode: bool = False,
        pga_command_in_fixed_gain_for_PGA_mode: int = DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
    ) -> None:
        """
        Args:
            high_pass_corner (float, optional): the corner frequency of the high-pass DC coupling.
            low_pass_corner (float, optional): the corner frequency of the low-pass response of the amplifier.
            max_audio_amplitude (float, optional): maximum possible amplitude of the signal within the chip.
            pga_gain_vec (np.ndarray, optional): a set of gains that are possible in amplifier. Defaults to array of size 16 as designed in envelope controller.
            settling_time (float, opt): how much time does it take for PGA gain to settle to its final value after the gain change command is received.
            fixed_gain_for_PGA_mode (bool, optional): flag showing if the gain of pga needs to be frozen. Defaults to False in AGC mode.
            pga_command_in_fixed_gain_for_PGA_mode (int, optional): which gain index should be used as the default one in the fixed gain mode of PGA.
        """

        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)

        self.high_pass_corner = SimulationParameter(high_pass_corner, shape=())
        self.low_pass_corner = SimulationParameter(low_pass_corner, shape=())
        self.max_audio_amplitude = SimulationParameter(max_audio_amplitude, shape=())
        self.settling_time = SimulationParameter(settling_time, shape=())

        # NOTE: for precision reason it is always better to run the amplifier with a higher clock rate
        if pga_gain_vec is None:
            self.pga_gain_vec = EXP_PGA_GAIN_VEC
        else:
            self.pga_gain_vec = SimulationParameter(
                pga_gain_vec, shape=(2**NUM_BITS_COMMAND,)
            )

        # See if PGA gain is frozen
        self.fixed_gain_for_PGA_mode = fixed_gain_for_PGA_mode

        # check the validity of the command in fixed-gain mode
        if not (
            0 <= pga_command_in_fixed_gain_for_PGA_mode <= len(self.pga_gain_vec) - 1
        ):
            raise ValueError(
                f"Invalid PGA command in the fixed-gain mode: it should be in the range [0, {len(self.pga_gain_vec)-1}]!"
            )

        self.pga_command_in_fixed_gain_for_PGA_mode = SimulationParameter(
            pga_command_in_fixed_gain_for_PGA_mode, shape=()
        )

        # low-pass and high-pass filter resistors
        self.r_low = SimulationParameter(1, shape=())
        self.r_high = SimulationParameter(
            self.r_low * self.low_pass_corner / self.high_pass_corner, shape=()
        )

        self.c_high = SimulationParameter(
            1 / (2 * np.pi * self.r_high * self.high_pass_corner), shape=()
        )
        self.c_low = SimulationParameter(
            1 / (2 * np.pi * self.r_low * self.low_pass_corner), shape=()
        )

        self.v_c_low = State(0.0, init_func=lambda: 0.0, shape=())
        self.v_c_high = State(0.0, init_func=lambda: 0.0, shape=())
        self.time_stamp = State(0.0, init_func=lambda: 0.0, shape=())
        self.last_pga_command = State(0, init_func=lambda: 0, shape=())
        self.last_gain_switch_time = State(
            -self.settling_time, init_func=lambda: -self.settling_time, shape=()
        )
        self.num_processed_samples = State(0, init_func=lambda: 0, shape=())

    def evolve(
        self, audio: float, time_in: float, pga_command: int = 0, record: bool = False
    ):
        """this module takes the input auido signal and also signal from AGC and simulates the behavior of amplifier.

        Args:
            audio (float): input audio sample.
            time_in (float): time instant of the input audio.
            pga_command (int, optional): command for adjusting the gain.
            record (bool, optional): record the state during the simulation. Defaults to False.

        Returns:
            pga_output (float): amplified signal
            state (dict): current state of the module
            rec (dict): record dictionary
        """

        # check if PGA is in frozen-gain mode and if yes ignore the command received from envelope-controller module.
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
