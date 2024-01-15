"""
This module implements a simple model of the amplifier as a high-pass module with possibility 
to switch the amplitudes abruptly if its gain changes.
"""

import numpy as np

from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_SAMPLING_RATE_AGC,
    DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
    EXP_PGA_GAIN_VEC,
    HIGH_PASS_CORNER,
    LOW_PASS_CORNER,
    NUM_BITS_COMMAND,
    XYLO_MAX_AMP,
)
from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter, State

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
        fixed_gain_for_PGA_mode: bool = False,
        pga_command_in_fixed_gain_for_PGA_mode: int = DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
        fs: float = AUDIO_SAMPLING_RATE_AGC,
    ) -> None:
        """
        Args:
            high_pass_corner (float, optional): the corner frequency of the high-pass DC coupling.
            low_pass_corner (float, optional): the corner frequency of the low-pass response of the amplifier.
            max_audio_amplitude (float, optional): maximum possible amplitude of the signal within the chip.
            pga_gain_vec (np.ndarray, optional): a set of gains that are possible in amplifier. Defaults to array of size 16 as designed in envelope controller.
            fixed_gain_for_PGA_mode (bool, optional): flag showing if the gain of pga needs to be frozen. Defaults to False in AGC mode.
            pga_command_in_fixed_gain_for_PGA_mode (int, optional): which gain index should be used as the default one in the fixed gain mode of PGA.
            fs (float): sampling or clock rate of the module.
        """

        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)

        self.high_pass_corner = SimulationParameter(high_pass_corner, shape=())
        self.low_pass_corner = SimulationParameter(low_pass_corner, shape=())
        self.max_audio_amplitude = SimulationParameter(max_audio_amplitude, shape=())

        # NOTE: for precision reason it is always better to run the amplifier with a higher clock rate
        if pga_gain_vec is None:
            self.pga_gain_vec = EXP_PGA_GAIN_VEC
        else:
            self.pga_gain_vec = SimulationParameter(
                pga_gain_vec, shape=(2**NUM_BITS_COMMAND,)
            )

        # See if PGA gain is frozen
        self.fixed_gain_for_PGA_mode = SimulationParameter(
            fixed_gain_for_PGA_mode, shape=()
        )

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

        if fs <= 0:
            raise ValueError(
                f"`fs` has to be a positive number, {fs} is given instead!"
            )

        self.dt = SimulationParameter(1 / fs, shape=())

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
        self.last_pga_command = State(0, init_func=lambda: 0, shape=())

    def evolve(self, audio: float, pga_command: int = 0, record: bool = False):
        """Takes the input audio signal and also signal from AGC and simulates the behavior of amplifier.

        Args:
            audio (float): input audio sample.
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

        self.last_pga_command = pga_command

        # - Simulate the first/fixed amplifier
        # change the order of two circuits
        q_low_res = (audio - self.v_c_low) / self.r_low * self.dt
        q_high_res = (self.v_c_low - self.v_c_high) / self.r_high * self.dt

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

        # - The effect of PGA and gain switch
        # check the saturation due to power supply voltage at the output of PGA
        pga_gain = self.pga_gain_vec[self.last_pga_command]
        pga_output = pga_gain * fixed_amp_output
        pga_output = (
            pga_output
            if np.abs(pga_output) <= self.max_audio_amplitude
            else np.sign(pga_output) * self.max_audio_amplitude
        )

        # record the state
        if record:
            __rec = {
                "fixed_amp_output": fixed_amp_output,
                "pga_gain": pga_gain,
                "pga_gain_index": pga_command,
            }
        else:
            __rec = {}

        return pga_output, self.state(), __rec
