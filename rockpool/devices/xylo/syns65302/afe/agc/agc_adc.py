"""
This file implements the analog input path for XyloAudio 3, which starts from an analog microphone and consists of several preprocessing modules.

    - `PGA (programmable gain amplifier)`: This gain of the amplifier can be controlled by the 4-bit command sent from `envelope controller` module. 
        So using PGA we can have one of the following gains $g_1, g_2, \dots, g_{16}$. In our design, we set $g_1=1$ and $g_{16}=64$ giving a total maximum gain of 36 dB. 
    - `ADC (anaolg to digital converter)`: this module takes the signal amplified with PGA and quantizes it into `num_bits = 10` bits. 
        So the signed output can be in the range $-2^{\text{num-bits - 1}} = -512, \dots, 2^\text{num-bits - 1}-1 = 511$. 
    - `EC (envelope controller)`: This module observes the `num-bits=10` bit quantized signal from ADC and makes processing as we explain in a moment to decide if the gain needs to be increased or decreased and how it should happen. 
        The output of this module is a 4-bit command that informs PGA and other modules of what is the best gain that should be selected for the next time steps.
    - `GS (gain smoother)`: One of the problems with AGC is that the PGA part has to be implemented in the analog domain. 
        As a result, we cannot vary the desired gain very smoothly. In particular, when the gain changes from some $g_i$ to some $g_j$ the output undergoes a jump of size $\frac{g_j}{g_i} - 1$. 
        This sudden jump in gain may create transient effects in the filters in the filterbank following AGC. 
        To solve this issue, we have added the gain smoother module, which makes sure that the gain transition from $g_i$ to $g_j$ happens smoothly in time so that the transient effect is not problematic.

"""

from copy import copy
from typing import Dict, Optional, Tuple

import numpy as np

from rockpool.devices.xylo.syns65302.transform.resample import ResampleAudio
from rockpool.devices.xylo.syns65302.afe.agc.adc import ADC
from rockpool.devices.xylo.syns65302.afe.agc.amplifier import Amplifier
from rockpool.devices.xylo.syns65302.afe.agc.envelope_controller import (
    EnvelopeController,
)
from rockpool.devices.xylo.syns65302.afe.agc.gain_smoother import GainSmoother
from rockpool.devices.xylo.syns65302.afe.params import (
    AMPLITUDE_THRESHOLDS,
    AUDIO_SAMPLING_RATE_AGC,
    DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
    EXP_PGA_GAIN_VEC,
    FALL_TIME_CONSTANT,
    HIGH_PASS_CORNER,
    LOW_PASS_CORNER,
    NUM_BITS_AGC_ADC,
    NUM_BITS_COMMAND,
    NUM_BITS_GAIN_QUANTIZATION,
    PGA_GAIN_INDEX_VARIATION,
    RELIABLE_MAX_HYSTERESIS,
    RISE_TIME_CONSTANT,
    WAITING_TIME_VEC,
    XYLO_MAX_AMP,
)
from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter, State

__all__ = ["AGCADC"]


class AGCADC(Module):
    """
    Automatic Gain Controller Analog-to-Digital (ADC) module for XyloAudio 3 chip consisting of

        (i)   Programmable gain amplifier (PGA), used for signal amplification.
        (ii)  ADC module, used for quantizing the PGA output.
        (iii) EnvelopeController module, used for detecting the envelope of the signal and adjusting it based on the gain-commands. It sends to PGA to adjust the gain.
        (iv)  GainSmoother, optional module, used to smooth out the gain to avoid gain jumps
    """

    def __init__(
        self,
        oversampling_factor: int = 2,
        enable_gain_smoother: bool = True,
        fixed_gain_for_PGA_mode: bool = False,
        fixed_pga_gain_index: int = DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
        pga_gain_index_variation: Optional[np.ndarray] = None,
        ec_amplitude_thresholds: Optional[np.ndarray] = None,
        ec_waiting_time_vec: Optional[np.ndarray] = None,
        ec_rise_time_constant: int = RISE_TIME_CONSTANT,
        ec_fall_time_constant: int = FALL_TIME_CONSTANT,
        ec_reliable_max_hysteresis: int = RELIABLE_MAX_HYSTERESIS,
        num_bits_gain_quantization: int = NUM_BITS_GAIN_QUANTIZATION,
        fs: float = AUDIO_SAMPLING_RATE_AGC,
    ) -> None:
        """
        Args:
            oversampling_factor (int, optional): oversampling factor of the high-rate ADC. Defaults to 2.
                Warning! Only `1`, `2` and `4` are feasible values regarding the current implementation.
                Sets `AGC_CTRL1.AAF_OS_MODE` register (2**N)
            enable_gain_smoother (bool, optional): Enables the gain smoother. Defaults to True.
                Sets `AGC_CTRL1.GS_DIACT` bit
            fixed_gain_for_PGA_mode (bool, optional): When it's True, it effectively by-passes `Envelope Controller` and provides a fixed gain command to the amplifier. Defaults to False.
                Sets `AGC_CTRL1.PGA_GAIN_BYPASS` register
            fixed_pga_gain_index (int, optional): Which gain index should be used as the default one in the fixed gain mode of PGA, effective when `fixed_gain_for_PGA_mode=True`. Defaults to None.
                Warning! Values between [0-15] (inclusive) are feasible regarding the current implementation.
                Sets `AGC_CTRL1.PGA_GAIN_IDX_CFG` register
            pga_gain_index_variation (Optional[np.ndarray], optional): Sets how much the `pga_gain_index` should be varied (increases, decreased, kept fixed) to track the signal amplitude.
                Defaults to -1: saturation, 0,0: 2 levels below saturation, +1: remaining regions, check PGA_GAIN_INDEX_VARIATION. Defaults to PGA_GAIN_INDEX_VARIATION.
                Sets AGC_PGIV_REG0 + AGC_PGIV_REG1 + AGC_CTRL2.PGIV16 (3 bits each,signed) registers
            ec_amplitude_thresholds (Optional[np.ndarray], optional): sequence of amplitude thresholds specifying the amplitude regions of the signal envelope in the `Envelope Controller`. Defaults to AMPLITUDE_THRESHOLDS.
                Sets `AGC_AT_REG0 - AGC_AT_REG7` [10 bit x2 each]. Two thresholds fit in one register§
            ec_waiting_time_vec (Optional[np.ndarray], optional): Specify how much waiting time (in seconds) is needed before varying the gain. Defaults to a square-root pattern with higher gains/amplitudes having larger waiting times. Defaults to WAITING_TIME_VEC.
                Sets `AGC_WT0 - AGC_WT15` registers, 24 bits each.
                Indirectly sets 24 bits `AGC_CTRL3.MAX_NUM_SAMPLE` since `max_waiting_time_before_gain_change = max(ec_waiting_time_vec)`
                Indirectly sets `AGC_CTRL2.AVG_BITSHIFT` since `gain_smoother.min_waiting_time=min(ec_waiting_time_vec)`
            ec_rise_time_constant (int, optional): Reaction time-constant of the envelope detection when the signal is rising. Defaults to RISE_TIME_CONSTANT = 0.1e-3.
                Sets `AGC_CTRL1.RISE_AVG_BITSHIFT` register, 5 bits
            ec_fall_time_constant (int, optional): Reaction time-constant of the envelope detection when the signal is falling. Defaults to FALL_TIME_CONSTANT = 300e-3.
                Sets `AGC_CTRL1.FALL_AVG_BITSHIFT` register, 5 bits
            ec_reliable_max_hysteresis (int, optional): Specify how much rise in maximum envelope is needed before a new maximum (thus, a new context) is identified.. Defaults to RELIABLE_MAX_HYSTERESIS.
                Sets `AGC_CTRL2.RELI_MAX_HYSTR` register, 10 bits
            num_bits_gain_quantization (int, optional): Number of bits used for quantizing the gain ratios, effective only when `enable_gain_smoother = True`. Defaults to NUM_BITS_GAIN_QUANTIZATION.
                Sets `AGC_CTRL2.NUM_BITS_GAIN_FRACTION` (4 bits)
            fs (float, optional): Target output sampling or clock rate of the module. Defaults to AUDIO_SAMPLING_RATE_AGC.
        """
        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)

        self.fs = fs

        ## - Set default arrays
        if ec_amplitude_thresholds is None:
            ec_amplitude_thresholds = AMPLITUDE_THRESHOLDS
        if ec_waiting_time_vec is None:
            ec_waiting_time_vec = WAITING_TIME_VEC
        if pga_gain_index_variation is None:
            pga_gain_index_variation = PGA_GAIN_INDEX_VARIATION

        # (ii) ADC
        self.adc = ADC(
            num_bits=NUM_BITS_AGC_ADC,
            max_audio_amplitude=XYLO_MAX_AMP,
            oversampling_factor=oversampling_factor,
            fs=self.fs,
        )

        self.oversampled_fs = SimulationParameter(self.adc.oversampled_fs, shape=())
        """Sampling rate for the ADC front and the Amplifier"""

        # (0) Resampling
        self.resample = ResampleAudio(fs_target=self.oversampled_fs)

        # (i) Amplifier
        self.amplifier = Amplifier(
            high_pass_corner=HIGH_PASS_CORNER,
            low_pass_corner=LOW_PASS_CORNER,
            max_audio_amplitude=XYLO_MAX_AMP,
            pga_gain_vec=EXP_PGA_GAIN_VEC,
            fixed_gain_for_PGA_mode=fixed_gain_for_PGA_mode,
            pga_command_in_fixed_gain_for_PGA_mode=fixed_pga_gain_index,
            fs=self.adc.oversampled_fs,
        )

        # (iii) Envelope Controller
        self.envelope_controller = EnvelopeController(
            num_bits=NUM_BITS_AGC_ADC,
            amplitude_thresholds=ec_amplitude_thresholds,
            rise_time_constant=ec_rise_time_constant,
            fall_time_constant=ec_fall_time_constant,
            reliable_max_hysteresis=ec_reliable_max_hysteresis,
            waiting_time_vec=ec_waiting_time_vec,
            max_waiting_time_before_gain_change=max(ec_waiting_time_vec),
            pga_gain_index_variation=pga_gain_index_variation,
            num_bits_command=NUM_BITS_COMMAND,
            fs=self.fs,
        )

        # (iv) GainSmoother
        if enable_gain_smoother:
            self.gain_smoother = GainSmoother(
                num_bits=NUM_BITS_AGC_ADC,
                min_waiting_time=min(ec_waiting_time_vec),
                num_bits_command=NUM_BITS_COMMAND,
                pga_gain_vec=EXP_PGA_GAIN_VEC,
                num_bits_gain_quantization=num_bits_gain_quantization,
                fs=self.fs,
            )
        else:
            self.gain_smoother = None

        # - State variables
        self.agc_pga_command = State(0, init_func=lambda _: 0, shape=())

    def reset_state(self) -> None:
        """
        Overrides the module state reset method to reset the states of all the sub-modules
        """
        super().reset_state()
        self.amplifier.reset_state()
        self.adc.reset_state()
        self.envelope_controller.reset_state()

        if self.gain_smoother is not None:
            self.gain_smoother.reset_state()

    def state(self) -> Dict[str, dict]:
        """
        Override the module state to include the states of all the sub-modules
        """
        # accumulate all the states from all modules and return it
        __state = {
            "self": super().state(),
            "amplifier": self.amplifier.state(),
            "adc": self.adc.state(),
            "envelope_controller": self.envelope_controller.state(),
        }

        # check if there is any gain smoother
        if self.gain_smoother is not None:
            __state["gain_smoother"] = self.gain_smoother.state()

        return __state

    def evolve(
        self, audio_in: Tuple[np.ndarray, float], record: bool = False
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Takes a input audio signal and simulates the behavior of the analog input path of XyloAudio 3 chip.

        Args:
            audio_in (Tuple[np.ndarray, float]): A tuple containing the input audio signal and its sampling rate.
            record (bool, optional): A flag to record the intermediate results of the sub-modules. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict, Dict]:
                output audio signal,
                current state of the module,
                record dictionary containing the intermediate results of the sub-modules.
        """
        audio, _, _ = self.resample(audio_in)

        # Prepare the record dictionary
        if record:
            __rec = {
                "amplifier": [],
                "pga_gain": [],
                "envelope_controller": [],
                "envelope": [],
            }
            if self.gain_smoother is not None:
                __rec["adc"] = []

        else:
            __rec = {}

        sig_out = []

        for sig_in in audio:
            # The old value of agc_pga_command computed in the past clock is used to produce amplifier output and ADC output
            __out_amp, _, __rec_amp = self.amplifier.evolve(
                audio=sig_in,
                pga_command=self.agc_pga_command,
                record=record,
            )

            __out, _, _ = self.adc.evolve(sig_in=__out_amp, record=record)
            if __out is None:
                continue
                # ATTENTION! We'll wait till the ADC is ready to produce the output.
                # Oversampling == 1 : ADC generates output every clock
                # Oversampling == 2 : ADC generates output every other clock
                # Oversampling == 4 : ADC generates output every 4th clock

            self.agc_pga_command, __state_envelope, _ = self.envelope_controller.evolve(
                sig_in=__out, record=record
            )

            if record:
                __rec["amplifier"].append(copy(__out_amp))
                __rec["pga_gain"].append(copy(__rec_amp["pga_gain"]))
                __rec["envelope_controller"].append(copy(self.agc_pga_command))
                __rec["envelope"].append(copy(__state_envelope["envelope"]))

            if self.gain_smoother is not None:
                if record:
                    __rec["adc"].append(copy(__out))

                __out, _, _ = self.gain_smoother.evolve(
                    audio=__out,
                    pga_gain_index=self.agc_pga_command,
                    record=record,
                )

            sig_out.append(__out)

        # Convert the output list to numpy array
        sig_out = np.asarray(sig_out)

        # Add resampled signal if rec is true
        if record:
            __rec["resampled_signal"] = audio

        return sig_out, self.state(), __rec
