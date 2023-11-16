"""
This file implements the analog input path for Xylo-A3, which starts from an analog microphone and consists of several preprocessing modules.

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
import warnings
from copy import copy
from typing import Dict, Optional, Tuple

import numpy as np

from rockpool.devices.xylo.syns65302.afe.agc.adc import ADC
from rockpool.devices.xylo.syns65302.afe.agc.amplifier import Amplifier
from rockpool.devices.xylo.syns65302.afe.agc.envelope_controller import (
    EnvelopeController,
)
from rockpool.devices.xylo.syns65302.afe.agc.gain_smoother import GainSmootherFPGA
from rockpool.devices.xylo.syns65302.afe.params import (
    AMPLITUDE_THRESHOLDS,
    AUDIO_SAMPLING_RATE,
    DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
    EXP_PGA_GAIN_VEC,
    FALL_TIME_CONSTANT,
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
    Automatic Gain Controller Analog-to-Digital (ADC) module for Xylo-A3 chip consisting of

        (i)   Programmable gain amplifier (PGA), used for signal amplification.
        (ii)  ADC module, used for quantizing the PGA output.
        (iii) EnvelopeController module, used for detecting the envelope of the signal and adjusting it based on the gain-commands. It sends to PGA to adjust the gain.
        (iv)  GainSmootherFPGA, optional module, used to smooth out the gain to avoid gain jumps
    """

    def __init__(
        self,
        oversampling_factor: int = 2,
        enable_gain_smoother: bool = True,
        fixed_pga_gain_index: Optional[float] = None,
        pga_gain_index_variation: Optional[np.ndarray] = None,
        ec_amplitude_thresholds: Optional[np.ndarray] = None,
        ec_waiting_time_vec: Optional[np.ndarray] = None,
        ec_rise_time_constant: int = RISE_TIME_CONSTANT,
        ec_fall_time_constant: int = FALL_TIME_CONSTANT,
        ec_reliable_max_hysteresis: int = RELIABLE_MAX_HYSTERESIS,
        num_bits_gain_quantization=NUM_BITS_GAIN_QUANTIZATION,
        target_fs: float = AUDIO_SAMPLING_RATE,
    ) -> None:
        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)

        self.fs = target_fs

        if fixed_pga_gain_index is None:
            fixed_gain_for_PGA_mode = False
            pga_command_in_fixed_gain_for_PGA_mode = (
                DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE
            )
        else:
            fixed_gain_for_PGA_mode = True
            pga_command_in_fixed_gain_for_PGA_mode = fixed_pga_gain_index

        ## - ADC
        self.adc = ADC(
            oversampling_factor=oversampling_factor,
            max_audio_amplitude=XYLO_MAX_AMP,
            fs=self.fs,
        )

        self.oversampled_fs = SimulationParameter(self.adc.oversampled_fs, shape=())

        ## - Amplifier
        self.amplifier = Amplifier(
            fixed_gain_for_PGA_mode=fixed_gain_for_PGA_mode,
            # PGA_GAIN_BYPASS
            pga_command_in_fixed_gain_for_PGA_mode=pga_command_in_fixed_gain_for_PGA_mode,
            # PGA_GAIN_IDX_CFG # [0-15] -> [1-32]
            max_audio_amplitude=XYLO_MAX_AMP,
            pga_gain_vec=EXP_PGA_GAIN_VEC,
            fs=self.adc.oversampled_fs,
        )

        # AGC_CTRL1.AAF_OS_MODE (2**N)
        # [1-2-4]

        ## - Envelope Controller
        if ec_amplitude_thresholds is None:
            ec_amplitude_thresholds = AMPLITUDE_THRESHOLDS
        if ec_waiting_time_vec is None:
            ec_waiting_time_vec = WAITING_TIME_VEC
        if pga_gain_index_variation is None:
            pga_gain_index_variation = PGA_GAIN_INDEX_VARIATION

        self.envelope_controller = EnvelopeController(
            amplitude_thresholds=ec_amplitude_thresholds,
            # AGC_AT_REG0 - AGC_AT_REG7 [10 bit each]
            # rise_time_constant = ,
            # RISE_AVG_BITSHIFT 5 bits
            rise_time_constant=ec_rise_time_constant,
            # fall_time_constant = ,
            # FALL_AVG_BITSHIFT 5 bits
            fall_time_constant=ec_fall_time_constant,
            reliable_max_hysteresis=ec_reliable_max_hysteresis,
            # AGC_CTRL2.RELI_MAX_HYSTR
            waiting_time_vec=ec_waiting_time_vec,
            # AGC_WT0 - AGC_WT15
            max_waiting_time_before_gain_change=max(ec_waiting_time_vec),
            # AGC_CTRL3.MAX_NUM_SAMPLE
            pga_gain_index_variation=pga_gain_index_variation,
            # AGC_PGIV_REG0 + AGC_PGIV_REG1 + AGC_CTRL3
            # 3 bits (signed or unsigned?)
            fs=self.fs,
        )

        # gain_smoother: GainSmootherFPGA
        if enable_gain_smoother:
            self.gain_smoother = GainSmootherFPGA(
                min_waiting_time=min(ec_waiting_time_vec),
                num_bits_gain_quantization=num_bits_gain_quantization,
                pga_gain_vec=EXP_PGA_GAIN_VEC,
                fs=self.fs,
            )
        else:
            self.gain_smoother = None

        # State variables
        self.agc_pga_command = State(0, init_func=lambda _: 0, shape=())
        # AGC_CTRL2.AVG_BITSHIFT
        # AGC_CTRL2.NUM_BITS_GAIN_FRACTION

        # modules = [self.adc, self.envelope_controller]
        # if self.gain_smoother is not None:
        #     modules.append(self.gain_smoother)

        # oversampling_factors = np.asarray(
        #     [module.fs / AUDIO_SAMPLING_RATE for module in modules]
        # )

        # self.amplifier_simulation_oversampling = self.amplifier.oversampling_factor / (
        #     np.mean(oversampling_factors) * self.adc.oversampling_factor
        # )

    def reset_state(self) -> None:
        self.amplifier.reset_state()
        self.adc.reset_state()
        self.envelope_controller.reset_state()

        # it may happen that we use or not use any gain smoothing
        if self.gain_smoother is not None:
            self.gain_smoother.reset_state()

    def state(self) -> Dict[str, dict]:
        # accumulate all the states from all modules and return it
        __state = {
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
        audio, sample_rate = audio_in
        # try:
        #     audio, sample_rate = audio_in

        #     if isinstance(audio, np.ndarray) and isinstance(sample_rate, (int, float)):
        #         pass
        # except:
        #     raise TypeError(
        #         "`audio_in` should be a tuple consisting of a numpy array containing the audio and its sample rate!"
        #     )

        # if audio.ndim != 1:
        #     raise ValueError(
        #         "only single-channel audio signals can be processed by the deltasigma modulator in PDM microphone!"
        #     )

        if sample_rate != self.oversampled_fs:
            warnings.warn(
                f"Resampling! Sample rate given = {sample_rate}, sample rate required = {self.oversampled_fs}"
            )
            time_in = np.arange(len(audio)) / sample_rate
            duration = (len(audio) - 1) / sample_rate
            time_target = np.arange(0, duration, step=1 / self.oversampled_fs)
            audio_resampled = np.interp(time_target, time_in, audio)

            # replace the original signal
            audio = audio_resampled

        if record:
            __rec = {
                "amplifier": [],
                "adc": [],
                "envelope_controller": [],
            }
            if self.gain_smoother is not None:
                __rec["gain_smoother"] = []

        else:
            __rec = {}

        sig_out = []

        for sig_in in audio:
            # The old value of agc_pga_command computed in the past clock is used to produce amplifier output and ADC output
            __out, amplifier_state, _ = self.amplifier.evolve(
                audio=sig_in,
                pga_command=self.agc_pga_command,
                record=record,
            )
            if record:
                __rec["amplifier"].append(copy(__out))

            # produce the ADC output and register the PGA gain used while ADC was quantizing the signal
            __out, adc_state, _ = self.adc.evolve(sig_in=__out, record=record)
            if __out is None:
                continue
            if record:
                __rec["adc"].append(copy(__out))

            # NOTE: PGA command is updated and sets the PGA gain value which will appear in the next clock
            (self.agc_pga_command, envelope_state, _) = self.envelope_controller.evolve(
                sig_in=__out, record=record
            )

            if record:
                __rec["envelope_controller"].append(copy(self.agc_pga_command))

            # use the ADC out and the command generated by PGA in gain smoother
            if self.gain_smoother is not None:
                (__out, gain_smoother_state, _) = self.gain_smoother.evolve(
                    audio=__out,
                    pga_gain_index=self.agc_pga_command,
                    record=record,
                )
                if record:
                    __rec["gain_smoother"].append(copy(__out))

            sig_out.append(__out)

        __state = {
            "self": self.state(),
            "amplifier": amplifier_state,
            "adc": adc_state,
            "envelope_controller": envelope_state,
        }
        if self.gain_smoother is not None:
            __state["gain_smoother"] = gain_smoother_state

        sig_out = np.asarray(sig_out)

        return sig_out, __state, __rec
