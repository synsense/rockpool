from rockpool.nn.modules import Module
from typing import Tuple, Dict, Optional
import numpy as np
from rockpool.devices.xylo.syns65302.afe.agc.amplifier import Amplifier
from rockpool.devices.xylo.syns65302.afe.agc.adc import ADC
from rockpool.devices.xylo.syns65302.afe.agc.envelope_controller import (
    EnvelopeController,
)
from rockpool.devices.xylo.syns65302.afe.agc.gain_smoother import GainSmootherFPGA

from rockpool.devices.xylo.syns65302.afe.params import (
    DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE,
    AMPLITUDE_THRESHOLDS,
    RELIABLE_MAX_HYSTERESIS,
    WAITING_TIME_VEC,
    PGA_GAIN_INDEX_VARIATION,
    NUM_BITS_GAIN_QUANTIZATION,
    RISE_TIME_CONSTANT,
    FALL_TIME_CONSTANT,
    XYLO_MAX_AMP,
    EXP_PGA_GAIN_VEC,
    AUDIO_SAMPLING_RATE,
)

__all__ = ["AGCADC"]


class AGCADC(Module):
    def __init__(
        self,
        oversampling_factor: int = 1,
        enable_gain_smoother: bool = True,
        fixed_pga_gain_index: Optional[float] = None,
        pga_gain_index_variation: Optional[np.ndarray] = None,
        ec_amplitude_thresholds: Optional[np.ndarray] = None,
        ec_waiting_time_vec: Optional[np.ndarray] = None,
        ec_rise_time_constant: int = RISE_TIME_CONSTANT,
        ec_fall_time_constant: int = FALL_TIME_CONSTANT,
        ec_reliable_max_hysteresis: int = RELIABLE_MAX_HYSTERESIS,
        num_bits_gain_quantization=NUM_BITS_GAIN_QUANTIZATION,
    ) -> None:
        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)

        if fixed_pga_gain_index is None:
            fixed_gain_for_PGA_mode = False
            pga_command_in_fixed_gain_for_PGA_mode = (
                DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE
            )
        else:
            fixed_gain_for_PGA_mode = True
            pga_command_in_fixed_gain_for_PGA_mode = fixed_pga_gain_index

        ## - Amplifier
        self.amplifier = Amplifier(
            fixed_gain_for_PGA_mode=fixed_gain_for_PGA_mode,
            # PGA_GAIN_BYPASS
            pga_command_in_fixed_gain_for_PGA_mode=pga_command_in_fixed_gain_for_PGA_mode,
            # PGA_GAIN_IDX_CFG # [0-15] -> [1-32]
            max_audio_amplitude=XYLO_MAX_AMP,
            pga_gain_vec=EXP_PGA_GAIN_VEC,
            fs=AUDIO_SAMPLING_RATE,
        )

        ## - ADC
        self.adc = ADC(
            oversampling_factor=oversampling_factor,
            max_audio_amplitude=XYLO_MAX_AMP,
            fs=AUDIO_SAMPLING_RATE,
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
            fs=AUDIO_SAMPLING_RATE,
        )

        # gain_smoother: GainSmootherFPGA
        if enable_gain_smoother:
            self.gain_smoother = GainSmootherFPGA(
                min_waiting_time=min(ec_waiting_time_vec),
                num_bits_gain_quantization=num_bits_gain_quantization,
                pga_gain_vec=EXP_PGA_GAIN_VEC,
                fs=AUDIO_SAMPLING_RATE,
            )
        else:
            self.gain_smoother = None
        # AGC_CTRL2.AVG_BITSHIFT
        # AGC_CTRL2.NUM_BITS_GAIN_FRACTION

        __submod_list = []
        # reset all the modules
        self.reset()

    def reset(self) -> None:
        self.amplifier.reset()
        self.adc.reset()
        self.envelope_controller.reset()

        # it may happen that we use or not use any gain smoothing
        if self.gain_smoother is not None:
            self.gain_smoother.reset()

    @property
    def state(self) -> Dict[str, dict]:
        # accumulate all the states from all modules and return it
        modules_inner_state = {
            "amplifier": self.amplifier.state,
            "adc": self.adc.state,
            "envelope_controller": self.envelope_controller.state,
        }

        # check if there is any gain smoother
        if self.gain_smoother is not None:
            modules_inner_state["gain_smoother"] = self.gain_smoother.state

        return modules_inner_state

    def evolve(
        self, audio_in: Tuple[np.ndarray, float], record: bool = False
    ) -> Tuple[np.ndarray, Dict, Dict]:
        try:
            audio, sample_rate = audio_in

            if isinstance(audio, np.ndarray) and isinstance(sample_rate, (int, float)):
                pass
        except:
            raise TypeError(
                "`audio_in` should be a tuple consisting of a numpy array containing the audio and its sample rate!"
            )

        if audio.ndim != 1:
            raise ValueError(
                "only single-channel audio signals can be processed by the deltasigma modulator in PDM microphone!"
            )

        # return out, self.state(), recording
