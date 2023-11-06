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
)


class AGCADC(Module):
    def __init__(
        self,
        fixed_pga_gain_index: Optional[float] = None,
        oversampling_factor: int = 1,
        amplitude_thresholds=AMPLITUDE_THRESHOLDS,
    ) -> None:
        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)

        # amplifier: Amplifier,

        if fixed_pga_gain_index is None:
            fixed_gain_for_PGA_mode = False
            pga_command_in_fixed_gain_for_PGA_mode = (
                DEFAULT_PGA_COMMAND_IN_FIXED_GAIN_FOR_PGA_MODE
            )
        else:
            fixed_gain_for_PGA_mode = True
            pga_command_in_fixed_gain_for_PGA_mode = fixed_pga_gain_index

        __amplifier = Amplifier(
            fixed_gain_for_PGA_mode=fixed_gain_for_PGA_mode,
            # PGA_GAIN_BYPASS
            pga_command_in_fixed_gain_for_PGA_mode=pga_command_in_fixed_gain_for_PGA_mode,
            # PGA_GAIN_IDX_CFG # [0-15] -> [1-32]
        )

        # adc: ADC
        __adc = ADC(oversampling_factor=oversampling_factor)
        # AGC_CTRL1.AAF_OS_MODE 0 - bypass
        # [1-2]

        # envelope_controller: EnvelopeController
        __envelope_ctrl = EnvelopeController(
            amplitude_thresholds=amplitude_thresholds,
            # AGC_AT_REG0 - AGC_AT_REG7 [10 bit each]
            # rise_time_constant = ,
            # RISE_AVG_BITSHIFT 5 bits
            rise_bitshift=x,
            # fall_time_constant = ,
            # FALL_AVG_BITSHIFT 5 bits
            fall_bitshift=x,
            reliable_max_hysteresis=x,
            # AGC_CTRL2.RELI_MAX_HYSTR
            waiting_time_vec=x,
            # AGC_WT0 - AGC_WT15
            max_waiting_time_before_gain_chang=x,
            # AGC_CTRL3.MAX_NUM_SAMPLE
            pga_gain_index_variation=x,
            # AGC_PGIV_REG0 + AGC_PGIV_REG1 + AGC_CTRL3
            # 3 bits (signed or unsigned?)
        )

        # gain_smoother: GainSmootherFPGA
        __gain_smoother = GainSmootherFPGA(
            min_waiting_time=min(waiting_time_vec), num_bits_gain_quantization=x
        )
        # AGC_CTRL2.AVG_BITSHIFT
        # AGC_CTRL2.NUM_BITS_GAIN_FRACTION

        __submod_list = []

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
