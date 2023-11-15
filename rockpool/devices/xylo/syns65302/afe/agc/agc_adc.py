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
    AUDIO_SAMPLING_RATE,
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
)

__all__ = ["AGCADC"]


class AGCADC(Module):
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
            fs=target_fs * oversampling_factor,
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

        adc_out_vec = []
        agc_pga_gain_vec = []
        agc_pga_command_vec = []
        amplifier_out_vec = []
        envelope_vec = []
        gain_smoother_vec = []

        adc_out = 0
        agc_pga_gain = 1.0
        agc_pga_command = 0
        amplifier_out = 0
        envelope = 0
        gain_smoother_out = 0

        for sig_in in audio:
            # input time instant

            # produce amplifier output
            # NOTE: the old value of agc_pga_command computed in the past clock is used to produce amplifier output and ADC output
            amplifier_out, _, _ = self.amplifier.evolve(
                audio=sig_in,
                pga_command=agc_pga_command,
                record=record,
            )

            # produce the ADC output and register the PGA gain used while ADC was quantizing the signal
            adc_out, _, _ = self.adc.evolve(sig_in=amplifier_out, record=record)

            # * record the gain and the gain index that was used at this time slot
            # Note: that as soon as the new clock comes, gain index is updated by envelope controller but that gain index will be used for the current clock
            # So it does not affect the gain used in ADC
            agc_pga_gain_index_used_in_adc = agc_pga_command
            agc_pga_gain_used_in_adc = self.amplifier.pga_gain_vec[
                agc_pga_gain_index_used_in_adc
            ]

            # * run envelope controller
            # NOTE: PGA command is updated and sets the PGA gain value which will appear in the next clock
            agc_pga_command, __state, __rec = self.envelope_controller.evolve(
                sig_in=adc_out, record=record
            )
            envelope = __state["envelope"]

            # compute/update the pga gain value as soon as the rising edge of the clock comes
            # NOTE: this new gain will be used in this clock and its effect will appear on ADC signal in the next clock
            # because during this clock period, ADC is still working to prepare the signal sample
            agc_pga_gain = self.amplifier.pga_gain_vec[agc_pga_command]

            # use the ADC out and the command generated by PGA in gain smoother
            if self.gain_smoother is not None:
                gain_smoother_out, _, _ = self.gain_smoother.evolve(
                    audio=adc_out,
                    pga_gain_index=agc_pga_gain_index_used_in_adc,
                    record=record,
                )

            # save the results
            # NOTE: due to skipping some samples when modules have various sampling rates,
            # the output of all modules is registered only at lowest sampling rate of all modules
            adc_out_vec.append(adc_out)
            agc_pga_gain_vec.append(agc_pga_gain)
            agc_pga_command_vec.append(agc_pga_command)
            amplifier_out_vec.append(amplifier_out)
            envelope_vec.append(envelope)

            if self.gain_smoother is not None:
                gain_smoother_vec.append(gain_smoother_out)

        adc_out_vec = np.asarray(adc_out_vec)
        agc_pga_command_vec = np.asarray(agc_pga_command_vec)
        agc_pga_gain_vec = np.asarray(agc_pga_gain_vec)
        amplifier_out_vec = np.asarray(amplifier_out_vec)
        envelope_vec = np.asarray(envelope_vec)
        gain_smoother_vec = np.asarray(gain_smoother_vec)

        out = adc_out_vec if self.gain_smoother is None else gain_smoother_vec

        rec = {
            "agc_pga_command": agc_pga_command_vec,
            "agc_pga_gain": agc_pga_gain_vec,
            "amplifier_output": amplifier_out_vec,
            "envelope": envelope_vec,
            "adc_output": adc_out_vec,
            "gain_smoother_output": gain_smoother_vec,
        }

        return out, self.state, rec
