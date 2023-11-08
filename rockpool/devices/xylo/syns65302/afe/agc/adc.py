"""
This module implements the ADC as a state machine.

NOTE: this modules consists of a possibly oversampled ADC followed by an anti-aliasing and decimation filter implemented in the digital domain.
The anti-aliasing filter is implemented as an IIR low-pass filter + decimation via some fine-tuning and optimization.

NOTE: This module contains the block-diagram implementation of the filter to make sure that it is compatible with Hardware.
      To design this filter, we apply optimization to reduce the aliasing noise due to sampling as much as possible.
      We use the class of IIR Elliptic filters to get the sharpest transition and smallest aliasing noise.
      Also we use an optimized truncation method so that:
          - the dynamic range of the ADC does not drop due to worst-case vs. average case amplitude gain of the filter.
          - only a slight nonlinearity is introduced (truncation) only when the signal amplitude is quite large.

For further details on the implementation of this filter and its performance evaluation, please refer to the original design repo
https://spinystellate.office.synsense.ai/saeid.haghighatshoar/anti-aliasing-filter-for-xylo-a2
"""

__all__ = ["ADC"]

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_SAMPLING_RATE,
    NUM_BITS_AGC_ADC,
    XYLO_MAX_AMP,
)
from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter, State


class FilterOverflowError(Exception):
    pass


@dataclass
class BlockDiagram:
    oversampling_factor: int
    """oversampling factor: the block works corresponds to how much ADC oversampling"""

    fs: float
    """clock rate with which the block-diagram should be simulated to be matched with other modules"""

    a_taps: np.ndarray
    """filter AR taps"""

    B_a: int
    """number of bits devoted to the AR taps"""

    B_af: int
    """number of bits devoted to the fractional part of AR taps"""

    b_taps: np.ndarray
    """filter MA taps"""

    B_out: int
    """number of bits devoted to the output of MA part"""

    surplus: int
    """surplus factor for adjusting the gain for clipping"""

    B_b: int = 8
    """number of bits devoted to b-tap"""

    B_in: int = 10
    """number of bits in the input"""

    B_w: int = 17
    """bitwidth of the AR output w[n]"""

    B_sur: int = 8
    """number of bits devoted to surplus factor"""


# - Block diagram used for oversampling factor 2 and 4
#! Note: we have no implementation for other oversampling factors
bd_oversampling_0 = None

# note for oversampling 1, we use just a dummy filter (it passes the signal without any filtering)
# this would be simply equivalent to an ordinary ADC without additional aliasing reduction via oversampling
bd_oversampling_1 = BlockDiagram(
    oversampling_factor=1,
    fs=1 * AUDIO_SAMPLING_RATE,
    a_taps=np.asarray([65536, 0, 0, 0, 0], dtype=np.int64),
    B_a=17,
    B_af=16,
    b_taps=np.asarray([256, 0, 0, 0, 0], dtype=np.int64),
    B_out=16,
    surplus=256,
)

bd_oversampling_2 = BlockDiagram(
    oversampling_factor=2,
    fs=2 * AUDIO_SAMPLING_RATE,
    a_taps=np.asarray([65536, -76101, 93600, -46155, 15598], dtype=np.int64),
    B_a=18,
    B_af=16,
    b_taps=np.asarray([45, 63, 96, 63, 45], dtype=np.int64),
    B_out=22,
    surplus=163,
)

bd_oversampling_3 = None

bd_oversampling_4 = BlockDiagram(
    oversampling_factor=4,
    fs=4 * AUDIO_SAMPLING_RATE,
    a_taps=np.asarray([32768, -93468, 113014, -65651, 15547], dtype=np.int64),
    B_a=18,
    B_af=15,
    b_taps=np.asarray([46, -62, 96, -62, 46], dtype=np.int64),
    B_out=22,
    surplus=33,
)


bd_list = [
    bd_oversampling_0,
    bd_oversampling_1,
    bd_oversampling_2,
    bd_oversampling_3,
    bd_oversampling_4,
]


class AntiAliasingDecimationFilter(Module):
    """
    Simulate the block-diagram model of the decimation anti-aliasing filter
    """

    def __init__(self, oversampling_factor: int = 2) -> None:
        """
        Args:
            oversampling_factor (int, optional): oversampling factor of ADC. Defaults to 2.
        """
        if oversampling_factor not in [1, 2, 4]:
            raise NotImplementedError(
                f"decimation filter in block-diagram format is not yet implemented for oversampling factor {self.oversampling_factor}!"
            )

        self.oversampling_factor = oversampling_factor
        self.bd: BlockDiagram = bd_list[self.oversampling_factor]
        self.sample_rate = SimulationParameter(self.bd.fs, shape=())
        self.op_state = State(
            np.zeros(len(self.bd.a_taps), dtype=np.int64),
            shape=(len(self.bd.a_taps)),
            init_func=lambda s: np.zeros(s, dtype=np.int64),
        )
        """
        Number of bit registers in AR part
        NOTE: we use an additional dummy register so that the computation of MA part is simplified.
        This is not needed in the ordinary computation.
        """

    def evolve(self, sig_in: float, record: bool = False) -> Tuple[float, dict, dict]:
        """
        Simulate for one step

        Args:
            sig_in (float): one time step signal
            record (bool, optional): Record the internal states. Defaults to False.

        Returns:
            Tuple[float, dict, dict]:
                out: decimation filter output
                state: the current state of the module
                rec: extra recordings
        """
        # * check the range of input
        if np.max(sig_in) >= 2 ** (self.bd.B_in - 1) or np.min(sig_in) < -(
            2 ** (self.bd.B_in - 1)
        ):
            raise ValueError(
                f"input signal should have {self.bd.B_in} and in the range [{-2**(self.bd.B_in-1)}, {2**(self.bd.B_in-1)-1}]!"
            )

        # * compute AR part
        # compute the feedback part: each branch individually
        feedback_branches_out = (
            -self.op_state[:-1] * self.bd.a_taps[1:]
        ) >> self.bd.B_af

        # add all branches
        feedback_out = np.sum(feedback_branches_out)

        # add the input
        next_w_sample = sig_in + feedback_out

        # update w
        self.op_state[1:] = self.op_state[:-1]
        self.op_state[0] = next_w_sample

        # * compute the MA part
        sig_out_MA = np.sum(self.op_state * self.bd.b_taps)

        # check the overflow in MA part of the filter
        if np.max(sig_out_MA) >= 2 ** (self.bd.B_out - 1) or np.min(sig_out_MA) < -(
            2 ** (self.bd.B_out - 1)
        ):
            raise FilterOverflowError(
                f"Overflow in the MA branch of the filter! the output of MA should fit in {self.bd.B_out} signed bits!"
            )

        # * compute the output after surplus scaling
        sig_out_surplus = (sig_out_MA * self.bd.surplus) >> self.bd.B_af

        # * compute the output after final truncation
        max_pos_amplitude = 2 ** (self.bd.B_in - 1) - 1
        min_neg_amplitude = -(2 ** (self.bd.B_in - 1))

        sig_out = sig_out_surplus

        if sig_out > max_pos_amplitude:
            sig_out = max_pos_amplitude

        if sig_out < min_neg_amplitude:
            sig_out = min_neg_amplitude

        # final output sample and its distortion
        if record:
            # compute the relative truncation distortion
            EPS = 0.000000001
            rel_distortion = np.abs(sig_out - sig_out_surplus) / (
                np.max([np.abs(sig_out), np.abs(sig_out_surplus)]) + EPS
            )
            __rec = {"rel_distortion": rel_distortion}

        return sig_out, self.state(), __rec


class ADC:
    """
    Equivalent ADC: consisting of oversampled ADC + anti-aliasing decimation filter

    NOTE: In hardware, ADC is implemented in two stages:
        - a simple high-rate ADC that quantizes the output of an analog low-pass filter
        - an anti-aliasing and decimation filter that removes the aliasing and reduces the sampling rate back to target audio sampling rate.
    """

    def __init__(
        self,
        num_bits: int = NUM_BITS_AGC_ADC,
        max_audio_amplitude: float = XYLO_MAX_AMP,
        oversampling_factor: int = 1,
        fs: float = AUDIO_SAMPLING_RATE,
    ):
        """

        Args:
            num_bits (int, optional): number of bits in ADC. Defaults to 10 in current Xylo-A3 hardware.
            max_audio_amplitude (float, optional): maximum audio amplitude that can be handled within the chip. Defaults to XYLO_MAX_AMP.
            oversampling_factor (int, optional): oversampling factor of the high-rate ADC used in the implementation of ADC. Defaults to 1.
            fs (float, optional): target sampling rate of the equivalent ADC (sampling rate of the audio). Defaults to AUDIO_SAMPLING_RATE.
        """

        self.num_bits = num_bits
        self.max_audio_amplitude = max_audio_amplitude

        if not isinstance(oversampling_factor, int):
            raise ValueError(
                "oversampling factor of the ADC should be an integer (1, 2, or 4 in the current implementation)!"
            )
        self.oversampling_factor = oversampling_factor

        self.fs = fs

        self.oversampled_fs = self.fs * self.oversampling_factor

        # build the decimation anti-aliasing filter
        self.anti_aliasing_filter = AntiAliasingDecimationFilter(
            oversampling_factor=oversampling_factor
        )

        if self.anti_aliasing_filter.bd.fs != self.oversampled_fs:
            raise ValueError(
                f"the sampling rate of anti-alising decimation filter should be {self.oversampling_factor} x sampling rate of ADC (target audio sampling rate)!\n"
                + "otherwise the filter specification (passband and bandwidth) may be different than the target design value!"
            )

        # reset the ADC and also the inner anti-aliasing decimation filter
        self.reset()

    def reset(self):
        self.anti_aliasing_filter.reset()

        self.time_stamp = 0
        self.sample = 0
        self.stable_sig_in = 0

        self.decimation_filter_out = 0

        # number of samples receieved
        self.num_processed_samples = 0

        # reset the state as well
        self.op_state = {}

    def evolve(self, sig_in: float, time_in: float, record: bool = False):
        """this function processes the input timed-sample and updates the state of ADC.

        Args:
            sig_in (float): input signal sample.
            time_in (float): time stamp of the signal sample.
            record (bool, optional): record simulation state. Defaults to False.
        """
        # NOTE: if PGA is not settled after gain change, its output signal will be unstable.
        # In such a case, we model the output of PGA by None.
        # When this happens, we assume that there is a buffer that keeps the stable value of ADC and avoid accepting new samples from ADC until
        # PGA returns to the stable mode. During this unstable period, the buffer keeps sending the last stable sample it received from ADC.

        if sig_in is None:
            # just repeat the previous sample since the input received from the amplifier is invalid
            sig_in = self.stable_sig_in
        else:
            # if stable record the sample for the next time instants
            self.stable_sig_in = sig_in

        # check the start of the simulation and set the gain values in PGA
        if self.num_processed_samples == 0:
            if record:
                self.op_state = {
                    "num_processed_samples": [],
                    "time_in": [],
                    "adc_in": [],
                    "adc_oversampled_out": [],
                    "anti_aliasing_filter_out": [],
                    "anti_aliasing_filter_rel_distortion": [],
                }
            else:
                self.op_state = {}

        # increase the number of processed samples
        self.num_processed_samples += 1

        # record the state if needed
        if record:
            self.op_state["num_processed_samples"].append(self.num_processed_samples)
            self.op_state["time_in"].append(time_in)
            self.op_state["adc_in"].append(sig_in)

        if time_in >= self.time_stamp / self.oversampled_fs:
            # * it is time to get the quantized version of the sample
            # NOTE: this sampling is done with the oversampled ADC -> then processed with anti-aliasing + deicmation filter
            EPS = 0.00001
            sig_in_norm = sig_in / (XYLO_MAX_AMP * (1 + EPS))

            # add a one unit of clock delay to the returned sample
            sample_return, self.sample = self.sample, int(
                np.fix(2 ** (self.num_bits - 1) * sig_in_norm)
            )

            self.time_stamp += 1

            # process this sample with the anti-aliasing + decimation filter
            (
                anti_aliasing_filter_out,
                anti_aliasing_filter_rel_distortion,
            ) = self.anti_aliasing_filter.evolve_onestep(sig_in=sample_return)

            # record the state: returned sample
            if record:
                self.op_state["adc_oversampled_out"].append(sample_return)
                self.op_state["anti_aliasing_filter_out"].append(
                    anti_aliasing_filter_out
                )
                self.op_state["anti_aliasing_filter_rel_distortion"].append(
                    anti_aliasing_filter_rel_distortion
                )

        else:
            # * otherwise: ignore those incoming signals
            # NOTE: we do this since the input to ADC comes from the amplifier which may need to be simulated with a much higher sampling rate.
            # So those extra samples are needed for the precision of amplifier simulation and are ignored by the ADC.
            pass

        # compute the output of decimation filter
        if self.time_stamp % self.oversampling_factor == 1:
            self.decimation_filter_out = anti_aliasing_filter_out

        return self.decimation_filter_out

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        this is the same as `evolve` function.
        """
        return self.evolve(*args, **kwargs)

    def __repr__(self) -> str:
        # string representation of the ADC module
        string = (
            "+" * 100
            + "\n"
            + "ADC module:\n"
            + f"clock rate: {self.fs}\n"
            + f"clock rate of the inner oversampled ADC: {self.oversampled_fs}\n"
            + f"specification of the anti-aliasing decimation filter:\n"
            + f"{self.anti_aliasing_filter}"
            + f"maximum amplitude: {self.max_audio_amplitude}\n"
            + f"number of bits used for signed quantization: {self.num_bits}\n"
        )

        return string
