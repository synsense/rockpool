"""
This module checks the output of the AGC and if there is any abrupt jump in amplitude
smooths it out to avoid distortion due to signal jumps propagating to the next layers (filters, etc.).
This module can be seen as the digital correction and smoothing of the quantized gain values in AGC
obtained via analog implementation.
In practice, we can activate and deactivate this module to see if it has a major effect on the transient
of the filters and in case not (since filters themselves may suppress this gain jump) we can deactivate it.
"""

from typing import Optional, Tuple

import numpy as np

from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_SAMPLING_RATE_AGC,
    EXP_PGA_GAIN_VEC,
    INIFINITY_OF_TRANSIENT_PHASE,
    MAX_WAITING_BITWIDTH,
    NUM_BITS_AGC_ADC,
    NUM_BITS_COMMAND,
    NUM_BITS_GAIN_QUANTIZATION,
)
from rockpool.nn.modules import Module
from rockpool.parameters import SimulationParameter, State


class GainSmoother(Module):
    """
    Applies gain smoothing so that there is no jump in signal amplitude when gain is switched.

    NOTE:
    Here we are using a bitshift implementation where the averaging parameter `gamma` in gain smoothing equation

                    ``beta[t+1] = (1-gamma) beta[t] + gamma x 1``

    which should be around `(INIFINITY_OF_TRANSIENT_PHASE=> 6)/min-waiting-time` is of the form `(INIFINITY_OF_TRANSIENT_PHASE => 4) / 2^num_waiting_bitshift`.

    """

    def __init__(
        self,
        num_bits: int = NUM_BITS_AGC_ADC,
        min_waiting_time: float = (2**MAX_WAITING_BITWIDTH - 1)
        / AUDIO_SAMPLING_RATE_AGC,
        num_bits_command: int = NUM_BITS_COMMAND,
        pga_gain_vec: Optional[np.ndarray] = None,
        num_bits_gain_quantization: int = NUM_BITS_GAIN_QUANTIZATION,
        fs: float = AUDIO_SAMPLING_RATE_AGC,
    ):
        """
        Args:
            num_bits (int, optional): number of bits in the input siganl. Defaults to 14.
            min_waiting_time (float, optional): minimum waiting time used for AGC implementation.
            num_bits_command (int, optional): numnber of bits used to send gain-change commands from AGC to PGA module.
            pga_gain_vec (Optional[np.ndarray], optional): a vector containing gain sequence in PGA.
            num_bits_gain_quantization (int, optional): number of bits used for quantizing the gain ratios.
            fs (float, optional): clock rate of the module.
        """
        super().__init__(shape=(1, 1), spiking_input=False, spiking_output=False)

        self.num_bits = SimulationParameter(num_bits, shape=())
        """number of bits in the input signal"""

        # - compute the bitshift level needed for implementing the gain smoothing low-pass filter
        self.fs = fs
        min_waiting_num_samples = int(min_waiting_time * self.fs)

        # closest power of two smaller than this value
        min_waiting_num_samples = 2 ** (int(np.log2(min_waiting_num_samples)))
        avg_bitshift = int(
            np.fix(np.log2(min_waiting_num_samples / INIFINITY_OF_TRANSIENT_PHASE))
        )

        self.avg_bitshift = SimulationParameter(avg_bitshift, shape=())
        """
        * set the settling time such that `(INIFINITY_OF_TRANSIENT_PHASE => 4) x settling time` is equal to minimum waiting time
        ! NOTE (1): we set the bitshift such that even for the minimum waiting time, the gain of the gain smoother obtained from low-pass filter reaches its steady state.
        ! NOTE (2): In theory, we could use various bitshifts with various settling time for different signal amplitude levels but for simplicity we don't do that!
        INIFINITY_OF_TRANSIENT_PHASE -> set to 6 in default setting -> 6 time-constant of the filter == Infinity time
        """

        # number of bits used for sending the command from AGC to PGA
        self.num_bits_command = num_bits_command

        # check the gain ratios
        if pga_gain_vec is None:
            pga_gain_vec = EXP_PGA_GAIN_VEC

        pga_gain_vec = np.asarray(pga_gain_vec, np.float64)
        if np.any(pga_gain_vec[1:] / pga_gain_vec[:-1] <= 1.0):
            raise ValueError(
                "the ratio between consecutive gains should be always larger than 1.0 (as the gains are an increasing sequence)!"
            )

        self.pga_gain_vec = SimulationParameter(
            pga_gain_vec, shape=(2**self.num_bits_command)
        )
        """a vector containing gain sequence in PGA"""

        # - quantize the gain vectors
        # maximum ratio between consecutive gain values
        max_gain_ratio = np.max(self.pga_gain_vec[1:] / self.pga_gain_vec[:-1])
        num_bits_gain_integer = int(np.ceil(np.log2(max_gain_ratio)))

        """how many bits are needed for storing the integer part of this ratio"""

        if num_bits_gain_quantization > 15:
            raise ValueError(
                "Number of bits for quantizing gain ratio should be <= 15!"
            )

        self.num_bits_gain_quantization = SimulationParameter(
            num_bits_gain_quantization, shape=()
        )
        """total number of bits used for quantization"""

        num_bits_gain_fraction = self.num_bits_gain_quantization - num_bits_gain_integer

        self.num_bits_gain_fraction = SimulationParameter(
            num_bits_gain_fraction, shape=()
        )
        """how many bits are needed for the fractional part"""

        # - Up gain ratio
        up_gain_ratio_float = self.pga_gain_vec[1:] / self.pga_gain_vec[:-1]
        up_gain_ratio = np.fix(
            up_gain_ratio_float * 2**self.num_bits_gain_fraction
        ).astype(np.int64)

        self.up_gain_ratio = SimulationParameter(
            up_gain_ratio, shape=(2**self.num_bits_command - 1)
        )
        """vector containing the quantized gain ratios case 1: when gain increases"""

        # - Down gain ratio
        down_gain_ratio_float = self.pga_gain_vec[:-1] / self.pga_gain_vec[1:]
        down_gain_ratio = np.fix(
            down_gain_ratio_float * 2**self.num_bits_gain_fraction
        ).astype(np.int64)

        self.down_gain_ratio = SimulationParameter(
            down_gain_ratio, shape=(2**self.num_bits_command - 1)
        )
        """vector containing the quantized gain ratios case 2: when gain decreases"""

        # - States

        self.num_processed_samples = State(0, shape=(), init_func=lambda _: 0)
        """number of samples received
        NOTE: by resetting the number of samples to 0, pga_gain value will be taken and initialized from the first data received in the input
        """

        self.pga_current_gain_index = State(0, shape=(), init_func=lambda _: 0)

        def __gain_high_res_init(s: tuple) -> int:
            return 1 << (self.avg_bitshift + self.num_bits_gain_fraction)

        def __gain_init(s: tuple) -> int:
            return __gain_high_res_init(s) >> self.avg_bitshift

        def __float_gain_init(s: tuple) -> int:
            return __gain_init(s) / (2**self.num_bits_gain_fraction)

        self.gain_high_res = State(
            __gain_high_res_init(None), shape=(), init_func=__gain_high_res_init
        )
        self.gain = State(__gain_init(None), shape=(), init_func=__gain_init)
        self.float_gain = State(
            __float_gain_init(None), shape=(), init_func=__float_gain_init
        )

    def evolve(
        self,
        audio: int,
        pga_gain_index: int,
        record: bool = False,
    ) -> Tuple[float, dict, dict]:
        """Process the signal coming from the AGC and tries to smooth it out to avoid gain jumps.
        NOTE: if PGA is not settled after gain change, its output signal will be unstable.
        In such a case, we model the output of PGA by None.
        ADC and other modules should stop working until a valid sample arrives.
        We model this by generating None for the ADC output so that the next layers also skip this unstable period.
        In this mode, we just return None and do not update the gain and gain-smoothed signal after PGA goes back to stable mode.

        Args:
            audio (int): quantized audio signal received from AGC.
            pga_gain_index (int): index of the PGA gain vector for this sample of the signal.
            record (bool, optional): dummy variable, to fit the rockpool module conventions

        Returns:
            audio_out (float): processed signal
            state (dict): state dictionary
            rec (dict): empty dictionary
        """

        # check that the input type is indeed an integer
        if not isinstance(audio, (int, np.int64)):
            raise ValueError(
                "the input to gain smoother module coming from ADC should be in integer format!"
            )

        # increase number of received samples
        self.num_processed_samples += 1

        # see if there is a jump in gain detected
        if self.pga_current_gain_index != pga_gain_index:
            # * gain needs to be adjusted
            # set the gain values:
            # NOTE: signal needs to initially scaled by this value where the scaling ratio should ultimately
            # converge to 1.0 to settle down in the current pga_gain value.

            # set the gain depending on if the gain is decreasing or increasing
            if self.pga_current_gain_index < pga_gain_index:
                # * gain has increased: so we need to attenuate the sigal at first (based on the previous/current gain index)
                # NOTE: for example, if index changes from current_gain_index:0 -> pga_gain_index:1, we need to scale the signal by pga_gain_vec[0]/pga_gain_vec[1] -> down_gain_ratio[0 => current_gain_index]
                self.gain_high_res = (
                    self.down_gain_ratio[self.pga_current_gain_index]
                    << self.avg_bitshift
                )
                self.gain = self.gain_high_res >> self.avg_bitshift
                self.float_gain = self.gain / (2**self.num_bits_gain_fraction)

                # NOTE: adjust the gain index after computing the `gain_high_res`
                self.pga_current_gain_index = pga_gain_index

            else:
                # * gain has decreased: so we need to amplify the signal at first (based on the new gain index)
                # NOTE: adjust the gain index before computing `gain_high_res`.
                # NOTE: for example, if index changes from current_gain_index:4 -> pga_gain_index:3, we need to scale it with pga_gain_vec[4]/pga_gain_vec[3] -> up_gain_ratio[3 => pga_gain_index]
                self.pga_current_gain_index = pga_gain_index

                # adjust the gain
                self.gain_high_res = (
                    self.up_gain_ratio[self.pga_current_gain_index] << self.avg_bitshift
                )
                self.gain = self.gain_high_res >> self.avg_bitshift
                self.float_gain = self.gain / (2**self.num_bits_gain_fraction)

        else:
            # gain has not changed in the mean time
            # continue to evolve the high-res gain value according to low-pass filter dynamics
            # NOTE: our final target relative gain is always 1.0 so that the new gain settles down.
            self.gain_high_res = (
                self.gain_high_res
                - (self.gain_high_res >> self.avg_bitshift)
                + (1 << self.num_bits_gain_fraction)
            )
            self.gain = self.gain_high_res >> self.avg_bitshift
            self.float_gain = self.gain / (2**self.num_bits_gain_fraction)

        # compute output audio signal always with the updated gain
        audio_out = (audio * self.gain) >> self.num_bits_gain_fraction

        # there may happen some over- and underflow especially when the PGA processes the signal in saturation mode
        # we apply truncation to fix this issue
        # NOTE: due to this saturation, we may need additional bits in the smoothing module to handle these cases.
        # NOTE: we assume 2's complement representation
        maximum_amplitude_positive = 2 ** (self.num_bits - 1) - 1
        maximum_amplitude_negative = 2 ** (self.num_bits - 1)
        if audio_out > maximum_amplitude_positive:
            audio_out = maximum_amplitude_positive

        if audio_out < -maximum_amplitude_negative:
            audio_out = -maximum_amplitude_negative

        return audio_out, self.state(), {}
