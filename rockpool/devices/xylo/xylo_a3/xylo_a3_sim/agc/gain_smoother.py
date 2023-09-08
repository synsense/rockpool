# ----------------------------------------------------------------------------------------------------------------------
# This module checks the output of the AGC and if there is any abrupt jump in amplitude
# smoothes it out to avoid distortion due to signal jumps propagating to the next layers (filters, etc.).
#
# This module can be seen as the digital correction and smoothing of the quantized gain values in AGC
# obtained via analog implementation.
#
# In practice, we can activate and deactivate this module to see if it has a major effect on the transient
# of the filters and in case not (since filters themselves may suppress this gain jump) we can deactivate it.
#
# ----------------------------------------------------------------------------------------------------------------------

from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.xylo_a3_agc_specs import (
    AUDIO_SAMPLING_RATE,
)
import numpy as np
from typing import Any

# ===========================================================================
# *                        Some constants needed in the design
# ===========================================================================
# default setting used for envelope controller
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.xylo_a3_agc_specs import (
    EXP_PGA_GAIN_VEC,
    NUM_BITS_COMMAND,
    NUM_BITS_ADC,
)

# maximum number of bits devoted for implementing waiting times in the AGC controller algorithm
# NOTE: with a clock rate of 50K, this is around 1 min waiting time which would definitely be enough for all ranges of applications
# MAX_WAITING_BITWIDTH is set 24 in default mode
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.xylo_a3_agc_specs import (
    MAX_WAITING_BITWIDTH,
)

# how many time-constants is considered `INFINITY` in low-pass filter transient period
# INIFINITY_OF_TRANSIENT_PHASE is set to 6 in default mode
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.xylo_a3_agc_specs import (
    INIFINITY_OF_TRANSIENT_PHASE,
)

# how many bits are needed to quantize the gain ratio associated with the start and end of the jump
from rockpool.devices.xylo.xylo_a3.xylo_a3_sim.agc.xylo_a3_agc_specs import (
    NUM_BITS_GAIN_QUANTIZATION,
)


# ===========================================================================
# *  A simple quantized version better suited for implementation on FPGA
#    NOTE: A floating-point version is available in the old design
# ===========================================================================
class GainSmootherFPGA:
    def __init__(
            self,
            num_bits: int = NUM_BITS_ADC,
            min_waiting_time: float = (2 ** MAX_WAITING_BITWIDTH - 1) / AUDIO_SAMPLING_RATE,
            num_bits_command: int = NUM_BITS_COMMAND,
            pga_gain_vec: np.ndarray = EXP_PGA_GAIN_VEC,
            num_bits_gain_quantization: int = NUM_BITS_GAIN_QUANTIZATION,
            fs: float = AUDIO_SAMPLING_RATE,
    ):
        """this calss applies gain smoothing so that there is no jump in signal amplitude when gain is switched.

        NOTE:
        Here we are using a bitshift implementation where the averaging parameter `gamma` in gain smoothing equation

                        ``beta[t+1] = (1-gamma) beta[t] + gamma x 1``

        which should be around `(INIFINITY_OF_TRANSIENT_PHASE=> 6)/min-waiting-time` is of the form `(INIFINITY_OF_TRANSIENT_PHASE => 4) / 2^num_waiting_bitshift`.

        Args:
            num_bits (int, optional): number of bits in the input siganl. Defaults to 14.
            min_waiting_time (float, optional): minimum waiting time used for AGC implementation.
            num_bits_command (int, optional): numnber of bits used to send gain-change commands from AGC to PGA module.
            pga_gain_vec (Union[float, np.ndarray], optional): a vector containing gain sequence in PGA.
            num_bits_gain_quantization (int, optional): number of bits used for quantizing the gain ratios.
            fs (float, optional): clock rate of the module.
        """
        # number of bits in the input signal
        self.num_bits = num_bits

        # minimum waiting time in AGC design
        self.min_waiting_time = min_waiting_time

        # clock rate of the module
        self.fs = fs

        # - compute the bitshift level needed for implementing the gain smoothing low-pass filter
        min_waiting_num_samples = int(self.min_waiting_time * self.fs)

        # closest power of two smaller than this value
        self.min_waiting_num_samples = 2 ** (int(np.log2(min_waiting_num_samples)))

        # * set the settling time such that `(INIFINITY_OF_TRANSIENT_PHASE => 4) x settling time` is equal to minimum waiting time
        # ! NOTE (1): we set the bitshift such that even for the minimum waiting time, the gain of the gain smoother obtained from low-pass filter reaches its steady state.
        # ! NOTE (2): In theory, we could use various bitshifts with various settling time for different signal amplitude levels but for simplicity we don't do that!
        # INIFINITY_OF_TRANSIENT_PHASE -> set to 6 in default setting -> 6 time-constant of the filter == Infinity time
        self.avg_bitshift = int(
            np.fix(np.log2(self.min_waiting_num_samples / INIFINITY_OF_TRANSIENT_PHASE))
        )

        # number of bits used for sending the command from AGC to PGA
        self.num_bits_command = num_bits_command

        # check the gain ratios
        try:
            if len(pga_gain_vec) != 2 ** num_bits_command:
                raise ValueError(
                    "number of PGA gains should be the same as number of commands that can be sent from AGC to PGA!"
                )
        except:
            raise ValueError("PGA gains should be a list or an array!")

        self.pga_gain_vec = np.asarray(pga_gain_vec, np.float64)

        if np.any(self.pga_gain_vec[1:] / self.pga_gain_vec[:-1] <= 1.0):
            raise ValueError(
                "the ratio between consecutive gains should be always larger than 1.0 (as the gains are an increasing sequence)!"
            )

        # - quantize the gain vectors
        # maximum ratio between consecutive gain values
        max_gain_ratio = np.max(self.pga_gain_vec[1:] / self.pga_gain_vec[:-1])

        # how many bits are needed for storing the integer part of this ratio
        self.num_bits_gain_integer = int(np.ceil(np.log2(max_gain_ratio)))

        # how many bits are needed for the fractional part
        self.num_bits_gain_fraction = (
                num_bits_gain_quantization - self.num_bits_gain_integer
        )

        # total number of bits used for quantization
        self.num_bits_gain_quantization = num_bits_gain_quantization

        # * vector containing the quantized gain ratios
        # case 1: when gain increases
        self.up_gain_ratio_float = self.pga_gain_vec[1:] / self.pga_gain_vec[:-1]
        self.up_gain_ratio = np.fix(
            self.up_gain_ratio_float * 2 ** self.num_bits_gain_fraction
        ).astype(np.int64)

        # case 2: when gain decreases
        self.down_gain_ratio_float = self.pga_gain_vec[:-1] / self.pga_gain_vec[1:]
        self.down_gain_ratio = np.fix(
            self.down_gain_ratio_float * 2 ** self.num_bits_gain_fraction
        ).astype(np.int64)

        self.reset()

    def reset(self):
        # number of samples received
        # NOTE: by resetting the number of samples to 0, pga_gain value will be taken and initialized from the first
        # data received in the input
        self.num_processed_samples = 0

        self.pga_current_gain_index = 0

        # set the high and low resolution gain values
        self.gain_high_res = 1 << (self.avg_bitshift + self.num_bits_gain_fraction)
        self.gain = self.gain_high_res >> self.avg_bitshift
        self.float_gain = self.gain / (2 ** self.num_bits_gain_fraction)

        # reset the state
        self.state = {}

    def evolve(
            self,
            audio: int,
            time_in: float,
            pga_gain_index: int,
            record: bool = False,
    ):
        """this module process the signal coming from the AGC and tries to smooth it out to avoid gain jumps.

        Args:
            audio (int): quantized audio signal received from AGC.
            time_in (float): time of the incoming signal.
            pga_gain (float): gain of the PGA used for this sample of the signal.
            pga_gain_index (int): index of the PGA gain vector for this sample of the signal.
            record (bool, optional): record the state during simulation.
        """
        # NOTE: if PGA is not settled after gain change, its output signal will be unstable.
        # In such a case, we model the output of PGA by None.
        # ADC and other modules should stop working until a valid sample arrives.
        # We model this by generating None for the ADC output so that the next layers also skip this unstable period.
        # In this mode, we just return None and do not update the gain and gain-smoothed signal after PGA goes back to stable mode.

        if audio is None:
            raise ValueError(
                "the signal received from the ADC should be always valid! If PGA is in transient mode, ADC should keep repeating the last valid sample!"
            )

        # check that the input type is indeed an integer
        if not isinstance(audio, (int, np.int64)):
            raise ValueError(
                "the input to gain smoother module coming from ADC should be in integer format!"
            )

        # check the start of the simulation and set the gain values in PGA
        if self.num_processed_samples == 0:
            # integer gain to be used
            self.gain_high_res = 1 << (self.avg_bitshift + self.num_bits_gain_fraction)
            self.gain = self.gain_high_res >> self.avg_bitshift
            self.float_gain = self.gain / (2 ** self.num_bits_gain_fraction)

            # set the gain index
            self.pga_current_gain_index = 0

            # ================================================
            #   create the state during the initialization
            # ================================================
            if record:
                self.state = {
                    "time_in": [],
                    "num_processed_samples": [],
                    "audio_in": [],
                    "audio_out": [],
                    "gain_high_res": [],
                    "gain": [],
                    "float_gain": [],
                    "pga_gain_index": [],
                }
            else:
                self.state = {}

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
                self.float_gain = self.gain / (2 ** self.num_bits_gain_fraction)

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
                self.float_gain = self.gain / (2 ** self.num_bits_gain_fraction)

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
            self.float_gain = self.gain / (2 ** self.num_bits_gain_fraction)

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

        # ================================================
        #      record the state if needed
        # ================================================
        if record:
            self.state["time_in"].append(time_in)
            self.state["num_processed_samples"].append(self.num_processed_samples)
            self.state["audio_in"].append(audio)
            self.state["audio_out"].append(audio_out)
            self.state["gain_high_res"].append(self.gain_high_res)
            self.state["gain"].append(self.gain)
            self.state["float_gain"].append(self.float_gain)
            self.state["pga_gain_index"].append(pga_gain_index)

        return audio_out

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        this function is the same as `evolve` function.
        """
        self.evolve(*args, **kwargs)

    def __repr__(self) -> str:
        # a simple string representation of the gain smoother
        string = (
                "+" * 100
                + "\n"
                + "Gain smoother module:\n"
                + f"clock rate:{self.fs}\n"
                + f"designed for PGA gain vector:\n{self.pga_gain_vec}\n"
                + f"number of bits in input signal coming from ADC: {self.num_bits}\n"
                + f"designed for minimum waiting time: {self.min_waiting_time} sec with {self.min_waiting_num_samples} samples\n"
                + f"number of bits used in quantizing the gain ratio at the transition time\n"
                + f"\ttotal number of bits used for gain quantization: {self.num_bits_gain_quantization}\n"
                + f"\tfractional bits: {self.num_bits_gain_fraction}\n"
                + f"\tinteger bits: {self.num_bits_gain_integer}\n"
                + f"number of bits used in gain smoothing filter: {self.avg_bitshift}\n"
                + f"up-gain ratio vector before quantization:\n{self.up_gain_ratio_float}\n"
                + f"up-gain ratio vector after quantization:\n{self.up_gain_ratio}\n"
                + f"down-gain ratio vector before quantization:\n{self.down_gain_ratio_float}\n"
                + f"down-gain ratio vector after quantization:\n{self.down_gain_ratio}\n"
        )

        return string
