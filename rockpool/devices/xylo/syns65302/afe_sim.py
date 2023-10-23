"""
Simulation of an analog audio filtering front-end

Defines :py:class:`.AFESim` module.

See Also:
    For example usage of the :py:class:`.AFESim` Module, see :ref:`/devices/xylo-a3/afesim.ipynb`
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

from rockpool.devices.xylo.syns65302.afe.digital_filterbank import ChipButterworth
from rockpool.devices.xylo.syns65302.afe.divisive_normalization import (
    DivisiveNormalization,
)
from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_SAMPLING_RATE,
    MAX_SPIKES_INPUT,
)
from rockpool.devices.xylo.syns65302.afe.raster import Raster
from rockpool.nn.combinators.sequential import ModSequential
from rockpool.utilities.backend_management import backend_available
from rockpool.parameters import SimulationParameter

if backend_available("samna"):
    print("from samna.xyloA3.configuration import InputInterfaceConfig")
else:
    InputInterfaceConfig = Any

__all__ = ["AFESim"]


class AFESim(ModSequential):
    """
    A :py:class:`.ModSequential` that simulates the audio signal preprocessing on Xylo A3 chip.


    See Also:
        For example usage of the :py:class:`.AFESim` Module, see :ref:`/devices/xylo-a3/afesim.ipynb`
    """

    def __init__(
        self,
        select_filters: Optional[Tuple[int]] = None,
        spike_gen_mode: str = "divisive_norm",
        dn_rate_scale_bitshift: Tuple[int] = (6, 0),
        dn_low_pass_bitshift: int = 12,
        dn_EPS: Union[int, Tuple[int]] = 1,
        fixed_threshold_vec: Union[int, Tuple[int]] = 2**27,
        down_sampling_factor: int = 50,
    ) -> None:
        """
        AFESim constructor

        Args:
            select_filters (Optional[Tuple[int]], optional): The indices of the filters to be used in the filter bank. Defaults to None: use all filters.
                i.e. select_filters = (0,2,4,8,15) will use Filter 0, Filter 2, Filter 4, Filter 8, and Filter 15.

            spike_gen_mode (str, optional): The spike generation mode of the AFE. There are two ways to generate spikes, "divisive_norm" and "threshold". Defaults to "divisive_norm".
                When "divisive_norm" is selected, adaptive thresholds apply, and `dn_rate_scale_bitshift`, `dn_low_pass_bitshift`, `dn_EPS` parameters are used.
                When "threshold" is selected, fixed thresholds apply, and `fixed_threshold_vec` parameter is used.
                For detailed information, please check `DivisiveNormalization` module

            dn_rate_scale_bitshift (Tuple[int], optional):  A tuple containing two bitshift values that determine how much the spike rate should be scaled compared with the sampling rate of the input audio. The first value is `b1` and the second is `b2`. Defaults to (6, 0).
                A bitshift of size specified by the tuple as `(b1, b2)` yields a spike rate scaling of fs/(2^b1 - 2^b2) where fs is the sampling rate of the input audio.
                A default value of (6, 0) yields an average of 1 (slightly larger than 1) spike per 2^6 - 1 (=63) clock periods. With a clock rate of around 50K -> around 800 ~ 1K spikes/sec per channel.
                Use `.from_specification()` method to perform a parameter search for (b1,b2) values given the target scaling ratio.

            dn_low_pass_bitshift (int): number of bitshifts used in low-pass filter implementation. A bitshift of size `b` implies an averaging window of `2^b` clock periods. Defaults to 12.
                The default value of 12, implies an averaging window of size 4096 clock periods. For an audio of clock rate 50K, this yields an averaging window of size 80 ms.
                Use `.from_specification()` method to perform a parameter search for b values given the target averaging window size.

            dn_EPS (Union[int, Tuple[int]]): lower bound on spike generation threshold. Defaults to 1.
                Using this parameter we can control the noise level in the sense that if average power in a channel is less than EPS, the spike rate of that channel is somehow diminished during spike generation.

            fixed_threshold_vec (Union[int, Tuple[int]]): A tuple containing threshold values per channel which determine the spike generation threshold. Defaults to 2 ** (27) = 2 ** (14 - 1 + 8 + 6).
                Thresholds of size `size_out`, in case of a singular value, broadcasted. These thresholds are used only when the `spike_gen_mode = "threshold"`.
                The default value 2**27 is to ensure a spike rate of around 1K for an input sinusoid signal quantized to 14 bits.

                .. seealso::
                    How to set the value of threshold for a target spike rate?

                    In the current implementation, input audio to filters has 14 bits which is further left-bit-shifted by 8 bits to improve numerical precision, thus, 22 bits.
                    This implies that the output signal may have a maximum amplitude of at most `2^21 - 1 ~ 2^22`, for example, when fed by a sinusoid signal
                    within the passband of the filter.
                    For a target rate of around 1K. e.g., 1 spike every 50 clock period for an audio of sampling rate 50K, then we need to choose a threshold as large as
                    `50 x 2^22 ~ 2^27`.

            down_sampling_factor (int): Determines how many time-steps will be accumulated into a single time-step before feeding the data to the SNN core. Defaults to 50.
                Resulting dt = 0.001024
                Use `.from_specification()` method to perform a parameter search for down_sampling_factor value given the target dt.
        """

        __filter_bank = ChipButterworth(select_filters=select_filters)

        if spike_gen_mode not in ["divisive_norm", "threshold"]:
            raise ValueError(
                f"Invalid spike_gen_mode: {spike_gen_mode}. Valid options are: 'divisive_norm' and 'threshold'"
            )

        else:
            enable_DN_channel = True if spike_gen_mode == "divisive_norm" else False

        __sub_shape = (__filter_bank.size_out, __filter_bank.size_out)

        # - Sub-modules
        __divisive_norm = DivisiveNormalization(
            shape=__sub_shape,
            enable_DN_channel=enable_DN_channel,
            spike_rate_scale_bitshift1=dn_rate_scale_bitshift[0],
            spike_rate_scale_bitshift2=dn_rate_scale_bitshift[1],
            low_pass_bitshift=dn_low_pass_bitshift,
            EPS_vec=dn_EPS,
            fixed_threshold_vec=fixed_threshold_vec,
            fs=AUDIO_SAMPLING_RATE,
        )
        __raster = Raster(
            shape=__sub_shape,
            rate_downsample_factor=down_sampling_factor,
            max_num_spikes=MAX_SPIKES_INPUT,
            fs=AUDIO_SAMPLING_RATE,
        )

        super().__init__(__filter_bank, __divisive_norm, __raster)

        self.spike_gen_mode = spike_gen_mode
        self.dn_rate_scale_bitshift = SimulationParameter(dn_rate_scale_bitshift)
        self.dn_low_pass_bitshift = SimulationParameter(dn_low_pass_bitshift)
        self.dn_EPS = SimulationParameter(dn_EPS)
        self.fixed_threshold_vec = SimulationParameter(fixed_threshold_vec)
        self.down_sampling_factor = SimulationParameter(down_sampling_factor)

    @classmethod
    def from_config(cls, config: Any) -> AFESim:
        raise NotImplementedError("To be implemented following `samna` support")

    @classmethod
    def from_specification(
        cls,
        select_filters: Optional[Tuple[int]] = None,
        spike_gen_mode: str = "divisive_norm",
        rate_scale_factor: Optional[float] = 63,
        low_pass_averaging_window: Optional[float] = 84e-3,
        dn_EPS: Union[int, Tuple[int]] = 1,
        fixed_threshold_vec: Union[int, Tuple[int]] = 2**27,
        dt: Optional[float] = 1024e-6,
        **kwargs,
    ) -> AFESim:
        """
        Create an instance of AFESim by specifying higher level parameters for AFESim.

        Args:
            select_filters (Optional[Tuple[int]], optional): Check :py:class:`.AFESim`. Defaults to None.
            spike_gen_mode (str, optional): Check :py:class:`.AFESim`. Defaults to "divisive_norm".
            rate_scale_factor (Optional[float], optional): Target `rate_scale_factor` for the `DivisiveNormalization` module. Defaults to 63.
                Depended upon the dn_rate_scale_bitshift. ``rate_scale_factor = 2**dn_rate_scale_bitshift[0] - 2**dn_rate_scale_bitshift[1]``
                Not always possible to obtain the exact value of `rate_scale_factor` due to the hardware constraints.
                In such cases, the closest possible value is reported with an error message.
            low_pass_averaging_window (Optional[float], optional): Target `low_pass_averaging_window` for the `DivisiveNormalization` module. Defaults to 84e-3.
                Depended upon the dn_low_pass_bitshift. ``low_pass_averaging_window = 2**dn_low_pass_bitshift / AUDIO_SAMPLING_RATE``
                Not always possible to obtain the exact value of `low_pass_averaging_window` due to the hardware constraints.
                In such cases, the closest possible value is reported with an error message.
                Note that a value within 3 decimal precision is accepted as equal.
            dn_EPS (Union[int, Tuple[int]], optional): Check :py:class:`.AFESim`. Defaults to 1.
            fixed_threshold_vec (Union[int, Tuple[int]], optional): Check :py:class:`.AFESim`. Defaults to 2**27.
            dt (Optional[float], optional): Target `dt` value for the SNN core. Defaults to 1024e-6.
                Depended upon the down_sampling_factor. ``dt = down_sampling_factor / AUDIO_SAMPLING_RATE``
                Not always possible to obtain the exact value of `dt` due to the hardware constraints.
                In such cases, the closest possible value is reported with an error message.
                Note that a value within 6 decimal precision is accepted as equal.

        Returns:
            AFESim: A AFESim instance constructed by specifying higher level parameters.
        """
        logger = logging.getLogger()

        # - Make reporting possible
        dn_rate_scale_bitshift = cls.get_dn_rate_scale_bitshift(rate_scale_factor)
        dn_low_pass_bitshift = cls.get_dn_low_pass_bitshift(low_pass_averaging_window)
        down_sampling_factor = cls.get_down_sampling_factor(dt)

        __obj = cls(
            select_filters=select_filters,
            spike_gen_mode=spike_gen_mode,
            dn_rate_scale_bitshift=dn_rate_scale_bitshift,
            dn_low_pass_bitshift=dn_low_pass_bitshift,
            dn_EPS=dn_EPS,
            fixed_threshold_vec=fixed_threshold_vec,
            down_sampling_factor=down_sampling_factor,
        )

        def __report(
            arg_name: str, param: str, locals_dict: Dict[str, Any] = locals()
        ) -> None:
            """
            Report the value of the parameter that is obtained given the target value with the deviation.

            Args:
                arg_name (str): The name of the object constructor argument
                param (str): The name of the higher level parameter
                locals_dict (Dict[str, Any], optional): The variable segment of `from_specification`. Defaults to locals().
            """
            diff = locals_dict[param] - __obj.__getattribute__(param)
            logger.warning(
                f"`{arg_name}` = {locals_dict[arg_name]} is obtained given the target `{param}` = {locals_dict[param]}, with diff = {diff:.6e}"
            )

        __report("dn_rate_scale_bitshift", "rate_scale_factor")
        __report("dn_low_pass_bitshift", "low_pass_averaging_window")
        __report("down_sampling_factor", "dt")

        return __obj

    @staticmethod
    def get_dn_rate_scale_bitshift(rate_scale_factor: float) -> Tuple[int]:
        """
        Get the bitshift values `dn_rate_scale_bitshift` which determine how much the spike rate should be scaled compared with the sampling rate of the input audio.
        Used as a utility function in `from_specification()` method.
        Raises an error if the target `rate_scale_factor` cannot be obtained within the specified decimal precision.
        Can be independently used to obtain the bitshift values given the target `rate_scale_factor`.

        Args:
            rate_scale_factor (float): Target `rate_scale_factor` for the `DivisiveNormalization` module.
                Depended upon the dn_rate_scale_bitshift. ``rate_scale_factor = 2**dn_rate_scale_bitshift[0] - 2**dn_rate_scale_bitshift[1]``

        Returns:
            Tuple[int]: A tuple containing two bitshift values that determine how much the spike rate should be scaled compared with the sampling rate of the input audio. The first value is `b1` and the second is `b2`.
                fs' = fs/(2^b1 - 2^b2) where fs is the sampling rate of the input audio.
        """
        if not isinstance(rate_scale_factor, (int, float)):
            raise ValueError(
                f"`rate_scale_factor` should be an int or a float!, type = {type(rate_scale_factor)}"
            )
        if rate_scale_factor <= 0:
            raise ValueError(
                f"`rate_scale_factor` should be a positive number!, rate_scale_factor = {rate_scale_factor}"
            )

        best_neg_diff = -np.inf
        best_neg_candidate = ()
        best_pos_diff = np.inf
        best_pos_candidate = ()

        b1 = rate_scale_factor.bit_length()
        b2 = 0

        # Check if the result satisfies the condition
        if 2**b1 - 2**b2 == rate_scale_factor:
            return (b1, b2)

        for b2 in range(1, b1):
            if 2**b1 - 2**b2 == rate_scale_factor:
                return (b1, b2)
            else:
                candidate = (b1, b2)
                candidate_diff = 2**b1 - 2**b2 - rate_scale_factor
                if candidate_diff < 0 and candidate_diff > best_neg_diff:
                    best_neg_diff = candidate_diff
                    best_neg_candidate = candidate
                elif candidate_diff > 0 and candidate_diff < best_pos_diff:
                    best_pos_diff = candidate_diff
                    best_pos_candidate = candidate

        b1_pos, b2_pos = best_pos_candidate
        b1_neg, b2_neg = best_neg_candidate

        __err_message = (
            f"`rate_scale_factor` = {rate_scale_factor} is not possible to implement!"
            + f"\n\t `rate_scale_factor` = {2**b1_pos - 2**b2_pos} is possible with (b1, b2) = ({b1_pos}, {b2_pos})"
            + f"\n\t `rate_scale_factor` = {2**b1_neg - 2**b2_neg} is possible with (b1, b2) = ({b1_neg}, {b2_neg})"
            + f"\nPick one of them!"
        )

        raise ValueError(__err_message)

    @staticmethod
    def get_dn_low_pass_bitshift(
        low_pass_averaging_window: float, decimal: int = 3
    ) -> int:
        """
        Get the bitshift value `dn_low_pass_bitshift` which determines the averaging window length of the low-pass filter.
        Used as a utility function in `from_specification()` method.
        Raises an error if the target `low_pass_averaging_window` cannot be obtained within the specified decimal precision.
        Can be independently used to obtain the bitshift value given the target `low_pass_averaging_window`.

        Args:
            low_pass_averaging_window (float): Target `low_pass_averaging_window` for the `DivisiveNormalization` module.
                Depended upon the dn_low_pass_bitshift. ``low_pass_averaging_window = 2**dn_low_pass_bitshift / AUDIO_SAMPLING_RATE``
            decimal (int, optional): The number of decimal points to be considered when comparing the target `low_pass_averaging_window` with the obtained value. Defaults to 3.

        Returns:
            int: The bitshift value that determines the averaging window length of the low-pass filter.
        """
        # Write unit tests

        if low_pass_averaging_window is None:
            raise ValueError(
                "`low_pass_averaging_window` should be provided to obtain `dn_low_pass_bitshift`!"
            )

        # low_pass_averaging_window
        candidate_1 = int(np.log2(AUDIO_SAMPLING_RATE * low_pass_averaging_window))
        candidate_2 = candidate_1 + 1

        diff_1 = abs(
            low_pass_averaging_window - ((2**candidate_1) / AUDIO_SAMPLING_RATE)
        )
        diff_2 = abs(
            low_pass_averaging_window - ((2**candidate_2) / AUDIO_SAMPLING_RATE)
        )

        if diff_1 < diff_2:
            if diff_1 < 10 ** (-decimal):
                return candidate_1
            else:
                candidate = candidate_1
                diff = diff_1
        else:
            if diff_2 < 10 ** (-decimal):
                return candidate_2
            else:
                candidate = candidate_2
                diff = diff_2

        raise ValueError(
            f"Closest we can get to `low_pass_averaging_window `= "
            f"{low_pass_averaging_window:.3f} is {(2**candidate) / AUDIO_SAMPLING_RATE:.3f}"
            f" with `dn_low_pass_bitshift` = {candidate}, diff = {diff:.3f}"
        )

    @staticmethod
    def get_down_sampling_factor(dt: float, decimal: int = 6) -> int:
        """
        Get the down_sampling_factor which determines how many time-steps will be accumulated into a single time-step before feeding the data to the SNN core.
        Used as a utility function in `from_specification()` method.
        Raises an error if the target `dt` cannot be obtained within the specified decimal precision.
        Can be independently used to obtain the down_sampling_factor given the target `dt`.

        Args:
            dt (float): Target `dt` value for the SNN core.
            decimal (int, optional): The number of decimal points to be considered when comparing the target `dt` with the obtained value. Defaults to 6.

        Returns:
            int: The down_sampling_factor which determines how many time-steps will be accumulated into a single time-step before feeding the data to the SNN core.
        """
        if dt is None:
            raise ValueError(
                "`dt` should be provided to obtain `down_sampling_factor`!"
            )

        candidate_1 = int(dt * AUDIO_SAMPLING_RATE)
        candidate_2 = candidate_1 + 1

        diff_1 = abs(dt - (candidate_1 / AUDIO_SAMPLING_RATE))
        diff_2 = abs(dt - (candidate_2 / AUDIO_SAMPLING_RATE))

        if diff_1 < diff_2:
            if diff_1 < 10 ** (-decimal):
                return candidate_1
            else:
                candidate = candidate_1
                diff = diff_1

        else:
            if diff_2 < 10 ** (-decimal):
                return candidate_2
            else:
                candidate = candidate_2
                diff = diff_2

        raise ValueError(
            f"Closest we can get to `dt` = "
            f"{dt:.6f} is {candidate / AUDIO_SAMPLING_RATE:.6f}"
            f" with `down_sampling_factor` = {candidate}, diff = {diff:.6f}"
        )

    @property
    def low_pass_averaging_window(self) -> float:
        """Averaging window length in seconds depended on the `dn_low_pass_bitshift` parameter. Defines the averaging window length of the low-pass filter"""
        return (2**self.dn_low_pass_bitshift) / AUDIO_SAMPLING_RATE

    @property
    def dt(self) -> float:
        """Time-step length in seconds depended on the `down_sampling_factor` parameter"""
        return self.down_sampling_factor / AUDIO_SAMPLING_RATE

    @property
    def rate_scale_factor(self) -> float:
        """Rate scaling factor depended on the `dn_rate_scale_bitshift` parameter. Defines how much the spike rate should be scaled compared with the sampling rate of the input audio"""
        return 2 ** self.dn_rate_scale_bitshift[0] - 2 ** self.dn_rate_scale_bitshift[1]

    def export_config(self) -> Any:
        raise NotImplementedError("To be implemented following `samna` support")


if __name__ == "__main__":
    for i in range(1, 65):
        try:
            print(i, AFESim.get_dn_rate_scale_bitshift(i))
        except Exception as e:
            print(e)
