"""
Simulation of an analog audio filtering front-end

Defines :py:class:`.AFESim` module.

See Also:
    For example usage of the :py:class:`.AFESim` Module, see :ref:`/devices/xylo-a3/afesim.ipynb`
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from rockpool.devices.xylo.syns65302.afe.digital_filterbank import ChipButterworth
from rockpool.devices.xylo.syns65302.afe.divisive_normalization import (
    DivisiveNormalization,
)
from rockpool.devices.xylo.syns65302.afe.params import (
    AUDIO_SAMPLING_RATE,
    MAX_SPIKES_INPUT,
)
from rockpool.devices.xylo.syns65302.afe.raster import Raster
from rockpool.nn.modules.module import Module
from rockpool.utilities.backend_management import backend_available
from rockpool.parameters import SimulationParameter
from rockpool.nn.combinators import Sequential

if backend_available("samna"):
    print("from samna.xyloA3.configuration import InputInterfaceConfig")
else:
    InputInterfaceConfig = Any

__all__ = ["AFESim"]


class AFESim(Module):
    """
    A :py:class:`.Module` that simulates the audio signal preprocessing on Xylo A3 chip.


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

            down_sampling_factor (int): The down-sampling factor of the raster module. Determines how many time-steps will be accumulated into a single time-step. Defaults to 50.
                Resulting dt = 0.001024
        """

        __filter_bank = ChipButterworth(select_filters=select_filters)

        super().__init__(
            shape=__filter_bank.shape, spiking_input=False, spiking_output=True
        )

        if spike_gen_mode not in ["divisive_norm", "threshold"]:
            raise ValueError(
                f"Invalid spike_gen_mode: {spike_gen_mode}. Valid options are: 'divisive_norm' and 'threshold'"
            )

        else:
            enable_DN_channel = True if spike_gen_mode == "divisive_norm" else False

        __sub_shape = (__filter_bank.size_out, __filter_bank.size_out)

        # - Sub-modules
        self.filter_bank = __filter_bank

        self.divisive_norm = DivisiveNormalization(
            shape=__sub_shape,
            enable_DN_channel=enable_DN_channel,
            spike_rate_scale_bitshift1=dn_rate_scale_bitshift[0],
            spike_rate_scale_bitshift2=dn_rate_scale_bitshift[1],
            low_pass_bitshift=dn_low_pass_bitshift,
            EPS_vec=dn_EPS,
            fixed_threshold_vec=fixed_threshold_vec,
            fs=AUDIO_SAMPLING_RATE,
        )
        self.raster = Raster(
            shape=__sub_shape,
            rate_downsample_factor=down_sampling_factor,
            max_num_spikes=MAX_SPIKES_INPUT,
            fs=AUDIO_SAMPLING_RATE,
        )

        self.model = Sequential(self.filter_bank, self.divisive_norm, self.raster)
        self.dt = SimulationParameter((down_sampling_factor) / AUDIO_SAMPLING_RATE)
        self.spike_gen_mode = spike_gen_mode

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Evolves the AFE state over input data

        Args:
            input_data (np.ndarray): Single dimensional input data to the AFE.
            record (bool, optional): If ``True``, the module should record internal state during evolution and return the record. If ``False``, no recording is required. Default: ``False``.

        Returns:
            Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
                ``output``: The output spike train
                ``state``: The state dictionary after evolution.
                ``record``: The record dictionary during evolution. Empty if ``record=False``.
        """
        __out, __state, __rec = self.model.evolve(input_data, record=record)

        return __out, __state, __rec

    @classmethod
    def from_config(cls, config: Any) -> AFESim:
        raise NotImplementedError("To be implemented following `samna` support")

    @classmethod
    def from_specification(
        cls,
        select_filters: Optional[Tuple[int]] = None,
        spike_gen_mode: str = "divisive_norm",
        dn_rate_scale_bitshift: Optional[Tuple[int]] = None,
        rate_scale_factor: Optional[float] = 1 / 64,
        dn_low_pass_bitshift: Optional[int] = None,
        low_pass_averaging_window: Optional[float] = 80e-3,
        dn_EPS: Union[int, Tuple[int]] = 1,
        fixed_threshold_vec: Union[int, Tuple[int]] = 2**27,
        down_sampling_factor: Optional[int] = None,
        dt: Optional[float] = 1024e-3,
        **kwargs,
    ) -> AFESim:
        def complain_if_both(p1: Any, p2: Any, name1: str, name2: str) -> None:
            if (p1 is not None and p2 is not None) or (p1 is None and p2 is None):
                raise ValueError(
                    f"Either `{name1}` or `{name2}` should be provided, but not both!"
                )

        complain_if_both(
            dn_rate_scale_bitshift,
            rate_scale_factor,
            "dn_rate_scale_bitshift",
            "rate_scale_factor",
        )

        if dn_rate_scale_bitshift is None:
            # Find dn_rate_scale_bitshift from rate_scale_factor
            dn_rate_scale_bitshift = (
                cls.get_dn_rate_scale_bitshift_from_rate_scale_factor(rate_scale_factor)
            )

        complain_if_both(
            dn_low_pass_bitshift,
            low_pass_averaging_window,
            "dn_low_pass_bitshift",
            "low_pass_averaging_window",
        )

        if dn_low_pass_bitshift is None:
            dn_low_pass_bitshift = (
                cls.get_dn_low_pass_bitshift_from_low_pass_averaging_window(
                    low_pass_averaging_window
                )
            )

        complain_if_both(down_sampling_factor, dt, "down_sampling_factor", "dt")

        if down_sampling_factor is None:
            down_sampling_factor = cls.get_down_sampling_factor_from_dt(dt)

        return cls(
            select_filters=select_filters,
            spike_gen_mode=spike_gen_mode,
            dn_rate_scale_bitshift=dn_rate_scale_bitshift,
            dn_low_pass_bitshift=dn_low_pass_bitshift,
            dn_EPS=dn_EPS,
            fixed_threshold_vec=fixed_threshold_vec,
            down_sampling_factor=down_sampling_factor,
        )

    @staticmethod
    def get_dn_rate_scale_bitshift_from_rate_scale_factor(
        rate_scale_factor: float,
    ) -> Tuple[int]:
        if rate_scale_factor is None:
            raise ValueError(
                "`rate_scale_factor` should be provided to obtain `dn_rate_scale_bitshift`!"
            )
        return ()

    @staticmethod
    def get_dn_low_pass_bitshift_from_low_pass_averaging_window(
        low_pass_averaging_window: float,
    ) -> int:
        if low_pass_averaging_window is None:
            raise ValueError(
                "`low_pass_averaging_window` should be provided to obtain `dn_low_pass_bitshift`!"
            )
        return -1

    @staticmethod
    def get_down_sampling_factor_from_dt(dt: float) -> int:
        if dt is None:
            raise ValueError(
                "`dt` should be provided to obtain `down_sampling_factor`!"
            )
        return -1

    def export_config(self) -> Any:
        raise NotImplementedError("To be implemented following `samna` support")
