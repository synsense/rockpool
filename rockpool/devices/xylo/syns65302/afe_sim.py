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
        dn_rate_scale_bitshift: Tuple[int] = (6, 0),
        dn_low_pass_bitshift: int = 12,
        dn_EPS: Union[int, List[int]] = 1,
        fixed_threshold_vec: Union[int, List[int]] = 2 ** (14 - 1 + 8 + 6),
        down_sampling_factor: int = 6,
    ) -> None:
        """
        AFESim constructor

        Args:
            select_filters (Optional[Tuple[int]], optional): The indices of the filters to be used in the filter bank. Defaults to None: use all filters.
                i.e. select_filters = (0,2,4,8,15) will use Filter 0, Filter 2, Filter 4, Filter 8, and Filter 15.

            dn_rate_scale_bitshift (Tuple[int], optional):  A tuple containing two bitshift values that determine how much the spike rate should be scaled compared with the sampling rate of the input audio. The first value is `b1` and the second is `b2`. Defaults to (6, 0).
                A bitshift of size specified by the tuple as `(b1, b2)` yields a spike rate scaling of fs/(2^b1 - 2^b2) where fs is the sampling rate of the input audio.
                A default value of (6, 0) yields an average of 1 (slightly larger than 1) spike per 2^6 - 1 (=63) clock periods. With a clock rate of around 50K -> around 800 ~ 1K spikes/sec per channel.
                Use `.from_specification()` method to perform a parameter search for (b1,b2) values given the target scaling ratio.

            dn_low_pass_bitshift (int): number of bitshifts used in low-pass filter implementation. A bitshift of size `b` implies an averaging window of `2^b` clock periods. Defaults to 12.
                The default value of 12, implies an averaging window of size 4096 clock periods. For an audio of clock rate 50K, this yields an averaging window of size 80 ms.
                Use `.from_specification()` method to perform a parameter search for b values given the target averaging window size.

            dn_EPS (Union[int, np.ndarray]): lower bound on spike generation threshold. Defaults to 1.
                Using this parameter we can control the noise level in the sense that if average power in a channel is less than EPS, the spike rate of that channel is somehow diminished during spike generation.

        """

        __filter_bank = ChipButterworth(select_filters=select_filters)

        super().__init__(
            shape=__filter_bank.shape, spiking_input=False, spiking_output=True
        )

        __sub_shape = (__filter_bank.size_out, __filter_bank.size_out)

        self.filter_bank = __filter_bank
        self.divisive_norm = DivisiveNormalization(
            shape=__sub_shape,
            spike_rate_scale_bitshift1=dn_rate_scale_bitshift[0],
            spike_rate_scale_bitshift2=dn_rate_scale_bitshift[1],
            low_pass_bitshift=dn_low_pass_bitshift,
            EPS_vec=dn_EPS,
            fixed_threshold_vec=fixed_threshold_vec,
            fs=AUDIO_SAMPLING_RATE,
        )
        self.raster = Raster(
            shape=__sub_shape,
            rate_downsample_factor=2**down_sampling_factor,
            max_num_spikes=MAX_SPIKES_INPUT,
            fs=AUDIO_SAMPLING_RATE,
        )

        self.model = Sequential(self.filter_bank, self.divisive_norm, self.raster)
        self.dt = SimulationParameter((2**down_sampling_factor) / AUDIO_SAMPLING_RATE)

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        __out, __state, __rec = self.model.evolve(input_data, record=record)

        return __out, __state, __rec

    @classmethod
    def from_config(cls, config: Any) -> AFESim:
        raise NotImplementedError("To be implemented following `samna` support")

    @classmethod
    def from_specification(cls, *args, **kwargs) -> AFESim:
        raise NotImplementedError(
            "Here we do not have any high-level specification that's different than __init__ parameters."
        )

    def export_config(self) -> Any:
        raise NotImplementedError("To be implemented following `samna` support")
