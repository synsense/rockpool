"""
Simulation of an analog audio filtering front-end

Defines :py:class:`.AFESim` module.

See Also:
    For example usage of the :py:class:`.AFESim` Module, see :ref:`/devices/xylo-a3/afesim.ipynb`
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from rockpool.devices.xylo.syns65302.afe.digital_filterbank import ChipButterworth
from rockpool.devices.xylo.syns65302.afe.divisive_normalization import (
    DivisiveNormalization,
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
        shape: Optional[Union[Tuple, int]] = (1, 16),
    ) -> None:
        super().__init__(shape=shape, spiking_input=False, spiking_output=True)

        self.filter_bank = ChipButterworth()
        self.divisive_norm = DivisiveNormalization()
        self.raster = Raster(shape=(self.size_out, self.size_out))
        self.model = Sequential(self.filter_bank, self.divisive_norm, self.raster)

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
