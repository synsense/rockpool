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
        select_filters: Tuple[int] = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ),
    ) -> None:
        """
        AFESim constructor

        Args:
            select_filters (Tuple[int], optional): The indices of the filters to be used in the filter bank. Defaults to ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ).
        """
        if self.validate_filter_selection(select_filters):
            self.__select_filters = select_filters
        else:
            raise ValueError("Invalid filter selection.")

        shape = (1, len(self.__select_filters))
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

    @staticmethod
    def validate_filter_selection(select_filters: Tuple[int]) -> bool:
        """
        Validate the filter selection, check if the range and the values are valid.

        Args:
            select_filters (Tuple[int]): The indices of the filters to be used in the filter bank.

        Raises:
            TypeError: select_filters must be of type tuple.
            TypeError: Filter index is not an integer.
            ValueError: Filter index is out of range. Valid indices are between 0 and 15.
            ValueError: All filter indices in select_filters must be unique.

        Returns:
            bool: True if the filter selection is valid.
        """
        if not isinstance(select_filters, tuple):
            raise TypeError("select_filters must be of type tuple.")

        # Check if all elements are integers and are within the allowed range
        for filter_index in select_filters:
            if not isinstance(filter_index, int):
                raise TypeError(f"Filter index {filter_index} is not an integer.")
            if filter_index < 0 or filter_index > 15:
                raise ValueError(
                    f"Filter index {filter_index} is out of range. Valid indices are between 0 and 15."
                )

        if len(select_filters) != len(set(select_filters)):
            raise ValueError("All filter indices in select_filters must be unique.")

        return True

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
