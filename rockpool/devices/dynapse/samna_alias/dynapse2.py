"""
Dynap-SE2 samna alias. Mimic the samna data structures

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
08/04/2022
"""

from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Dict, List

from dataclasses import dataclass

import json
import numpy as np
from enum import Enum


class ParameterType(int, Enum):
    """
    ParameterType implements the parameter type enumerator to descriminate P type and N type transistor paramters
    """

    p: int = 0
    n: int = 1


@dataclass
class SamnaAlias:
    """
    SamnaAlias embodies the common samna object utilities that are used by fake samna objects like
    snake to camel case conversion and json constructor
    """

    @staticmethod
    def snake_to_camel(name: str) -> str:
        """
        snake_to_camel converts a snake_case variable name to camelCase variable name

        :param name: the snake_case formatted variable name
        :type name: str
        :return: a camelCase formatted variable name
        :rtype: str
        """
        # __attr_name -> __attrName
        for i, c in enumerate(name):
            if c.isalpha():
                first_index = i
                break

        # Split the rest of the words
        split_list = name[first_index:].split("_")
        name = name[:first_index] + split_list[0]
        name += "".join(word.title() for word in split_list[1:])
        return name

    @property
    def ctor(self) -> Dict[str, Any]:
        """
        ctor creates a valid constructor to update a samna object using `from_json()` method
        convert the snake_case named variables to camelCase to be compatible with C++ methods

        :return: a dictionary of the object datastructure
        :rtype: Dict[str, Any]
        """
        extend = lambda o: o.ctor if isinstance(o, SamnaAlias) else o

        _dict = self.__dict__
        _keys = map(self.snake_to_camel, _dict.keys())
        _values = map(extend, _dict.values())

        return dict(zip(_keys, _values))

    def to_json(self) -> str:
        """
        to_json converts the contructor dictionary of the object to a samna compatible json string.

        :return: json string to update a samna object
        :rtype: str
        """
        _wrapper = {"value0": self.ctor}
        return json.dumps(_wrapper, indent="    ")

    def samna_object(self, cls: Any) -> Any:
        """
        samna_object converts the samna alias object to a real samna object

        i.e.
        event = NormalGridEvent()
        event.samna_object(samna.dynapse2.NormalGridEvent)

        :param cls: the samna class
        :type cls: Any
        :return: the samna object
        :rtype: Any
        """
        obj = cls()
        obj.from_json(self.to_json())
        return obj

    @abstractclassmethod
    def from_samna(cls, obj: Any) -> SamnaAlias:
        """
        from_samna is a abstract class method, which should be implemented individually

        :param obj: the reciprocal samna object
        :type obj: Any
        :return: the samna alias of the actual samna object
        :rtype: SamnaAlias
        """
        pass


@dataclass
class Dynapse2Parameter(SamnaAlias):
    """
    Dynapse2Parameter mimics the parameter object for Dynap-SE2.
    Converting to samna.dynapse2.Dynapse2Parameter is not recommended!

    :param coarse_value: integer coarse base value :math:`C \\in [0,5]`
    :type coarse_value: np.uint8
    :param fine_value: integer fine value to scale the coarse current :math:`f \\in [0,255]`
    :type fine_value: np.uint8
    :param type: the type of the parameter : N or P
    :type type: ParameterType
    :param _address: stores PG number and branch shifted appropriately for the hardware, defaults to 0
    :type _address: np.uint64, optional
    :param _cookie: a cookie(number) assigned to parameter with regards to the address, defaults to 0
    :type _cookie: np.uint64, optional
    :param _initial_type: the initial type of the transistor, defaults to None
    :type _initial_type: ParameterType, optional
    :param _switchable_type: set true for type changeable "..._V" parameters, defaults to False
    :type _switchable_type: bool, optional
    """

    coarse_value: np.uint8
    fine_value: np.uint8
    type: ParameterType
    _address: np.uint64 = 0
    _cookie: np.uint64 = 0
    _initial_type: ParameterType = None
    _switchable_type: bool = False

    def __post_init__(self) -> None:
        if self._initial_type is None:
            self._initial_type = self.type


@dataclass
class Dynapse2Destination(SamnaAlias):
    """
    Dynapse2Destination mimics the address part of the samna AER package for DynapSE2

    :param core: the core mask used while sending the events
            [1,1,1,1] means all 4 cores are on the target
            [0,0,1,0] means the event will arrive at core 2 only
    :type core: List[bool]
    :param x_hop: number of chip hops on x axis
    :type x_hop: int
    :param y_hop: number of chip hops on y axis
    :type y_hop: int
    :param tag: globally multiplexed locally unique event tag which is used to identify the connection between two neurons.
    :type tag: np.uint
    """

    core: List[bool]
    x_hop: int
    y_hop: int
    tag: np.uint

    def __post_init__(self) -> None:
        if abs(self.x_hop) > 7:
            raise ValueError("Cannot reach beyond +-7 chips in x axis")
        if abs(self.y_hop) > 7:
            raise ValueError("Cannot reach beyond +-7 chips in y axis")
        if self.tag > 2048 or self.tag < 0:
            raise ValueError("Illegal tag!")


@dataclass
class NormalGridEvent(SamnaAlias):
    """
    NormalGridEvent mimics the samna AER package for DynapSE2

    :param event: the destination of the package including routing information
    :type event: Dynapse2Destination
    :param timestamp: the timestamp of the event in microseconds (1e-6)
    :type timestamp: np.uint32
    """

    event: Dynapse2Destination
    timestamp: np.uint32

# --- For Typehinting --- #


class Dynapse2Interface(ABC):
    @abstractmethod
    def configure_opal_kelly(self):
        pass

    @abstractmethod
    def enable_output(self):
        pass

    @abstractmethod
    def get_device_type_name(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_number_of_chips(self):
        pass

    @abstractmethod
    def get_output_enable_status(self):
        pass

    @abstractmethod
    def grid_bus_write_events(self):
        pass

    @abstractmethod
    def grid_bus_write_events_u32(self):
        pass

    @abstractmethod
    def input_interface_write_events(self):
        pass

    @abstractmethod
    def input_interface_write_events_u32(self):
        pass

    @abstractmethod
    def input_interface_write_raw(self):
        pass

    @abstractmethod
    def output_read(self):
        pass

    @abstractmethod
    def read_events(self):
        pass

    @abstractmethod
    def reset_fpga(self):
        pass

    @abstractmethod
    def set_fpga_dynapse2_module_config(self):
        pass

    @abstractmethod
    def set_number_of_chip(self):
        pass


class Dynapse2Model(ABC):
    @abstractmethod
    def apply_configuration(self):
        pass

    @abstractmethod
    def clear_error_queue(self):
        pass

    @abstractmethod
    def get_configuration(self):
        pass

    @abstractmethod
    def get_error_events(self):
        pass

    @abstractmethod
    def get_firing_rate(self):
        pass

    @abstractmethod
    def get_firing_rates(self):
        pass

    @abstractmethod
    def get_output_events(self):
        pass

    @abstractmethod
    def get_sadc_sample_period_ms(self):
        pass

    @abstractmethod
    def get_sadc_value(self):
        pass

    @abstractmethod
    def get_sadc_values(self):
        pass

    @abstractmethod
    def read_events(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_sadc_sample_period_ms(self):
        pass


if __name__ == "__main__":
    _dest = Dynapse2Destination([True, True, False, True], 0, 1, 12)
    print(_dest.to_json())
