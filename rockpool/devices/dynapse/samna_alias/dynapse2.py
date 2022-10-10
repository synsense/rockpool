"""
Dynap-SE2 samna alias. Mimic the samna data structures

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
08/04/2022
[] TODO: samna object addresses
[] TODO: save and load json strings
"""

from __future__ import annotations
import logging
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Dict, List, Optional

from dataclasses import dataclass

import json
import numpy as np
from enum import Enum

SAMNA_AVAILABLE = False

try:
    import samna
    import samna.dynapse2 as se2

    SAMNA_AVAILABLE = True
except:
    samna = Any
    logging.warning(
        "samna installation not found in the environtment! You can still work with simulator to full extend but the objects cannot be converted to device configuration."
    )

# --- Enums --- #


class ParameterType(int, Enum):
    """
    ParameterType implements the parameter type enumerator to descriminate P type and N type transistor paramters
    """

    p: int = 0
    n: int = 1


class DvsMode(int, Enum):
    """
    DvsMode implements the DVS generation enumerator to describe the model DVS128, Davis240c, or Davis346
    """

    Dvs128: int = 0
    Davis240c: int = 2
    Davis346: int = 4


class Dendrite(int, Enum):
    """
    Dendrite implements the dynapse dendrite types enumerator
    """

    none: int = 0
    ampa: int = 1024
    gaba: int = 512
    nmda: int = 256
    shunt: int = 128


# --- Base Class --- #


@dataclass
class SamnaAlias:
    """
    SamnaAlias embodies the common samna object utilities that are used by fake samna objects like
    snake to camel case conversion and json constructor
    """

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
        if SAMNA_AVAILABLE:
            obj = cls()
            obj.from_json(self.to_json())
            return obj
        else:
            raise ModuleNotFoundError(
                "samna installation is not found in the environment!"
            )

    def save(self, path: Optional[str] = None) -> None:
        """
        save the file as a json

        :param path: the path to save, default to __class__.__name__.json
        :type path: str
        """
        if path is None:
            path = self.__class__.__name__ + ".json"

        with open(path, "w") as f:
            f.write(self.to_json())

    @abstractclassmethod
    def from_samna(cls, obj: Any) -> SamnaAlias:
        """
        from_samna is a abstract class method, which should be implemented individually

        :param obj: the reciprocal samna object
        :type obj: Any
        :return: the samna alias of the actual samna object
        :rtype: SamnaAlias
        """
        raise NotImplementedError("This method needs per class implementation!")

    @abstractclassmethod
    def to_samna(cls) -> Any:
        """
        to_samna is a abstract class method, which should be implemented individually
        it returns a samna object created by the same data-structure with the alias object
        """
        raise NotImplementedError("This method needs per class implementation!")

    # --- JSON Converter Utils --- #

    def to_json(self) -> str:
        """to_json creates a proper & samna-compatible json string from the samna alias object"""
        return json.dumps({"value0": self.json_wrapper()}, indent="    ")

    def json_wrapper(self) -> str:
        """
        json_wrapper returns a dictionary which is ready to be converted to a json string.
        wrapper stands in an intermediate place in between json and object dictionary
        """
        return self.ctor

    @property
    def ctor(self) -> Dict[str, Any]:
        """
        ctor creates a valid constructor to be able to update a samna object using `from_json()` method
        convert the snake_case named variables to camelCase to be compatible with C++ methods

        :return: a dictionary of the object datastructure
        :rtype: Dict[str, Any]
        """
        return dict(
            map(
                lambda kv: (self.snake_to_camel(kv[0]), kv[1]),
                self.__dict__.items(),
            )
        )

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

    @staticmethod
    def jlist_alias(__list: List[SamnaAlias]) -> Dict[str, Dict[str, Any]]:
        """
        jlist_alias process the list of samna objects and convert them to a dictionary of json wrappers

        :param __list: a list of samna objects
        :type __list: _type_
        :return: a dictionary of the dictionaries of the object datastructures inside the samna object list
        :rtype: Dict[str, Dict[str, Any]]
        """
        return {f"value{i}": alias.json_wrapper() for (i, alias) in enumerate(__list)}

    @staticmethod
    def jlist_regular(__list: List[Any]) -> Dict[str, Any]:
        """
        jlist_regular process the list of any standard python objects like str, bool, int or so
        then convert them to a dictionary which could be converted to json strings

        :param __list: a list of python object
        :type __list: List[Any]
        :return: a dictionary with enumerated value keys
        :rtype: Dict[str, Any]
        """
        return {f"value{i}": regular for (i, regular) in enumerate(__list)}

    @staticmethod
    def jdict_alias(__dict: Dict[str, SamnaAlias]) -> List[Dict[str, Any]]:
        """
        jdict_alias process the dictionary of samna alias objects (usually parameter maps),
        then convert them into a list of dictionaries of samna compatible entries

        :param __dict: the dictionary of samna alias objects
        :type __dict: Dict[str, SamnaAlias]
        :return: a list of dictionaries
        :rtype: List[Dict[str, Any]]
        """
        return list(
            map(
                lambda __kv: {"key": __kv[0], "value": __kv[1].json_wrapper()},
                __dict.items(),
            )
        )

    @staticmethod
    def jlist_dict_alias(
        __list_dict: List[Dict[str, SamnaAlias]]
    ) -> Dict[List[Dict[str, Any]]]:
        """
        jlist_dict_alias processes the list of samna object dictionaries (usually parameter map lists)
        then convert them into a dictionaries of lists of dictionaries of samna compatible entries

        :param __list_dict: list of samna object dictionaries
        :type __list_dict: List[Dict[str, SamnaAlias]]
        :return: the dictionaries of samna compatible entries
        :rtype: Dict[List[Dict[str, Any]]]
        """
        return SamnaAlias.jlist_regular(
            [SamnaAlias.jdict_alias(d) for d in __list_dict]
        )


# --- Mainly used samna aliases --- #


@dataclass
class Dynapse2Parameter(SamnaAlias):
    """
    Dynapse2Parameter mimics the parameter object for Dynap-SE2.
    Converting to samna.dynapse2.Dynapse2Parameter is not recommended!

    :Parameters:

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

    type: str
    coarse_value: np.uint8
    fine_value: np.uint8
    _address: np.uint64 = 0
    _cookie: np.uint64 = 0
    _initial_type: ParameterType = None
    _switchable_type: bool = False

    def __post_init__(self) -> None:
        """
        __post_init__ runs after initialization
        """
        if self._initial_type is None:
            self._initial_type = self.type

    @property
    def ctor(self) -> Dict[str, Any]:
        """
        ctor overwrites `SamnaAlias.ctor()` method to express `type` field as unicode

        :return: a dictionary of the object datastructure
        :rtype: Dict[str, Any]
        """
        __ctor = super().ctor
        __ctor["type"] = ord(__ctor["type"])
        return __ctor

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Parameter:
        """
        from_samna converts a `Dynapse2Parameter` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2Parameter` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Parameter
        """

        return cls(
            coarse_value=obj.coarse_value,
            fine_value=obj.fine_value,
            type=obj.type,
            _address=obj._address,
            _cookie=obj._cookie,
            _initial_type=ParameterType(obj._initial_type.value),
            _switchable_type=obj._switchable_type,
        )

    def to_samna(self) -> Any:
        """
        to_samna converts the samna alias object to a samna object
        """
        return self.samna_object(samna.dynapse2.Dynapse2Parameter)


@dataclass
class Dynapse2Destination(SamnaAlias):
    """
    Dynapse2Destination mimics the address part of the samna AER package for DynapSE2

    :Parameters:

    :param core: the core mask used while sending the events
            [1,1,1,1] means all 4 cores are on the target
            [0,0,1,0] means the event will arrive at core 2 only
    :type core: List[bool]
    :param x_hop: number of chip hops on x axis
    :type x_hop: int
    :param y_hop: number of chip hops on y axis
    :type y_hop: int
    :param tag: globally multiplexed locally unique event tag which is used to identify the connection between two neurons.
    :type tag: int
    """

    core: List[bool]
    x_hop: int
    y_hop: int
    tag: int

    def __post_init__(self) -> None:
        """
        __post_init__ runs after initialization and checks if the data valid

        :raises ValueError: Core mask requires 4 entries!
        :raises ValueError: Cannot reach beyond +-7 chips in x axis
        :raises ValueError: Cannot reach beyond +-7 chips in y axis
        :raises ValueError: Illegal tag!
        """
        self.core = list(map(lambda e: bool(e), self.core))
        if len(self.core) != 4:
            raise ValueError("Core mask requires 4 entries!")
        if abs(self.x_hop) > 7:
            raise ValueError("Cannot reach beyond +-7 chips in x axis")
        if abs(self.y_hop) > 7:
            raise ValueError("Cannot reach beyond +-7 chips in y axis")
        if self.tag > 2048 or self.tag < 0:
            raise ValueError("Illegal tag!")

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Destination:
        """
        from_samna converts a `Dynapse2Destination` samna object to an alias object

        :param obj: a samna.dynapse2.Dynapse2Destination object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Destination
        """
        return cls(core=obj.core, x_hop=obj.x_hop, y_hop=obj.y_hop, tag=obj.tag)

    def __hash__(self) -> int:
        """
        __hash__ creates a unique hash code for the object which is used in sorting, and indexing
        the hash code is created stacking the datafield together in a 24 bit integer object

        | core   | xhop   | y_hop  | tag     |
        | ------ | ------ | ------ | ------- |
        | 4-bits | 4-bits | 4-bits | 12-bits |

        :return: _description_
        :rtype: int
        """
        hash = 0
        for bit in reversed(self.core):
            hash = (hash << 1) | bit
        hash = (hash << 4) | (self.x_hop + 7)
        hash = (hash << 4) | (self.y_hop + 7)
        hash = hash << 12 | self.tag
        return hash

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["core"] = self.jlist_regular(self.core)
        return wrapper

    def to_samna(self) -> Any:
        """
        to_samna converts the samna alias object to a samna object
        """
        return self.samna_object(samna.dynapse2.Dynapse2Destination)


@dataclass
class NormalGridEvent(SamnaAlias):
    """
    NormalGridEvent mimics the samna AER package for DynapSE2

    :Parameters:

    :param event: the destination of the package including routing information
    :type event: Dynapse2Destination
    :param timestamp: the timestamp of the event in microseconds (1e-6)
    :type timestamp: np.uint32
    """

    event: Dynapse2Destination
    timestamp: np.uint32

    @classmethod
    def from_samna(cls, obj: Any) -> NormalGridEvent:
        """
        from_samna converts a `NormalGridEvent` samna object to an alias object

        :param obj: a `samna.dynapse2.NormalGridEvent` object
        :type obj: Any
        :return: the samna alias version
        :rtype: NormalGridEvent
        """

        return cls(
            event=Dynapse2Destination.from_samna(obj.event), timestamp=obj.timestamp
        )

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["event"] = self.event.json_wrapper()
        return wrapper

    def to_samna(self) -> Any:
        """
        to_samna converts the samna alias object to a samna object
        """
        return self.samna_object(samna.dynapse2.NormalGridEvent)


# - Dynapse2Configuration - #


class ParamMap(dict):
    """
    ParamMap is in fact Dict[str, Dynapse2Parameter]
    it extends dict and implements from_samna method
    """

    _param_mapper = lambda kv: (kv[0], Dynapse2Parameter.from_samna(kv[1]))

    @classmethod
    def from_samna(cls, obj: Any) -> Dict[str, Dynapse2Parameter]:
        """
        from_samna converts a `Dict[str, Dynapse2Parameter]` samna object to an alias dictionary

        :param obj: a `Dict[str, samna.dynapse2.Dynapse2Parameter]` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dict[str, Dynapse2Parameter]
        """
        return dict(map(ParamMap._param_mapper, obj.items()))


@dataclass
class Dynapse2Configuration(SamnaAlias):
    """
    Dynapse2Configuration mimics the big `samna.dynapse2.Dynapse2Configuration` object which is used to configure the device
    It's the main object used to configure chip/core paramaters, weights, memory and so on.

    :Parameters:

    :param chips: a list of `Dynapse2Chip` objects
    :type chips: List[Dynapse2Chip]
    """

    chips: List[Dynapse2Chip]

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Configuration:
        """
        from_samna converts a `Dynapse2Configuration` samna object to an alias object

        :param obj: a `samna.dynapse2.Dynapse2Configuration` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Configuration
        """

        return cls(
            chips=[Dynapse2Chip.from_samna(chip) for chip in obj.chips],
        )

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["chips"] = [c.json_wrapper() for c in self.chips]
        return wrapper

    def to_samna(self) -> Any:
        """
        to_samna converts the samna alias object to a samna object
        """
        return self.samna_object(samna.dynapse2.Dynapse2Configuration)


@dataclass
class Dynapse2Chip(SamnaAlias):
    cores: List[Dynapse2Core]
    global_parameters: ParamMap
    shared_parameters01: ParamMap
    shared_parameters23: ParamMap
    sadc_group_parameters01: List[ParamMap]
    sadc_group_parameters23: List[ParamMap]
    sadc_enables: Dynapse2Chip_ConfigSadcEnables
    dvs_if: Dynapse2DvsInterface
    bioamps: Dynapse2Bioamps
    enable_pg0_reference_monitor: bool
    enable_pg1_reference_monitor: bool
    param_gen0_powerdown: bool
    param_gen1_powerdown: bool

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Chip:
        """
        from_samna converts a `Dynapse2Chip` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2Chip` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Chip
        """

        return cls(
            bioamps=Dynapse2Bioamps.from_samna(obj.bioamps),
            cores=[Dynapse2Core.from_samna(core) for core in obj.cores],
            dvs_if=Dynapse2DvsInterface.from_samna(obj.dvs_if),
            enable_pg0_reference_monitor=obj.enable_pg0_reference_monitor,
            enable_pg1_reference_monitor=obj.enable_pg1_reference_monitor,
            global_parameters=ParamMap.from_samna(obj.global_parameters),
            param_gen0_powerdown=obj.param_gen0_powerdown,
            param_gen1_powerdown=obj.param_gen1_powerdown,
            sadc_enables=Dynapse2Chip_ConfigSadcEnables.from_samna(obj.sadc_enables),
            sadc_group_parameters01=[
                ParamMap.from_samna(p_map) for p_map in obj.sadc_group_parameters01
            ],
            sadc_group_parameters23=[
                ParamMap.from_samna(p_map) for p_map in obj.sadc_group_parameters23
            ],
            shared_parameters01=ParamMap.from_samna(obj.shared_parameters01),
            shared_parameters23=ParamMap.from_samna(obj.shared_parameters23),
        )

    def json_wrapper(self) -> str:
        """json_wrapper overrides the base method"""
        wrapper = self.ctor
        wrapper["bioamps"] = self.bioamps.json_wrapper()
        wrapper["cores"] = self.jlist_alias(self.cores)
        wrapper["dvsIf"] = self.dvs_if.json_wrapper()
        wrapper["globalParameters"] = self.jdict_alias(self.global_parameters)
        wrapper["sadcEnables"] = self.sadc_enables.json_wrapper()
        wrapper["sadcGroupParameters01"] = self.jlist_dict_alias(
            self.sadc_group_parameters01
        )
        wrapper["sadcGroupParameters23"] = self.jlist_dict_alias(
            self.sadc_group_parameters23
        )
        wrapper["sharedParameters01"] = self.jdict_alias(self.shared_parameters01)
        wrapper["sharedParameters23"] = self.jdict_alias(self.shared_parameters23)
        return wrapper


@dataclass
class Dynapse2Chip_ConfigSadcEnables(SamnaAlias):
    nccf_extin_vi_group0_pg1: bool
    nccf_cal_refbias_v_group1_pg1: bool
    nccf_cal_refbias_v_group2_pg1: bool
    nccf_extin_vi_group2_pg1: bool
    nccf_extin_vi_group0_pg0: bool
    nccf_cal_refbias_v_group1_pg0: bool
    nccf_cal_refbias_v_group2_pg0: bool
    nccf_extin_vi_group2_pg0: bool

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Chip_ConfigSadcEnables:
        """
        from_samna converts a `Dynapse2Chip_ConfigSadcEnables` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2Chip_ConfigSadcEnables` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Chip_ConfigSadcEnables
        """

        return cls(
            nccf_cal_refbias_v_group1_pg0=obj.nccf_cal_refbias_v_group1_pg0,
            nccf_cal_refbias_v_group1_pg1=obj.nccf_cal_refbias_v_group1_pg1,
            nccf_cal_refbias_v_group2_pg0=obj.nccf_cal_refbias_v_group2_pg0,
            nccf_cal_refbias_v_group2_pg1=obj.nccf_cal_refbias_v_group2_pg1,
            nccf_extin_vi_group0_pg0=obj.nccf_extin_vi_group0_pg0,
            nccf_extin_vi_group0_pg1=obj.nccf_extin_vi_group0_pg1,
            nccf_extin_vi_group2_pg0=obj.nccf_extin_vi_group2_pg0,
            nccf_extin_vi_group2_pg1=obj.nccf_extin_vi_group2_pg1,
        )

    @property
    def ctor(self) -> Dict[str, Any]:
        """
        ctor creates a valid constructor to update a samna object using `from_json()` method
        convert the snake_case named variables to camelCase to be compatible with C++ methods

        :return: a dictionary of the object datastructure
        :rtype: Dict[str, Any]
        """
        # extend = lambda o: o.ctor if isinstance(o, SamnaAlias) else o
        return dict(
            map(
                lambda kv: (self.snake_to_camel(kv[0]), kv[1]),
                self.__dict__.items(),
            )
        )

    def snake_to_camel(self, name: str) -> str:
        """
        snake_to_camel converts a snake_case variable name to camelCase variable name

        :param name: the snake_case formatted variable name
        :type name: str
        :return: a camelCase formatted variable name
        :rtype: str
        """

        # Split the rest of the words
        name = "".join(word.title() for word in name.split("_"))
        return name


@dataclass
class Dynapse2Bioamps(SamnaAlias):
    common_parameters: ParamMap
    channel_parameters: List[ParamMap]
    param_gen2_powerdown: bool
    gain: int
    separate_parameters: bool
    monitor_channel_thuc: bool
    monitor_channel_thdc: bool
    monitor_channel_qfruc: bool
    monitor_channel_oruc: bool
    monitor_channel_osuc: bool
    monitor_channel_oauc: bool
    route: List[Dynapse2Destination]

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Bioamps:
        """
        from_samna converts a `Dynapse2Bioamps` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2Bioamps` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Bioamps
        """

        return cls(
            channel_parameters=[
                ParamMap.from_samna(p_map) for p_map in obj.channel_parameters
            ],
            common_parameters=ParamMap.from_samna(obj.common_parameters),
            gain=obj.gain,
            monitor_channel_oauc=obj.monitor_channel_oauc,
            monitor_channel_oruc=obj.monitor_channel_oruc,
            monitor_channel_osuc=obj.monitor_channel_osuc,
            monitor_channel_qfruc=obj.monitor_channel_qfruc,
            monitor_channel_thdc=obj.monitor_channel_thdc,
            monitor_channel_thuc=obj.monitor_channel_thuc,
            param_gen2_powerdown=obj.param_gen2_powerdown,
            route=[Dynapse2Destination.from_samna(dest) for dest in obj.route],
            separate_parameters=obj.separate_parameters,
        )

    def json_wrapper(self) -> str:
        """json_wrapper overrides the base method"""
        wrapper = self.ctor
        wrapper["channelParameters"] = self.jlist_dict_alias(self.channel_parameters)
        wrapper["commonParameters"] = self.jdict_alias(self.common_parameters)
        wrapper["route"] = self.jlist_alias(self.route)
        return wrapper


@dataclass
class Dynapse2DvsInterface(SamnaAlias):
    drop_events: bool
    dvs_mode: DvsMode
    off_events: bool
    on_events: bool
    pooling_shift: Vec2_unsigned_int
    copy_events: bool
    copy_hop: Vec2_int
    davis_req_ack_bugfix_delay_ns: int
    origin: Vec2_unsigned_int
    max: Vec2_unsigned_int
    pixel_destinations: List[Dynapse2Destination]
    filter: Dynapse2DvsFilter

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2DvsInterface:
        """
        from_samna converts a `Dynapse2DvsInterface` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2DvsInterface` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2DvsInterface
        """

        return cls(
            copy_events=obj.copy_events,
            copy_hop=Vec2_int.from_samna(obj.copy_hop),
            davis_req_ack_bugfix_delay_ns=obj.davis_req_ack_bugfix_delay_ns,
            drop_events=obj.drop_events,
            dvs_mode=DvsMode(obj.dvs_mode.value),
            filter=Dynapse2DvsFilter.from_samna(obj.filter),
            max=Vec2_unsigned_int.from_samna(obj.max),
            off_events=obj.off_events,
            on_events=obj.on_events,
            origin=Vec2_unsigned_int.from_samna(obj.origin),
            pixel_destinations=[
                Dynapse2Destination.from_samna(dest) for dest in obj.pixel_destinations
            ],
            pooling_shift=Vec2_unsigned_int.from_samna(obj.pooling_shift),
        )

    def snake_to_camel(self, name: str) -> str:
        """
        snake_to_camel converts a snake_case variable name to camelCase variable name

        :param name: the snake_case formatted variable name
        :type name: str
        :return: a camelCase formatted variable name
        :rtype: str
        """
        if name == "davis_req_ack_bugfix_delay_ns":
            return "davisReqAckBugfixDelay_ns"
        else:
            return super().snake_to_camel(name)

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["copyHop"] = self.copy_hop.json_wrapper()
        wrapper["pixelDestinations"] = self.jlist_alias(self.pixel_destinations)
        wrapper["copyHop"] = self.copy_hop.json_wrapper()
        wrapper["filter"] = []
        wrapper["max"] = self.max.json_wrapper()
        wrapper["origin"] = self.origin.json_wrapper()
        wrapper["pixelDestinations"] = self.jlist_alias(self.pixel_destinations)
        wrapper["poolingShift"] = self.pooling_shift.json_wrapper()
        return wrapper


@dataclass
class Vec2_int(SamnaAlias):
    x: int
    y: int

    @classmethod
    def from_samna(cls, obj: Any) -> Vec2_int:
        """
        from_samna converts a `Vec2_int` samna object to an alias object

        :param obj: a `samna.dynapse.Vec2_int` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Vec2_int
        """

        return cls(x=obj.x, y=obj.y)


@dataclass
class Vec2_unsigned_int(Vec2_int):
    pass


@dataclass
class Dynapse2DvsFilter(SamnaAlias):
    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def discard(self):
        pass

    @abstractmethod
    def pop(self):
        pass

    @abstractmethod
    def remove(self):
        pass

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2DvsFilter:
        """
        from_samna converts a `Dynapse2DvsFilter` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2DvsFilter` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2DvsFilter
        """

        return cls()


@dataclass
class Dynapse2Core(SamnaAlias):
    neurons: List[Dynapse2Neuron]
    neuron_monitoring_on: bool
    monitored_neuron: int
    parameters: ParamMap
    sadc_enables: Dynapse2Core_CoreSadcEnables
    enable_syaw_stdbuf_an: bool
    enable_pulse_extender_monitor1: bool
    enable_pulse_extender_monitor2: bool
    _id: int = 0
    _cookie: int = 0

    @abstractmethod
    def get_id(self) -> int:
        pass

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Core:
        """
        from_samna converts a `Dynapse2Core` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2Core` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Core
        """

        return cls(
            enable_pulse_extender_monitor1=obj.enable_pulse_extender_monitor1,
            enable_pulse_extender_monitor2=obj.enable_pulse_extender_monitor2,
            enable_syaw_stdbuf_an=obj.enable_syaw_stdbuf_an,
            monitored_neuron=obj.monitored_neuron,
            neuron_monitoring_on=obj.neuron_monitoring_on,
            neurons=[Dynapse2Neuron.from_samna(n) for n in obj.neurons],
            parameters=ParamMap.from_samna(obj.parameters),
            sadc_enables=Dynapse2Core_CoreSadcEnables.from_samna(obj.sadc_enables),
            _id=obj._id,
            _cookie=obj._cookie,
        )

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["neurons"] = self.jlist_alias(self.neurons)
        wrapper["parameters"] = self.jdict_alias(self.parameters)
        wrapper["sadcEnables"] = self.sadc_enables.json_wrapper()
        return wrapper


@dataclass
class Dynapse2Core_CoreSadcEnables(SamnaAlias):
    soif_mem: bool
    soif_refectory: bool
    soad_dpi: bool
    soca_dpi: bool
    deam_edpi: bool
    deam_idpi: bool
    denm_edpi: bool
    denm_idpi: bool
    dega_idpi: bool
    desc_idpi: bool
    sy_w42: bool
    sy_w21: bool
    soho_sogain: bool
    soho_degain: bool

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Core_CoreSadcEnables:
        """
        from_samna converts a `Dynapse2Core_CoreSadcEnables` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2Core_CoreSadcEnables` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Core_CoreSadcEnables
        """

        return cls(
            deam_edpi=obj.deam_edpi,
            deam_idpi=obj.deam_idpi,
            dega_idpi=obj.dega_idpi,
            denm_edpi=obj.denm_edpi,
            denm_idpi=obj.denm_idpi,
            desc_idpi=obj.desc_idpi,
            soad_dpi=obj.soad_dpi,
            soca_dpi=obj.soca_dpi,
            soho_degain=obj.soho_degain,
            soho_sogain=obj.soho_sogain,
            soif_mem=obj.soif_mem,
            soif_refectory=obj.soif_refectory,
            sy_w21=obj.sy_w21,
            sy_w42=obj.sy_w42,
        )

    def snake_to_camel(self, name: str) -> str:
        """
        snake_to_camel converts a snake_case variable name to camelCase variable name

        :param name: the snake_case formatted variable name
        :type name: str
        :return: a camelCase formatted variable name
        :rtype: str
        """

        # Split the rest of the words
        name = "".join(word.title() for word in name.split("_"))
        return name


@dataclass
class Dynapse2Neuron(SamnaAlias):
    synapses: List[Dynapse2Synapse]
    destinations: List[Dynapse2Destination]
    latch_so_dc: bool
    latch_so_adaptation: bool
    latch_soif_kill: bool
    latch_coho_ca_mem: bool
    latch_ho_enable: bool
    latch_ho_so_de: bool
    latch_deam_ampa: bool
    latch_denm_nmda: bool
    latch_de_conductance: bool
    latch_deam_alpha: bool
    latch_denm_alpha: bool
    latch_de_mux: bool
    latch_soif_type: bool
    latch_ho_active: bool

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Neuron:
        """
        from_samna converts a `Dynapse2Neuron` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2Neuron` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Neuron
        """

        return cls(
            destinations=[
                Dynapse2Destination.from_samna(dest) for dest in obj.destinations
            ],
            latch_coho_ca_mem=obj.latch_coho_ca_mem,
            latch_de_conductance=obj.latch_de_conductance,
            latch_de_mux=obj.latch_de_mux,
            latch_deam_alpha=obj.latch_deam_alpha,
            latch_deam_ampa=obj.latch_deam_ampa,
            latch_denm_alpha=obj.latch_denm_alpha,
            latch_denm_nmda=obj.latch_denm_nmda,
            latch_ho_active=obj.latch_ho_active,
            latch_ho_enable=obj.latch_ho_enable,
            latch_ho_so_de=obj.latch_ho_so_de,
            latch_so_adaptation=obj.latch_so_adaptation,
            latch_so_dc=obj.latch_so_dc,
            latch_soif_kill=obj.latch_soif_kill,
            latch_soif_type=obj.latch_soif_type,
            synapses=[Dynapse2Synapse.from_samna(syn) for syn in obj.synapses],
        )

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["synapses"] = self.jlist_alias(self.synapses)
        wrapper["destinations"] = self.jlist_alias(self.destinations)
        return wrapper


@dataclass
class Dynapse2Synapse(SamnaAlias):
    dendrite: Dendrite
    stp: bool
    weight: List[bool]
    precise_delay: bool
    mismatched_delay: bool
    tag: int

    def __post_init__(self):
        self.weight = list(map(lambda e: bool(e), self.weight))
        if len(self.weight) != 4:
            raise ValueError("Weight mask requires 4 entries!")

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Synapse:
        """
        from_samna converts a `Dynapse2Synapse` samna object to an alias object

        :param obj: a `samna.dynapse.Dynapse2Synapse` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Synapse
        """
        return cls(
            dendrite=Dendrite(obj.dendrite.value),
            mismatched_delay=obj.mismatched_delay,
            precise_delay=obj.precise_delay,
            stp=obj.stp,
            tag=obj.tag,
            weight=obj.weight,
        )

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["weight"] = self.jlist_regular(self.weight)
        return wrapper


# --- For Typehinting --- #


@dataclass
class DeviceInfo:

    daughter_board_num: np.uint8
    device_type_name: str
    logic_version: int
    serial_number: str
    usb_bus_number: np.uint32
    usb_device_address: np.uint32


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
