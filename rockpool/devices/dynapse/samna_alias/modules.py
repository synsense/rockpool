"""
Dynap-SE2 samna alias implementations

The modules listed here do not replace the exact samna classes.
All modules listed copy the data segments of the samna classes and provide type hints.
The package is not aimed to be maintained long-term. 
To be removed at some point when the missing functionality is shipped to samna.

* Non User Facing *
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass

import numpy as np
import logging

from .base import SamnaAlias
from .definitions import ParameterType, DvsMode, Dendrite

SAMNA_AVAILABLE = False

try:
    import samna

    SAMNA_AVAILABLE = True
except:
    samna = Any
    logging.warning(
        "samna installation not found in the environtment! You can still work with simulator to full extend but the objects cannot be converted to device configuration."
    )

    __all__ = [
        "Dynapse2Parameter",
        "Dynapse2Destination",
        "NormalGridEvent",
        "ParamMap",
        "Dynapse2Configuration",
        "Dynapse2Chip",
        "Dynapse2Chip_ConfigSadcEnables",
        "Dynapse2Bioamps",
        "Dynapse2DvsInterface",
        "Vec2_int",
        "Vec2_unsigned_int",
        "Dynapse2DvsFilter",
        "Dynapse2Core",
        "Dynapse2Core_CoreSadcEnables",
        "Dynapse2Neuron",
        "Dynapse2Synapse",
        "DeviceInfo",
        "Dynapse2Interface",
        "Dynapse2Model",
    ]


@dataclass
class Dynapse2Parameter(SamnaAlias):
    """
    Dynapse2Parameter mimics the parameter object for Dynap-SE2.
    Converting to samna.dynapse2.Dynapse2Parameter is not recommended!
    """

    type: str
    """integer coarse base value :math:`C \\in [0,5]`"""

    coarse_value: np.uint8
    """integer fine value to scale the coarse current :math:`f \\in [0,255]`"""

    fine_value: np.uint8
    """the type of the parameter : N or P"""

    _address: np.uint64 = 0
    """stores PG number and branch shifted appropriately for the hardware, defaults to 0"""

    _cookie: np.uint64 = 0
    """a cookie(number) assigned to parameter with regards to the address, defaults to 0"""

    _initial_type: ParameterType = None
    """the initial type of the transistor, defaults to None"""

    _switchable_type: bool = False
    """set true for type changeable "..._V" parameters, defaults to False"""

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
    """

    core: List[bool]
    """the core mask used while sending the events"""

    x_hop: int
    """number of chip hops on x axis"""

    y_hop: int
    """number of chip hops on y axis"""

    tag: int
    """globally multiplexed locally unique event tag which is used to identify the connection between two neurons."""

    def __post_init__(self) -> None:
        """
        __post_init__ runs after initialization and checks if the data valid

        :raises ValueError: Core mask requires 4 entries!
        :raises ValueError: Cannot reach beyond +-7 chips in x axis
        :raises ValueError: Cannot reach beyond +-7 chips in y axis
        :raises ValueError: Illegal tag!
        """
        if self.core is not None:
            self.core = list(map(lambda e: bool(e), self.core))
            if len(self.core) != 4:
                raise ValueError("Core mask requires 4 entries!")
        if self.x_hop is not None and abs(self.x_hop) > 7:
            raise ValueError("Cannot reach beyond +-7 chips in x axis")
        if self.y_hop is not None and abs(self.y_hop) > 7:
            raise ValueError("Cannot reach beyond +-7 chips in y axis")
        if self.tag is not None and (self.tag > 2048 or self.tag < 0):
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
    """

    event: Dynapse2Destination
    """the destination of the package including routing information"""

    timestamp: np.uint32
    """the timestamp of the event in microseconds (1e-6)"""

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
    """

    chips: List[Dynapse2Chip]
    """a list of `Dynapse2Chip` objects"""

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
    """
    Dynapse2Chip copies the data segment of a `samna.dynapse2.Dynapse2Chip` object
    """

    cores: List[Dynapse2Core]
    """list of `Dynapse2Core` objects"""

    global_parameters: ParamMap
    """param settings of R2R monitoring output buffers: R2R_BUFFER_CCB and R2R_BUFFER_AMPB"""

    shared_parameters01: ParamMap
    """some calibration parameters shared between core 0 and 1"""

    shared_parameters23: ParamMap
    """some calibration parameters shared between core 2 and 3"""

    sadc_group_parameters01: List[ParamMap]
    """current-based spiking analog to digital converters (sADC) ensure easy monitoring of all relevant neural signals. This group paarameters set some calibration parameters shared between core 0 and 1"""

    sadc_group_parameters23: List[ParamMap]
    """current-based spiking analog to digital converters (sADC) ensure easy monitoring of all relevant neural signals. This group paarameters set some calibration parameters shared between core 2 and 3"""

    sadc_enables: Dynapse2Chip_ConfigSadcEnables
    """boolean switches to set for current-based spiking analog to digital converters (sADC)"""

    dvs_if: Dynapse2DvsInterface
    """DVS sensor interface object"""

    bioamps: Dynapse2Bioamps
    """Bio amplifiers setting"""

    enable_pg0_reference_monitor: bool
    """parameter generator 0 ref monitoring (do not know what it means)"""

    enable_pg1_reference_monitor: bool
    """parameter generator 1 ref monitoring (do not know what it means)"""

    param_gen0_powerdown: bool
    """parameter generator 0 power down (do not know what it means)"""

    param_gen1_powerdown: bool
    """parameter generator 1 power down (do not know what it means)"""

    @classmethod
    def from_samna(cls, obj: Any) -> Dynapse2Chip:
        """
        from_samna converts a `Dynapse2Chip` samna object to an alias object

        :param obj: a `samna.dynapse2.Dynapse2Chip` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Chip
        """

        return cls(
            cores=[Dynapse2Core.from_samna(core) for core in obj.cores],
            global_parameters=ParamMap.from_samna(obj.global_parameters),
            shared_parameters01=ParamMap.from_samna(obj.shared_parameters01),
            shared_parameters23=ParamMap.from_samna(obj.shared_parameters23),
            sadc_group_parameters01=[
                ParamMap.from_samna(p_map) for p_map in obj.sadc_group_parameters01
            ],
            sadc_group_parameters23=[
                ParamMap.from_samna(p_map) for p_map in obj.sadc_group_parameters23
            ],
            sadc_enables=Dynapse2Chip_ConfigSadcEnables.from_samna(obj.sadc_enables),
            dvs_if=Dynapse2DvsInterface.from_samna(obj.dvs_if),
            bioamps=Dynapse2Bioamps.from_samna(obj.bioamps),
            enable_pg0_reference_monitor=obj.enable_pg0_reference_monitor,
            enable_pg1_reference_monitor=obj.enable_pg1_reference_monitor,
            param_gen0_powerdown=obj.param_gen0_powerdown,
            param_gen1_powerdown=obj.param_gen1_powerdown,
        )

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["cores"] = self.jlist_alias(self.cores)
        wrapper["globalParameters"] = self.jdict_alias(self.global_parameters)
        wrapper["sharedParameters01"] = self.jdict_alias(self.shared_parameters01)
        wrapper["sharedParameters23"] = self.jdict_alias(self.shared_parameters23)
        wrapper["sadcGroupParameters01"] = self.jlist_dict_alias(
            self.sadc_group_parameters01
        )
        wrapper["sadcGroupParameters23"] = self.jlist_dict_alias(
            self.sadc_group_parameters23
        )
        wrapper["sadcEnables"] = self.sadc_enables.json_wrapper()
        wrapper["dvsIf"] = self.dvs_if.json_wrapper()
        wrapper["bioamps"] = self.bioamps.json_wrapper()
        return wrapper

    def to_samna(self) -> Any:
        """
        to_samna converts the samna alias object to a samna object
        """
        return self.samna_object(samna.dynapse2.Dynapse2Chip)


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

        :param obj: a `samna.dynapse2.Dynapse2Chip_ConfigSadcEnables` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Chip_ConfigSadcEnables
        """

        return cls(
            nccf_extin_vi_group0_pg1=obj.nccf_extin_vi_group0_pg1,
            nccf_cal_refbias_v_group1_pg1=obj.nccf_cal_refbias_v_group1_pg1,
            nccf_cal_refbias_v_group2_pg1=obj.nccf_cal_refbias_v_group2_pg1,
            nccf_extin_vi_group2_pg1=obj.nccf_extin_vi_group2_pg1,
            nccf_extin_vi_group0_pg0=obj.nccf_extin_vi_group0_pg0,
            nccf_cal_refbias_v_group1_pg0=obj.nccf_cal_refbias_v_group1_pg0,
            nccf_cal_refbias_v_group2_pg0=obj.nccf_cal_refbias_v_group2_pg0,
            nccf_extin_vi_group2_pg0=obj.nccf_extin_vi_group2_pg0,
        )

    def snake_to_camel(self, name: str) -> str:
        """
        snake_to_camel overrides the base method.
        This class requires special attention because the conversion is not as straightforward.
        The first letters are also capitalized
        That's because in samna it's defined as a strcut whose members defined full capital letters
        """

        # Split the rest of the words
        name = "".join(word.title() for word in name.split("_"))
        return name

    def to_samna(self) -> Any:
        """
        to_samna converts the samna alias object to a samna object
        """
        return self.samna_object(samna.dynapse2.Dynapse2Chip_ConfigSadcEnables)


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

        :param obj: a `samna.dynapse2.Dynapse2Bioamps` object
        :type obj: Any
        :return: the samna alias version
        :rtype: Dynapse2Bioamps
        """

        return cls(
            common_parameters=ParamMap.from_samna(obj.common_parameters),
            channel_parameters=[
                ParamMap.from_samna(p_map) for p_map in obj.channel_parameters
            ],
            param_gen2_powerdown=obj.param_gen2_powerdown,
            gain=obj.gain,
            separate_parameters=obj.separate_parameters,
            monitor_channel_thuc=obj.monitor_channel_thuc,
            monitor_channel_thdc=obj.monitor_channel_thdc,
            monitor_channel_qfruc=obj.monitor_channel_qfruc,
            monitor_channel_oruc=obj.monitor_channel_oruc,
            monitor_channel_osuc=obj.monitor_channel_osuc,
            monitor_channel_oauc=obj.monitor_channel_oauc,
            route=[Dynapse2Destination.from_samna(dest) for dest in obj.route],
        )

    def json_wrapper(self) -> str:
        """
        json_wrapper overrides the base method
        """
        wrapper = self.ctor
        wrapper["channelParameters"] = self.jlist_dict_alias(self.channel_parameters)
        wrapper["commonParameters"] = self.jdict_alias(self.common_parameters)
        wrapper["route"] = self.jlist_alias(self.route)
        return wrapper

    def to_samna(self) -> Any:
        """
        to_samna converts the samna alias object to a samna object
        """
        return self.samna_object(samna.dynapse2.Dynapse2Bioamps)


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
        if self.weight is not None:
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

    def to_samna(self) -> Any:
        """
        to_samna converts the samna alias object to a samna object
        """
        return self.samna_object(samna.dynapse2.Dynapse2Synapse)


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
