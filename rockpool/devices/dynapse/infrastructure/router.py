"""
Dynap-SE common router simulator 

split_from : router_aliased.py -> router.py @ 220509

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
09/05/2021

[] TODO : Implement samna aliases
[] TODO : connection alias option
[] TODO : Multiple CAMs defined between neurons
[] TODO : n_gate = 4, syn=none options for se2
"""
from __future__ import annotations
from dataclasses import dataclass

from typing import (
    Callable,
    Optional,
    Tuple,
    Any,
    Dict,
    Union,
    List,
)

import numpy as np

from rockpool.devices.dynapse.base import CoreKey, NeuronKey


Dynapse1Configuration = Any
Dynapse2Configuration = Any
Dynapse1Synapse = Any
Dynapse2Synapse = Any
Dynapse1Destination = Any
Dynapse2Destination = Any
SynDict = Dict[NeuronKey, List[Union[Dynapse1Synapse, Dynapse2Synapse]]]
DestDict = Dict[NeuronKey, List[Union[Dynapse1Destination, Dynapse2Destination]]]


@dataclass
class Router:
    """
    Router stores the weight_mask reading of the memory and the neuron-to-neuron connections indicated

    :param n_chips: number of chips installed in the system, defaults to None
    :type n_chips: np.uint8, optional
    :param n_cores: maximum number of cores that a chip has in the system, defaults to None
    :type n_cores: np.uint8, optional
    :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys, defaults to None
    :type idx_map: Dict[int, NeuronKey], optional
    :param weight_mask: A matrix of encoded bit masks representing bitselect values to select and dot product the base Iw currents (pre, post, gate), defaults to None
    :type weight_mask: np.ndarray, optional
    """

    n_chips: np.uint8 = None
    n_cores: np.uint8 = None
    idx_map: Dict[int, NeuronKey] = None
    weight_mask: np.ndarray = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after init and validates the router configuration

        :raises ValueError: number of chips given and indicated does not match
        :raises ValueError: number of cores indicated and given does not match
        """
        n_chips = len(np.unique(list(map(lambda nkey: nkey[0], self.idx_map.values()))))
        n_cores = len(np.unique(list(map(lambda nkey: nkey[1], self.idx_map.values()))))

        if self.n_chips < n_chips:
            raise ValueError(
                f"More than {self.n_chips} different chips ({n_chips}) found in  in active neuron list!"
            )

        if self.n_cores < n_cores:
            raise ValueError(
                f"More than {self.n_cores} different cores ({n_cores}) found in  in active neuron list!"
            )

    @classmethod
    def __from_samna(
        cls,
        config: Union[Dynapse1Configuration, Dynapse2Configuration],
        active_synapse: Callable[[Union[Dynapse1Synapse, Dynapse2Synapse]], bool],
        active_destination: Callable[
            [Union[Dynapse1Destination, Dynapse2Destination]], bool
        ],
        get_mem_connect: Callable[[SynDict, DestDict], MemConnect],
    ) -> Router:
        """
        __from_samna is the common configurable class factory method for both DynapSE1 and DynapSE2 architectures

        :param config: a samna configuration object used to configure all the system level properties
        :type config: Union[Dynapse1Configuration, Dynapse2Configuration]
        :param active_synapse: a method to identify active synapses from inactive synapses
        :type active_synapse: Callable[[Union[Dynapse1Synapse, Dynapse2Synapse]], bool]
        :param active_destination: a method to identify active destinations from inactive destinations
        :type active_destination: Callable[ [Union[Dynapse1Destination, Dynapse2Destination]], bool ]
        :param get_mem_connect: a method to get a device specific memory connector object
        :type get_mem_connect: Callable[[SynDict, DestDict], MemConnect]
        :return: a Router simulation object whose parameters are imported from a device configuration object
        :rtype: Router
        """

        synapses = {}
        destinations = {}

        # Traverse the chip for active neruon-neuron connections
        for h, chip in enumerate(config.chips):  # 1-4
            for c, core in enumerate(chip.cores):  # 4
                for n, neuron in enumerate(core.neurons):  # 256
                    syn_list = []
                    dest_list = []

                    # FAN-IN (64 connections) CAM
                    for syn in neuron.synapses:
                        if active_synapse(syn):
                            syn_list.append((syn))

                    # FAN-OUT (4 chips) SRAM
                    for dest in neuron.destinations:
                        if active_destination(dest):
                            dest_list.append(dest)

                    if syn_list:
                        synapses[(h, c, n)] = syn_list

                    if dest_list:
                        destinations[(h, c, n)] = dest_list

        connector = get_mem_connect(synapses, destinations)

        _mod = cls(
            n_chips=len(config.chips),
            n_cores=max([len(chip.cores) for chip in config.chips]),
            idx_map=connector.idx_map,
            weight_mask=connector.weight_mask,
        )
        return _mod

    @classmethod
    def from_Dynapse1Configuration(cls, config: Dynapse1Configuration) -> Router:
        """
        from_Dynapse1Configuration is a class factory method which uses Dynapse1Configuration object to extract Router simulator parameters

        :param config: a samna Dynapse1Configuration object used to configure all the system level parameters
        :type config: Dynapse1Configuration
        :return: a router simulator object whose parameters are imported from a device configuration object
        :rtype: Router
        """
        return cls.__from_samna(
            config=config,
            active_synapse=lambda syn: syn.listen_neuron_id != 0,
            active_destination=lambda dest: dest.target_chip_id != 16 and dest.in_use,
            get_mem_connect=MemConnect.from_content_se1,
        )

    @classmethod
    def from_Dynapse2Configuration(
        cls, config: Dynapse2Configuration, pos_map: Dict[int, Tuple[int]] = {0: (1, 0)}
    ) -> Router:
        """
        from_Dynapse2Configuration is a class factory method which uses Dynapse2Configuration object to extract Router simulator parameters
        :param config: a samna Dynapse2Configuration object used to configure all the system level parameters
        :type config: Dynapse2Configuration
        :param pos_map: a dictionary holding the relative coordinate positions of the chips installed, defaults to {0: (1, 0)}
        :type pos_map: Dict[int, Tuple[int]]
        :return: a router simulator object whose parameters are imported from a device configuration object
        :rtype: Router
        """
        if len(pos_map.keys()) != len(config.chips):
            raise IndexError(
                f"Position map does not represent the configuration object"
            )

        return cls.__from_samna(
            config=config,
            active_synapse=lambda syn: sum(syn.weight) > 0,
            active_destination=lambda dest: sum(dest.core) > 0,
            get_mem_connect=lambda syns, dests: MemConnect.from_content_se2(
                syns, dests, pos_map
            ),
        )


@dataclass
class MemConnect:
    """
    MemConnect encapsulates binary weight mask and index map which project
    the connections between neurons obtained from the device memory content

    :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys, defaults to None
    :type idx_map: Dict[int, NeuronKey], optional
    :param weight_mask: A matrix of encoded bit masks representing bitselect values to select and dot product the base Iw currents (pre, post, gate), defaults to None
    :type weight_mask: np.ndarray, optional
    """

    idx_map: Dict[int, NeuronKey]
    weight_mask: np.ndarray

    @classmethod
    def __from_content(
        cls,
        synapses: SynDict,
        destinations: DestDict,
        get_connector: Callable[[Dict[int, NeuronKey], __Connect]],
    ) -> MemConnect:
        """
        __from_content is the common configurable class factory method for both Dynap-SE1 and Dynap-SE2 architectures

        :param synapses: a dictionary of active synapses
        :type synapses: SynDict
        :param destinations: a dictionary of active destinations
        :type destinations: DestDict
        :param get_connector: a method to a device specific connector object
        :type get_connector: Callable[[Dict[int, NeuronKey], __Connect]]
        :return: a MemConnect object whose parameters are obtained from the active synapse and destination lists
        :rtype: MemConnect
        """

        idx_map = cls.__idx_map_from_content(synapses, destinations)
        syn_connections = cls.__connect_synapses(synapses, idx_map)

        # --- Device specific implementation --- #
        connector = get_connector(idx_map=idx_map)
        dest_connections = connector.connect_destinations(destinations)
        connections = connector.intersect_connections(syn_connections, dest_connections)
        # ---o--- #

        weight_mask = cls.__get_weight_mask(connections, n_neuron=len(idx_map))
        _mod = cls(idx_map=idx_map, weight_mask=weight_mask)
        return _mod

    @classmethod
    def from_content_se1(
        cls,
        synapses: Dict[NeuronKey, List[Dynapse1Synapse]],
        destinations: Dict[NeuronKey, List[Dynapse1Destination]],
    ) -> MemConnect:
        """
        from_content_se1 is a class factory method which uses Dynap-SE1 synapse and destination lists

        :param synapses: a dictionary of active synapses
        :type synapses: Dict[NeuronKey, List[Dynapse1Synapse]]
        :param destinations: a dictionary of active destinations
        :type destinations: Dict[NeuronKey, List[Dynapse1Destination]]
        :return: a MemConnect object whose parameters are obtained from the active synapse and destination lists
        :rtype: MemConnect
        """
        return cls.__from_content(
            synapses=synapses,
            destinations=destinations,
            get_connector=ConnectSE1.from_maps,
        )

    @classmethod
    def from_content_se2(
        cls,
        synapses: Dict[NeuronKey, List[Dynapse2Synapse]],
        destinations: Dict[NeuronKey, List[Dynapse2Destination]],
        pos_map: Dict[int, Tuple[int]],
    ) -> MemConnect:
        """
        from_content_se2 is a class factory method which uses Dynap-SE2 synapse and destination lists

        :param synapses: a dictionary of active synapses
        :type synapses: Dict[NeuronKey, List[Dynapse2Synapse]]
        :param destinations: a dictionary of active destinations
        :type destinations: Dict[NeuronKey, List[Dynapse2Destination]]
        :param pos_map: a dictionary holding the relative coordinate positions of the chips installed
        :type pos_map: Dict[int, Tuple[int]]
        :return: a MemConnect object whose parameters are obtained from the active synapse and destination lists
        :rtype: MemConnect
        """
        return cls.__from_content(
            synapses=synapses,
            destinations=destinations,
            get_connector=lambda idx_map: ConnectSE2.from_maps(
                idx_map=idx_map, pos_map=pos_map
            ),
        )

    @staticmethod
    def __idx_map_from_content(
        synapses: SynDict, destinations: DestDict
    ) -> Dict[int, NeuronKey]:
        """
        __idx_map_from_content obtains an index map from active synapse and destinations dictionaries

        :param synapses: a dictionary of active synapses
        :type synapses: SynDict
        :param destinations: a dictionary of active destinations
        :type destinations: DestDict
        :return: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :rtype: Dict[int, NeuronKey]
        """

        keys = list(synapses.keys()) if synapses is not None else []
        keys += list(destinations.keys()) if destinations is not None else []
        keys = sorted(list(set(keys))) if keys else []
        idx_map = dict(zip(range(len(keys)), keys))
        return idx_map

    @staticmethod
    def __connect_synapses(
        synapses: SynDict, idx_map: Dict[int, NeuronKey]
    ) -> List[Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]]]:
        """
        __connect_synapses creates a list of all possible connections between neurons indicated by the CAM content stored in synapse objects

        :param synapses: a dictionary of active synapses
        :type synapses: SynDict
        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :type idx_map: Dict[int, NeuronKey]
        :return: a list of all possible connections indicated by synapse object content
        :rtype: List[Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]]]
        """
        __conn = []
        r_idx_map = {v: k for k, v in idx_map.items()}

        for nkey, syn_list in synapses.items():
            for syn in syn_list:
                # Any indexed neuron can send spikes to the post-synaptic neuron
                __conn += [(pre_idx, r_idx_map[nkey], syn) for pre_idx in idx_map]

        return __conn

    @staticmethod
    def __get_weight_mask(
        connections: List[Tuple[int, int, int, int]], n_neuron: int, n_gate: int = 4
    ) -> np.ndarray:
        """
        __get_weight_mask creates and fills a weight mask object using a intersected connection list

        :param connections: the neuron to neuron connection list [(pre_idx, post_idx, gate_idx, weight_mask)]
        :type connections: List[Tuple[int, int, int, int]]
        :param n_neuron: number of neurons (1st and 2nd dimensions)
        :type n_neuron: int
        :param n_gate: then number of gates (3rd dimension), defaults to 4
        :type n_gate: int, optional
        :return: A matrix of encoded bit masks representing bitselect values to select and dot product the base Iw currents (pre, post, gate)
        :rtype: np.ndarray
        """
        weight_mask = np.zeros((n_neuron, n_neuron, n_gate))

        for pre, post, gate, weight in connections:
            weight_mask[pre, post, gate] = weight

        return weight_mask

    @staticmethod
    def int_to_binary(number: int, n_bits: Optional[int] = 0) -> List[bool]:
        """
        int_to_binary converts an integer value to its binary representation in the form of list of boolean values

            7 -> [1,1,1]
            8 -> [0,0,0,1]
            9 -> [1,0,0,1]

        :param number: the integer value to be decoded
        :type number: int
        :param n_bits: the minimum number of bits, defaults to 0
        :type n_bits: Optional[int], optional
        :raises ValueError: Numeric value should be greater than 0!
        :return: the binary representation of the numerical value
        :rtype: List[bool]
        """
        if number < 0:
            raise ValueError("Numeric value should be greater than 0!")
        __binary = []

        while number != 0:
            __binary.append(1) if number & 1 else __binary.append(0)
            number = number >> 1

        if n_bits > len(__binary):
            __binary += [0] * (n_bits - len(__binary))

        return __binary

    @staticmethod
    def binary_to_int(binary: List[bool]) -> int:
        """
        binary_to_int converts the binary representation of a number into an integer.

            [1,1,1] -> 7
            [0,0,0,1] -> 8
            [1,0,0,1] -> 9

        :param binary: the binary representation of a numerical value
        :type binary: List[bool]
        :return: the integer value of the binary
        :rtype: int
        """
        __number = 0
        for bit in reversed(binary):
            __number = (__number << 1) | bit
        return __number


@dataclass
class __Connect:
    """
    __Connect is a boilerplate class to be inherited by SE1 and SE2 modules for version specific connectivity implementation

    :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
    :type idx_map: Dict[int, NeuronKey], optional
    """

    idx_map: Dict[int, NeuronKey]

    def __post_init__(self) -> None:
        self.neuron_dict = self.neuron_dict_from_idx_map(self.idx_map)
        self.r_idx_map = {v: k for k, v in self.idx_map.items()}

    @staticmethod
    def neuron_dict_from_idx_map(idx_map) -> Dict[CoreKey, List[np.uint8]]:
        """
        neuron_dict_from_idx_map converts an index map to a dictionary that presents the active cores and their active neruons.

            core_key : List[neurons]
            (0,0) : [1,5,7,2]
            (0,1) : [0,18,201]

        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :type idx_map: _type_
        :return: a dictionary of the mapping between active cores and list of active neurons
        :rtype: Dict[CoreKey, List[np.uint8]]
        """
        __dict: Dict[CoreKey, List[np.uint8]] = {}
        for h, c, n in idx_map.values():
            if (h, c) not in __dict:
                __dict[h, c] = [n]
            else:
                __dict[h, c].append(n)

        return __dict

    def connect_destinations(
        self, *args, **kwargs
    ) -> List[int, int, Union[Dynapse1Destination, Dynapse2Destination]]:
        """
        connect_destinations is an abstract method. The class inheriting `__Connect` should implement this

        :return: a list of possible connections indicated by destination object content
        :rtype: List[int, int, Union[Dynapse1Destination, Dynapse2Destination]]
        """
        raise NotImplementedError("Version specific implementation required!")

    def intersect_connections(self, *args, **kwargs) -> List[Tuple[int, int, int, int]]:
        """
        connect_destinations is an abstract method. The class inheriting `__Connect` should implement this

        :return: the neuron to neuron connection list [(pre_idx, post_idx, gate_idx, weight_mask)]
        :rtype: List[Tuple[int, int, int, int]]
        """
        raise NotImplementedError("Version specific implementation required!")


@dataclass
class ConnectSE1(__Connect):
    """
    ConnectSE1 encapsulates Dynap-SE1 specific router connectivity utilities

    :Attributes:

    :attr syn_weight_map: Dynap-SE1 does not have connection specific weight mask facility.
        Instead, it has fixed current parameters per synapse gate. Therefore, syn_weight_map
        has the mapping from the synapse gate index to definite weight mask.
    :type syn_weight_map: Dict[int, int]
    """

    syn_weight_map = {
        0: 0b0001,  # GABA
        1: 0b0010,  # SHUNT
        2: 0b0100,  # NMDA
        3: 0b1000,  # AMPA
    }

    @classmethod
    def from_maps(cls, idx_map: Dict[int, NeuronKey]) -> ConnectSE1:
        """
        from_maps is a class factory method which construct a `ConnectSE1` object using index map

        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :type idx_map: Dict[int, NeuronKey]
        :return: a ConnectSE1 object obtained from the index map
        :rtype: ConnectSE1
        """
        return cls(idx_map=idx_map)

    @staticmethod
    def target_core_keys(dest: Dynapse1Destination) -> List[CoreKey]:
        """
        target_core_keys lists the cores that the destination object provided aims to convey the events.

        :param dest: DynapSE1 destination object containing SRAM content
        :type dest: Dynapse1Destination
        :return: list of cores targeted
        :rtype: List[CoreKey]
        """
        core = MemConnect.int_to_binary(dest.core_mask)
        dest_cores = np.argwhere(core).flatten()
        target_list = [(dest.target_chip_id, dest_core) for dest_core in dest_cores]
        return target_list

    def connect_destinations(
        self, destinations: Dict[NeuronKey, List[Dynapse1Destination]]
    ) -> List[Tuple[int, int, Dynapse1Destination]]:
        """
        connect_destinations creates a list of all possible connections between neurons indicated by the SRAM content stored in the destination object.

        :param destinations: a dictionary of an active destinations
        :type destinations: Dict[NeuronKey, List[Dynapse1Destination]]
        :return: a list of all possible connections indicated by destination object content
        :rtype: List[Tuple[int, int, Dynapse1Destination]]
        """

        candidates = []

        for (h, c, n), dest_list in destinations.items():
            for dest in dest_list:
                cv = dest.virtual_core_id

                # Get target cores from the destination object
                for th, tc in self.target_core_keys(dest):

                    # If the core indicated has some active neurons
                    if (th, tc) in self.neuron_dict:
                        neurons = self.neuron_dict[th, tc]

                        # Extend candidate list
                        if (h, cv, n) in self.r_idx_map:
                            candidates += [
                                (
                                    self.r_idx_map[h, cv, n],
                                    self.r_idx_map[th, tc, tn],
                                    dest,
                                )
                                for tn in neurons
                            ]

        return candidates

    def intersect_connections(
        self,
        syn_connections: List[Tuple[int, int, Dynapse1Synapse]],
        dest_connections: List[Tuple[int, int, Dynapse1Destination]],
    ) -> List[Tuple[int, int, int, int]]:
        """
        intersect_connections finds the common connections infered by both the SRAM and CAM.
        That is if a connection is allowed in the CAMs and it's listed in the SRAMs,
        there is a link between those two neurons. If one sends an event, the other one catches.
        `intersect_connections` finds those links.

        :param syn_connections: a list of all possible connections indicated by synapse object content
        :type syn_connections: List[Tuple[int, int, Dynapse1Synapse]]
        :param dest_connections: a list of all possible connections indicated by destination object content
        :type dest_connections: List[Tuple[int, int, Dynapse1Destination]]
        :return: the neuron to neuron connection list [(pre_idx, post_idx, gate_idx, weight_mask)]
        :rtype: List[Tuple[int, int, int, int]]
        """

        def __match_dest_connection(
            dest_connection: Tuple[int, int, Dynapse1Destination]
        ) -> Tuple[int, int, np.uint8, np.uint8]:
            """
            __match_dest_connection processes the destination connection candidates and creates a comparable identity

            :param dest_connection:  a connection candidate that is obtained from a destination object
            :type dest_connection: Tuple[int, int, Dynapse1Destination]
            :return: a standard form connection candidate which can be compared against any other SE1 connection candidate
            :rtype: Tuple[int, int, np.uint8, np.uint8]
            """
            pre_idx, post_idx, dest = dest_connection
            pre_chip_idx, pre_core_idx, pre_neuron_idx = self.idx_map[pre_idx]
            return pre_idx, post_idx, pre_core_idx, pre_neuron_idx

        def __match_syn_connection(
            syn_connection: Tuple[int, int, Dynapse1Synapse]
        ) -> Tuple[int, int, np.uint8, np.uint8]:
            """
            __match_syn_connection processes the synapse connection candidates and creates a comparable identity

            :param syn_connection: A connection candidate that is obtained from a synapse object
            :type syn_connection: Tuple[int, int, Dynapse1Synapse]
            :return: a standard from connection candidate which can be compared against any other SE1 connection candidate
            :rtype: Tuple[int, int, np.uint8, np.uint8]
            """
            pre_idx, post_idx, syn = syn_connection
            return pre_idx, post_idx, syn.listen_core_id, syn.listen_neuron_id

        def __read_syn_connection(
            syn_connection: Tuple[int, int, Dynapse1Synapse]
        ) -> Tuple[int, int, int, int]:
            """
            __read_syn_connection processes the synapse connection identities and returns a version-invariant connection list identity

            :param syn_connection: A connection candidate that is obtained from a synapse object
            :type syn_connection: Tuple[int, int, Dynapse1Synapse]
            :return: a version invariant connection list identity
            :rtype: Tuple[int, int, int, int]
            """
            pre_idx, post_idx, syn = syn_connection
            gate = syn.syn_type.value
            weight = ConnectSE1.syn_weight_map[syn.syn_type.value]
            return pre_idx, post_idx, gate, weight

        dest_candidates = set(map(__match_dest_connection, dest_connections))

        __intersection = [
            __read_syn_connection(connection)
            for connection in syn_connections
            if __match_syn_connection(connection) in dest_candidates
        ]
        return __intersection


@dataclass
class ConnectSE2(__Connect):
    """
    ConnectSE2  encapsulates Dynap-SE2 specific router connectivity utilities

    :Attributes:

    :attr pos_map: a dictionary holding the relative coordinate positions of the chips installed
    :type pos_map: Dict[int, Tuple[int]]
    :attr syn_map: the mapping between Dynap-SE2 binary synapse values and gate indices
    :type syn_map: Dict[int, int]
    """

    pos_map: Dict[int, Tuple[int]]
    syn_map = {
        1024: 3,  # AMPA
        512: 0,  # GABA
        256: 2,  # NMDA
        128: 1,  # SHUNT
    }

    @classmethod
    def from_maps(
        cls, idx_map: Dict[int, NeuronKey], pos_map: Dict[int, Tuple[int]]
    ) -> ConnectSE2:
        """
        from_maps is a class factory method which construct a `ConnectSE2` object using index and position maps

        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :type idx_map: Dict[int, NeuronKey]
        :param pos_map: a dictionary holding the relative coordinate positions of the chips installed
        :type pos_map: Dict[int, Tuple[int]]
        :return: a ConnectSE2 object obtained from the index and position maps
        :rtype: ConnectSE2
        """
        return cls(idx_map=idx_map, pos_map=pos_map)

    def route_AER(self, src_chip: np.uint8, dest: Dynapse2Destination) -> int:
        """
        route_AER finds the destination chip id using the destination object and the position map

        :param src_chip: the departure chip ID
        :type src_chip: np.uint8
        :param dest: Dynap-SE2 destination object containing SRAM content
        :type dest: Dynapse2Destination
        :return: the arrival chip ID
        :rtype: int
        """
        r_pos_map = {v: k for k, v in self.pos_map.items()}

        x_src, y_src = self.pos_map[src_chip]
        x_dest = x_src + dest.x_hop
        y_dest = y_src + dest.y_hop

        if (x_dest, y_dest) in r_pos_map:
            chip_id = r_pos_map[x_dest, y_dest]
            return chip_id
        else:
            return -1

    def target_core_keys(
        self, src_chip: np.uint8, dest: Dynapse2Destination
    ) -> List[CoreKey]:
        """
        target_core_keys lists the cores that the destination object provided aims to convey the events.

        :param src_chip: the departure chip ID
        :type src_chip: np.uint8
        :param dest: DynapSE2 destination object containing SRAM content
        :type dest: Dynapse1Destination
        :return: list of cores targeted
        :rtype: List[CoreKey]
        """

        dest_chip = self.route_AER(src_chip, dest)
        dest_cores = np.argwhere(dest.core).flatten()
        target_list = [(dest_chip, dest_core) for dest_core in dest_cores]
        return target_list

    def connect_destinations(
        self, destinations: Dict[NeuronKey, List[Dynapse2Destination]]
    ) -> List[Tuple[int, int, Dynapse2Destination]]:
        """
        connect_destinations creates a list of all possible connections between neurons indicated by the SRAM content stored in the destination object.

        :param destinations: a dictionary of an active destinations
        :type destinations: Dict[NeuronKey, List[Dynapse2Destination]]
        :return: a list of all possible connections indicated by destination object content
        :rtype: List[Tuple[int, int, Dynapse2Destination]]
        """
        candidates = []

        for (h, c, n), dest_list in destinations.items():
            for dest in dest_list:

                # Get target cores from the source chip and the destination object
                for th, tc in self.target_core_keys(h, dest):

                    # If the core indicated has some active neurons
                    if (th, tc) in self.neuron_dict:
                        neurons = self.neuron_dict[th, tc]

                        # Extend candidate list
                        candidates += [
                            (self.r_idx_map[h, c, n], self.r_idx_map[th, tc, tn], dest)
                            for tn in neurons
                        ]

        return candidates

    @staticmethod
    def intersect_connections(
        syn_connections: List[Tuple[int, int, Dynapse2Synapse]],
        dest_connections: List[Tuple[int, int, Dynapse2Destination]],
    ) -> List[Tuple[int, int, int, int]]:
        """
        intersect_connections finds the common connections infered by both the SRAM and CAM.
        That is if a connection is allowed in the CAMs and it's listed in the SRAMs,
        there is a link between those two neurons. If one sends an event, the other one catches.
        `intersect_connections` finds those links.

        :param syn_connections: a list of all possible connections indicated by synapse object content
        :type syn_connections: List[Tuple[int, int, Dynapse2Synapse]]
        :param dest_connections: a list of all possible connections indicated by destination object content
        :type dest_connections: List[Tuple[int, int, Dynapse2Destination]]
        :return: the neuron to neuron connection list [(pre_idx, post_idx, gate_idx, weight_mask)]
        :rtype: List[Tuple[int, int, int, int]]
        """

        def __match_dest_connection(
            dest_connection: Tuple[int, int, Dynapse2Destination]
        ) -> Tuple[int, int, np.uint8, np.uint8]:
            """
            __match_dest_connection processes the destination connection candidates and creates a comparable identity

            :param dest_connection:  a connection candidate that is obtained from a destination object
            :type dest_connection: Tuple[int, int, Dynapse2Destination]
            :return: a standard form connection candidate which can be compared against any other SE2 connection candidate
            :rtype: Tuple[int, int, np.uint8, np.uint8]
            """
            pre_idx, post_idx, dest = dest_connection
            return pre_idx, post_idx, dest.tag

        def __match_syn_connection(
            syn_connection: Tuple[int, int, Dynapse2Synapse]
        ) -> Tuple[int, int, np.uint8, np.uint8]:
            """
            __match_syn_connection processes the synapse connection candidates and creates a comparable identity

            :param syn_connection: A connection candidate that is obtained from a synapse object
            :type syn_connection: Tuple[int, int, Dynapse2Synapse]
            :return: a standard from connection candidate which can be compared against any other SE2 connection candidate
            :rtype: Tuple[int, int, np.uint8, np.uint8]
            """
            pre_idx, post_idx, syn = syn_connection
            return pre_idx, post_idx, syn.tag

        def __read_syn_connection(
            syn_connection: Tuple[int, int, Dynapse2Synapse]
        ) -> Tuple[int, int, int, int]:
            """
            __read_syn_connection processes the synapse connection identities and returns a version-invariant connection list identity

            :param syn_connection: A connection candidate that is obtained from a synapse object
            :type syn_connection: Tuple[int, int, Dynapse2Synapse]
            :return: a version invariant connection list identity
            :rtype: Tuple[int, int, int, int]
            """
            pre_idx, post_idx, syn = syn_connection
            gate = ConnectSE2.syn_map[syn.dendrite.value]
            weight = MemConnect.binary_to_int(syn.weight)
            return pre_idx, post_idx, gate, weight

        dest_candidates = set(map(__match_dest_connection, dest_connections))

        __intersection = [
            __read_syn_connection(connection)
            for connection in syn_connections
            if __match_syn_connection(connection) in dest_candidates
        ]
        return __intersection
