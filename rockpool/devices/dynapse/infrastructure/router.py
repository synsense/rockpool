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
[] TODO : common intersect connections
[] TODO : FIX NONE = AMPA
[] TODO : Solve pos_map issue
"""
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod

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

from rockpool.devices.dynapse.definitions import CoreKey, NeuronKey
from rockpool.devices.dynapse.default import NUM_CORES

from rockpool.devices.dynapse.samna_alias import (
    Dynapse1Synapse,
    Dynapse1Destination,
    Dynapse1Configuration,
    Dynapse2Synapse,
    Dynapse2Destination,
    Dynapse2Configuration,
)

SynDict = Dict[NeuronKey, List[Union[Dynapse1Synapse, Dynapse2Synapse]]]
DestDict = Dict[NeuronKey, List[Union[Dynapse1Destination, Dynapse2Destination]]]


@dataclass
class Router:
    """
    Router stores the weight_mask readings of the memory and the neuron-to-neuron connections indicated

    :param n_chips: number of chips installed in the system, defaults to None
    :type n_chips: np.uint8, optional
    :param core_map:a dictionary of the mapping between active cores and list of active neurons, defaults to None
    :type core_map: Dict[CoreKey, List[np.uint8]], optional
    :param tag_map_in: a dictionary of the mapping between matrix indexes of the incoming events and their tags. Used to interpret the input weight matrix, defaults to None
    :type tag_map_in: Dict[int, int], optional
    :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys. Used to interpret the recurrent weight matrix, defaults to None
    :type idx_map: Dict[int, NeuronKey], optional
    :param tag_map_out: a dictionary of the mapping between matrix indexes of the outgoing events and their tags. Used to interpret the output weight matrix, defaults to None
    :type tag_map_out: Dict[int, int], optional
    :param w_in_mask: A matrix of encoded bit masks representing bitselect values to select and dot product the base Iw currents (tag_in, post, gate), for input connections, defaults to None
    :type w_in_mask: np.ndarray, optional
    :param w_rec_mask: A matrix of encoded bit masks representing bitselect values to select and dot product the base Iw currents (pre, post, gate), for recurrent connections, defaults to None
    :type w_rec_mask: np.ndarray, optional
    :param w_out_mask: A boolean output mask revealing the relation between neurons and their sram tags, (pre, tag_out) defaults to None
    :type w_out_mask: np.ndarray, optional
    """

    n_chips: np.uint8 = None
    shape: Tuple[int] = None
    core_map: Dict[CoreKey, List[np.uint8]] = None
    tag_map_in: Dict[int, int] = None
    idx_map: Dict[int, NeuronKey] = None
    tag_map_out: Dict[int, int] = None
    w_in_mask: np.ndarray = None
    w_rec_mask: np.ndarray = None
    w_out_mask: np.ndarray = None

    def __post_init__(self) -> None:
        """
        __post_init__ runs after init and validates the router configuration

        :raises ValueError: number of chips given and indicated does not match
        :raises ValueError: number of cores indicated and given does not match
        """
        n_chips = len(np.unique(list(map(lambda nkey: nkey[0], self.idx_map.values()))))

        if self.n_chips < n_chips:
            raise ValueError(
                f"More than {self.n_chips} different chips ({n_chips}) found in  in active neuron list!"
            )

    @classmethod
    def __from_samna(
        cls,
        config: Union[Dynapse1Configuration, Dynapse2Configuration],
        active_synapse: Callable[[Union[Dynapse1Synapse, Dynapse2Synapse]], bool],
        active_destination: Callable[
            [Union[Dynapse1Destination, Dynapse2Destination]], bool
        ],
        get_connector: Callable[[SynDict, DestDict], Connector],
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
        :type get_mem_connect: Callable[[SynDict, DestDict], Connector]
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

        connector: Connector = get_connector(synapses, destinations)

        _mod = cls(
            n_chips=len(config.chips),
            shape=connector.shape,
            core_map=connector.get_core_map(),
            tag_map_in=connector.get_tag_map_in(),
            idx_map=connector.get_idx_map(),
            tag_map_out=connector.get_tag_map_out(),
            w_in_mask=connector.get_w_in_mask(),
            w_rec_mask=connector.get_w_rec_mask(),
            w_out_mask=connector.get_w_out_mask(),
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
            get_connector=lambda syns, dests: ConnectorSE1(syns, dests),
        )

    @classmethod
    def from_Dynapse2Configuration(cls, config: Dynapse2Configuration) -> Router:
        """
        from_Dynapse2Configuration is a class factory method which uses Dynapse2Configuration object to extract Router simulator parameters
        :param config: a samna Dynapse2Configuration object used to configure all the system level parameters
        :type config: Dynapse2Configuration
        :param pos_map: a dictionary holding the relative coordinate positions of the chips installed, defaults to {0: (1, 0)}
        :type pos_map: Dict[int, Tuple[int]]
        :return: a router simulator object whose parameters are imported from a device configuration object
        :rtype: Router
        """

        return cls.__from_samna(
            config=config,
            active_synapse=lambda syn: sum(syn.weight) > 0,
            active_destination=lambda dest: sum(dest.core) > 0,
            get_connector=lambda syns, dests: ConnectorSE2(syns, dests),
        )


@dataclass
class Connector(ABC):
    """
    Connector encapsulates binary weight mask and index map which project the connections between neurons obtained from the device memory content
    It also provide a abstract base class for device specific implementations

    :param synapses: a dictionary of active synapses
    :type synapses: SynDict
    :param destinations: a dictionary of active destinations
    :type destinations: DestDict
    """

    synapses: Optional[SynDict] = None
    destinations: Optional[DestDict] = None

    def __post_init__(self) -> None:
        self.__idx_map = None
        self.__core_map = None
        self.__tag_map_in = None
        self.__tag_map_out = None
        self.__r_idx_map = None
        self.__r_tag_map_in = None
        self.__r_tag_map_out = None

    @property
    def shape(self) -> Tuple[int]:
        """shape of the network indicated"""
        return (self.n_tag_in, self.n_neuron, self.n_tag_out)

    @property
    def n_tag_in(self) -> int:
        """number of input tags"""
        return len(self.get_tag_map_in())

    @property
    def n_tag_out(self) -> int:
        """number of output tags"""
        return len(self.get_tag_map_out())

    @property
    def n_neuron(self) -> int:
        """number of neurons"""
        return len(self.get_idx_map())

    def synapse_connections(
        self,
    ) -> List[Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]]]:
        """
        synapse_connections creates a list of all possible connections between neurons indicated by the CAM content stored in synapse objects

        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :type idx_map: Dict[int, NeuronKey]
        :return: a list of all possible connections indicated by synapse object content
        :rtype: List[Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]]]
        """
        __conn = []
        idx_map = self.get_idx_map()
        r_idx_map = self.get_r_idx_map()

        for nkey, syn_list in self.synapses.items():
            for syn in syn_list:
                # Any indexed neuron can send spikes to the post-synaptic neuron
                __conn += [(pre_idx, r_idx_map[nkey], syn) for pre_idx in idx_map]

        return __conn

    def destination_connections(
        self,
    ) -> List[Tuple[int, int, Union[Dynapse1Destination, Dynapse2Destination]]]:
        """
        destination_connections creates a list of all possible connections between neurons indicated by the SRAM content stored in the destination object.

        :param destinations: a dictionary of an active destinations
        :type destinations: Dict[NeuronKey, List[Dynapse2Destination]]
        :return: a list of all possible connections indicated by destination object content
        :rtype: List[Tuple[int, int, Dynapse2Destination]]
        """
        core_map = self.get_core_map()
        r_idx_map = self.get_r_idx_map()

        candidates = []

        for (h, c, n), dest_list in self.destinations.items():
            for dest in dest_list:

                # Get target cores from the source chip and the destination object
                for th, tc in self.route(dest, h):

                    # If the core indicated has some active neurons
                    if (th, tc) in core_map:
                        neurons = core_map[th, tc]

                        # Extend candidate list
                        candidates += [
                            (
                                r_idx_map[h, c, n],
                                r_idx_map[th, tc, tn],
                                dest,
                            )
                            for tn in neurons
                        ]

        return candidates

    def input_connections(self) -> List[Tuple[int, int, int, int]]:
        """
        input_connections finds the input connections inferred by the CAM content.
        It includes both the FPGA to neuron connections and neuron to neuron connections

        :return: the input to neuron conneciton list [(tag_idx, neuron_idx, gate_idx, weight_mask)]
        :rtype: List[Tuple[int, int, int, int]]
        """
        r_idx_map = self.get_r_idx_map()
        r_tag_map_in = self.get_r_tag_map_in()

        __connections = [
            self.read_syn_connection(
                (r_tag_map_in[self.cam_tag(syn)], r_idx_map[nkey], syn)
            )
            for nkey, _list in self.synapses.items()
            for syn in _list
        ]
        return __connections

    def intersect_connections(
        self,
        syn_connections: List[Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]]],
        dest_connections: List[
            Tuple[int, int, Union[Dynapse1Destination, Dynapse2Destination]]
        ],
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

        dest_candidates = set(map(self.match_dest_connection, dest_connections))

        __intersection = [
            self.read_syn_connection(connection)
            for connection in syn_connections
            if self.match_syn_connection(connection) in dest_candidates
        ]
        return __intersection

    def match_syn_connection(
        self, syn_connection: Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]]
    ) -> Tuple[int, int, int]:
        """
        match_syn_connection processes the synapse connection candidates and creates a comparable identity

        :param syn_connection: A connection candidate that is obtained from a synapse object
        :type syn_connection: Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]]
        :return: a standard from connection candidate which can be compared against any other connection candidate (pre, post, tag)
        :rtype: Tuple[int, int, int]
        """
        pre_idx, post_idx, syn = syn_connection
        return pre_idx, post_idx, self.cam_tag(syn)

    ### --- Getters --- ###

    def get_idx_map(self) -> Dict[int, NeuronKey]:
        """
        get_idx_map obtains an index map from active synapse and destinations dictionaries

        :return: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :rtype: Dict[int, NeuronKey]
        """
        if self.__idx_map is not None:
            return self.__idx_map

        syn_keys = list(self.synapses.keys()) if self.synapses is not None else []
        dest_keys = (
            list(self.destinations.keys()) if self.destinations is not None else []
        )
        keys = sorted(list(set(syn_keys + dest_keys)))
        self.__idx_map = dict(zip(range(len(keys)), keys))
        return self.__idx_map

    def get_core_map(self) -> Dict[CoreKey, List[np.uint8]]:
        """
        get_core_map converts an index map to a dictionary that presents the active cores and their active neruons.

            core_key : List[neurons]
            (0,0) : [1,5,7,2]
            (0,1) : [0,18,201]

        :return: a dictionary of the mapping between active cores and list of active neurons
        :rtype: Dict[CoreKey, List[np.uint8]]
        """
        if self.__core_map is not None:
            return self.__core_map

        # Initiate
        idx_map = self.get_idx_map()
        self.core_map = self.core_map_from_idx_map(idx_map)
        return self.core_map

    def get_tag_map_in(self) -> Dict[int, int]:
        """
        get_tag_map_in obtains a tag map from active synapses searching for all the tags given to any input connection

        :return: a dictionary of the mapping between matrix indexes of all the incoming events and their tags
        :rtype: Dict[int, int]
        """
        if self.__tag_map_in is not None:
            return self.__tag_map_in

        tags = [self.cam_tag(syn) for _list in self.synapses.values() for syn in _list]
        tags = sorted(list(set(tags)))
        self.__tag_map_in = dict(zip(range(len(tags)), tags))
        return self.__tag_map_in

    def get_tag_map_out(self) -> Dict[int, int]:
        """
        get_tag_map_out obtains a tag map from active destinations searching for all the tags given to any output connection

        :return: a dictionary of the mapping between matrix indexes of all the outcoming events and their tags
        :rtype: Dict[int, int]
        """
        if self.__tag_map_out is not None:
            return self.__tag_map_out

        tags = [
            self.sram_tag(dest, n)
            for (h, c, n), _list in self.destinations.items()
            for dest in _list
        ]
        tags = sorted(list(set(tags)))
        self.__tag_map_out = dict(zip(range(len(tags)), tags))
        return self.__tag_map_out

    def get_r_idx_map(self) -> Dict[NeuronKey, int]:
        """
        get_r_idx_map inverts index map. See also `Connect.get_idx_map()`

        :return: inverted index map
        :rtype: Dict[NeuronKey, int]
        """
        if self.__r_idx_map is not None:
            return self.__r_idx_map
        self.__r_idx_map = self.invert_dict(self.get_idx_map())
        return self.__r_idx_map

    def get_r_tag_map_in(self) -> Dict[int, int]:
        """
        get_r_tag_map_in inverts input tag map. See also `Connect.get_tag_map_in()`

        :return: inverted tag map
        :rtype: Dict[int, int]
        """
        if self.__r_tag_map_in is not None:
            return self.__r_tag_map_in
        self.__r_tag_map_in = self.invert_dict(self.get_tag_map_in())
        return self.__r_tag_map_in

    def get_r_tag_map_out(self) -> Dict[int, int]:
        """
        get_r_tag_map_out inverts output tag map. See also `Connect.get_tag_map_out()`

        :return: inverted tag map
        :rtype: Dict[int, int]
        """
        if self.__r_tag_map_out is not None:
            return self.__r_tag_map_out
        self.__r_tag_map_out = self.invert_dict(self.get_tag_map_out())
        return self.__r_tag_map_out

    def get_w_in_mask(self) -> np.ndarray:
        """
        get_w_in_mask creates the input weight mask using input connecitons

        :return: a matrix of encoded bit masks representing bitselect values to select and dot product the base Iw currents (pre, post, gate), for input connections
        :rtype: np.ndarray
        """
        w_in_mask = self.fill_weight_mask(
            self.input_connections(), n_pre=self.n_tag_in, n_post=self.n_neuron
        )
        return w_in_mask

    def get_w_out_mask(self) -> np.ndarray:
        """
        get_w_out_mask creates the output weight mask using output connecitons

        :return: A boolean output mask revealing the relation between neurons and their sram tags, (pre, tag_out) for output cconnections
        :rtype: np.ndarray
        """
        r_idx_map = self.get_r_idx_map()
        r_tag_map_out = self.get_r_tag_map_out()
        w_out_mask = np.zeros((self.n_neuron, self.n_tag_out))

        for (h, c, n), _list in self.destinations.items():
            for dest in _list:
                w_out_mask[
                    r_idx_map[(h, c, n)], r_tag_map_out[self.sram_tag(dest, n)]
                ] = 1

        return w_out_mask

    def get_w_rec_mask(self) -> np.ndarray:
        """
        get_w_rec_mask creates the recurrent weight matrix using the intersection of destination connections and input connections

        :return: A matrix of encoded bit masks representing bitselect values to select and dot product the base Iw currents (pre, post, gate), for recurrent connections
        :rtype: np.ndarray
        """
        syn_connections = self.synapse_connections()
        dest_connections = self.destination_connections()

        # Connect candidates
        recurrent_connections = self.intersect_connections(
            syn_connections, dest_connections
        )

        w_rec_mask = self.fill_weight_mask(
            recurrent_connections, n_pre=self.n_neuron, n_post=self.n_neuron
        )
        return w_rec_mask

    ### --- Utilities --- ###

    @staticmethod
    def core_map_from_idx_map(
        idx_map: Dict[int, NeuronKey]
    ) -> Dict[CoreKey, List[np.uint8]]:
        """
        core_map_from_idx_map is a utility function used to get a core map from index map

        :param idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :type idx_map: Dict[int, NeuronKey]
        :return: a dictionary of the mapping between active cores and list of active neurons
        :rtype: Dict[CoreKey, List[np.uint8]]
        """

        __core_map: Dict[CoreKey, List[np.uint8]] = {}

        for h, c, n in idx_map.values():
            if (h, c) not in __core_map:
                __core_map[h, c] = [n]
            else:
                __core_map[h, c].append(n)

        return __core_map

    @staticmethod
    def fill_weight_mask(
        connections: List[Tuple[int, int, int, int]],
        n_pre: int,
        n_post: int,
        n_gate: int = 4,
    ) -> np.ndarray:
        """
        fill_weight_mask creates and fills a weight mask object using a intersected connection list

        :param connections: the neuron to neuron connection list [(pre_idx, post_idx, gate_idx, weight_mask)]
        :type connections: List[Tuple[int, int, int, int]]
        :param n_neuron: number of neurons (1st and 2nd dimensions)
        :type n_neuron: int
        :param n_gate: then number of gates (3rd dimension), defaults to 4
        :type n_gate: int, optional
        :return: A matrix of encoded bit masks representing bitselect values to select and dot product the base Iw currents (pre, post, gate)
        :rtype: np.ndarray
        """
        weight_mask = np.zeros((n_pre, n_post, n_gate), dtype=int)

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

    @staticmethod
    def invert_dict(__dict: Dict[Any]) -> Dict[Any]:
        """
        invert_dict inverts any dictioary, keys become values and the values become keys

        :param __dict: any dictionary
        :type __dict: Dict[Any]
        :return: inverted dictionary
        :rtype: Dict[Any]
        """
        return {v: k for k, v in __dict.items()}

    ### --- Device Specific Implementation Required --- ###

    @staticmethod
    @abstractmethod
    def cam_tag(syn: Union[Dynapse1Synapse, Dynapse2Synapse], *args, **kwargs) -> int:
        """
        cam_tag obtains the cam tag from the samna synapse objects

        :param syn: a synapse object
        :type syn: Union[Dynapse1Synapse, Dynapse2Synapse]
        :return: the CAM tag indicated in the synapse object
        :rtype: int
        """
        pass

    @staticmethod
    def sram_tag(
        dest: Union[Dynapse1Destination, Dynapse2Destination], *args, **kwargs
    ) -> int:
        """
        sram_tag obtains the sram tag from the samna synapse destination object

        :param dest: a destination object
        :type dest: Union[Dynapse1Destination, Dynapse2Destination]
        :return: the SRAM tag indicated in the destination object
        :rtype: int
        """
        pass

    @abstractmethod
    def route(
        self,
        dest: Union[Dynapse1Destination, Dynapse2Destination],
        *args,
        **kwargs,
    ) -> List[CoreKey]:
        """
        route lists the cores that the destination object provided aims to convey the events.

        :param dest: DynapSE1 destination object containing SRAM content
        :type dest: Union[Dynapse1Destination, Dynapse2Destination]
        :return: list of cores targeted
        :rtype: List[CoreKey]
        """
        pass

    @abstractmethod
    def match_dest_connection(
        self,
        dest_connection: Tuple[
            int, int, Union[Dynapse1Destination, Dynapse2Destination]
        ],
        *args,
        **kwargs,
    ) -> Tuple[int, int, int]:
        """
        match_dest_connection processes the destination connection candidates and creates a comparable identity

        :param dest_connection: a connection candidate that is obtained from a destination object
        :type dest_connection: Tuple[ int, int, Union[Dynapse1Destination, Dynapse2Destination] ]
        :return: a standard form connection candidate which can be compared against any other connection candidate (pre, post, tag)
        :rtype: Tuple[int, int, int]
        """
        pass

    @abstractmethod
    def read_syn_connection(
        self,
        syn_connection: Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]],
        *args,
        **kwargs,
    ) -> Tuple[int, int, int, int]:
        """
        read_syn_connection processes the synapse connection identities and returns a version-invariant connection list identity

        :param syn_connection: A connection candidate that is obtained from a synapse object
        :type syn_connection: Tuple[int, int, Union[Dynapse1Synapse, Dynapse2Synapse]]
        :return: a version invariant connection list identity
        :rtype: Tuple[int, int, int, int]
        """
        pass


@dataclass
class ConnectorSE1(Connector):

    syn_weight_map = {
        0: 0b0001,  # GABA
        1: 0b0010,  # SHUNT
        2: 0b0100,  # NMDA
        3: 0b1000,  # AMPA
    }

    syn_map = {
        0: 1,  # GABA
        1: 3,  # SHUNT
        2: 2,  # NMDA
        3: 0,  # AMPA
    }

    @staticmethod
    def __tagger(core_id: np.uint8, neuron_id: np.uint8) -> int:
        """
        __tagger calculates the tag using the core id and neuron id
        """
        return neuron_id + (core_id * NUM_CORES)

    ### --- Abstract methods --- ###

    @staticmethod
    def cam_tag(syn: Dynapse1Synapse, *args, **kwargs) -> int:
        return ConnectorSE1.__tagger(syn.listen_core_id, syn.listen_neuron_id)

    @staticmethod
    def sram_tag(dest: Dynapse1Destination, neuron_id: int, *args, **kwargs) -> int:
        return ConnectorSE1.__tagger(dest.virtual_core_id, neuron_id)

    def route(self, dest: Dynapse1Destination, *args, **kwargs) -> List[CoreKey]:
        core = self.int_to_binary(dest.core_mask)
        dest_cores = np.argwhere(core).flatten()
        target_list = [(dest.target_chip_id, dest_core) for dest_core in dest_cores]
        return target_list

    def match_dest_connection(
        self, dest_connection: Tuple[int, int, Dynapse1Destination], *args, **kwargs
    ) -> Tuple[int, int, int]:
        idx_map = self.get_idx_map()
        pre_idx, post_idx, dest = dest_connection
        pre_chip_idx, pre_core_idx, pre_neuron_idx = idx_map[pre_idx]
        return pre_idx, post_idx, self.sram_tag(dest, pre_neuron_idx)

    def read_syn_connection(
        self, syn_connection: Tuple[int, int, Dynapse1Synapse], *args, **kwargs
    ) -> Tuple[int, int, int, int]:
        pre_idx, post_idx, syn = syn_connection
        gate = self.syn_map[syn.syn_type.value]
        weight = self.syn_weight_map[syn.syn_type.value]
        return pre_idx, post_idx, gate, weight


@dataclass
class ConnectorSE2(Connector):
    pos_map = {0: (1, 0)}
    syn_map = {
        0: 0,  # AMPA [] TODO !!!!! DO NOT DO IT !!!!!!
        1024: 0,  # AMPA
        512: 1,  # GABA
        256: 2,  # NMDA
        128: 3,  # SHUNT
    }

    def __post_init__(self) -> None:
        self.__r_pos_map = {v: k for k, v in self.pos_map.items()}
        super().__post_init__()

    def route_AER(self, dest: Dynapse2Destination, src_chip: np.uint8) -> int:
        """
        route_AER finds the destination chip id using the destination object and the position map

        :param src_chip: the departure chip ID
        :type src_chip: np.uint8
        :param dest: Dynap-SE2 destination object containing SRAM content
        :type dest: Dynapse2Destination
        :return: the arrival chip ID
        :rtype: int
        """
        x_src, y_src = self.pos_map[src_chip]
        x_dest = x_src + dest.x_hop
        y_dest = y_src + dest.y_hop

        if (x_dest, y_dest) in self.__r_pos_map:
            chip_id = self.__r_pos_map[x_dest, y_dest]
            return chip_id
        else:
            return -1

    ### --- Abstract methods --- ###

    @staticmethod
    def cam_tag(syn: Dynapse2Synapse, *args, **kwargs) -> int:
        return syn.tag

    @staticmethod
    def sram_tag(dest: Dynapse2Destination, *args, **kwargs) -> int:
        return ConnectorSE2.cam_tag(dest)

    def route(
        self, dest: Dynapse2Destination, src_chip: np.uint8, *args, **kwargs
    ) -> List[CoreKey]:
        dest_chip = self.route_AER(dest, src_chip)
        dest_cores = np.argwhere(dest.core).flatten()
        target_list = [(dest_chip, dest_core) for dest_core in dest_cores]
        return target_list

    def match_dest_connection(
        self, dest_connection: Tuple[int, int, Dynapse2Destination], *args, **kwargs
    ) -> Tuple[int, int, int]:
        return self.match_syn_connection(dest_connection)

    def read_syn_connection(
        self, syn_connection: Tuple[int, int, Dynapse2Synapse]
    ) -> Tuple[int, int, int, int]:
        pre_idx, post_idx, syn = syn_connection
        gate = self.syn_map[syn.dendrite.value]
        weight = self.binary_to_int(syn.weight)
        return pre_idx, post_idx, gate, weight
