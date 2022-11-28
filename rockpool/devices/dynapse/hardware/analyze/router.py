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

from typing import Callable, Tuple, Dict, Union, List

import numpy as np

from rockpool.devices.dynapse.typehints import CoreKey, NeuronKey

from rockpool.devices.dynapse.samna_alias import (
    Dynapse1Synapse,
    Dynapse1Destination,
    Dynapse1Configuration,
    Dynapse2Synapse,
    Dynapse2Destination,
    Dynapse2Configuration,
)

from .memory import Connector, ConnectorSE1, ConnectorSE2, SynDict, DestDict

__all__ = ["Router"]


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
