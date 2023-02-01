"""
Dynap-SE2 distributed memory reading, serves as a backend tool

* Non User Facing * 
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import bisect

from rockpool.devices.dynapse.samna_alias import (
    Dynapse2Configuration,
    Dynapse2Synapse,
    Dynapse2Destination,
)
from rockpool.devices.dynapse.quantization import WeightHandler
from rockpool.devices.dynapse.lookup import CHIP_POS

from rockpool.devices.dynapse.hardware.config.allocator import WeightAllocator


@dataclass
class MemorySE2:
    """
    MemorySE2 traces the connections between neurons indicated by the SRAM and CAM content
    """

    chip_pos: Dict[int, Tuple[int]] = field(default_factory=lambda: CHIP_POS)
    """global chip position dictionary (chip_id : (xpos,ypos)), defaults to CHIP_POS"""

    def spec_from_config(self, config: Dynapse2Configuration) -> Dict[str, list]:
        """
        spec_from_config creates a specification object which contains input and recurrent weights, dendrites and core map from the samna configuration object

        :param config: a samna configuration object used to configure all the system level properties
        :type config: Dynapse2Configuration
        :return: a dictionary containing weights and dendrites between connections, enough to restore weight matrices for simulation
        :rtype: Dict[str, list]
        """
        spec = self.read_content_from_config(config)
        spec = self.get_weights(**spec)
        return spec

    def read_content_from_config(
        self, config: Dynapse2Configuration
    ) -> Dict[str, list]:
        """
        read_content_from_config collects distributed SRAM and CAM entries

        :param config: a samna configuration object used to configure all the system level properties
        :type config: Dynapse2Configuration
        :return:
            :synapses: list of synapses read from neuron CAMs, contains weights and dendrites
            :syn_addresses: the list of physical addresses (chip, core, neuron) of ``synapses``
            :destinations: list of destinations read from neuron SRAMs, contains relative destination chip position
            :dest_addresses: the list of physical addresses (chip, core, neruon) of ``destinations``
        :rtype: Dict[str, list]
        """
        synapses = []
        syn_addresses = []
        destinations = []
        dest_addresses = []

        # Traverse the chip for active neruon-neuron connections
        for h, chip in enumerate(config.chips):  # 1-4
            for c, core in enumerate(chip.cores):  # 4
                for n, neuron in enumerate(core.neurons):  # 256
                    # FAN-IN (up-to 64) CAM
                    for syn in neuron.synapses:
                        if sum(syn.weight) > 0 or syn.tag > 0:
                            syn_addresses.append([h, c, n])
                            synapses.append(Dynapse2Synapse.from_samna(syn))

                    # FAN-OUT (up-to 4) SRAM
                    for dest in neuron.destinations:
                        if sum(dest.core) > 0:
                            dest_addresses.append([h, c, n])
                            destinations.append(Dynapse2Destination.from_samna(dest))

        return {
            "synapses": synapses,
            "syn_addresses": syn_addresses,
            "destinations": destinations,
            "dest_addresses": dest_addresses,
        }

    def get_weights(
        self,
        synapses: List[Dynapse2Synapse],
        syn_addresses: List[Tuple[int]],
        destinations: List[Dynapse2Destination],
        dest_addresses: List[Tuple[int]],
    ) -> Dict[str, list]:
        """
        get_weights is the function which processes synapse and dendrite lists obtained from the samna configuration object and produce weight and dendrite matrices

        :param synapses: list of synapses read from neuron CAMs, contains weights and dendrites
        :type synapses: List[Dynapse2Synapse]
        :param syn_addresses: the list of physical addresses (chip, core, neuron) of ``synapses``
        :type syn_addresses: List[Tuple[int]]
        :param destinations: list of destinations read from neuron SRAMs, contains relative destination chip position
        :type destinations: List[Dynapse2Destination]
        :param dest_addresses: the list of physical addresses (chip, core, neruon) of ``destinations``
        :type dest_addresses: List[Tuple[int]]
        :return: w_in, dendrite_in, w_rec, dendrite_rec, core_map
            :w_in: input (virtual -> hardware) weight matrix (integer masks)
            :dendrite_in: input (virtual -> hardware) connection gates (AMPA, GABA, NMDA, SHUNT)
            :w_rec: recurrent (hardware -> hardware) weight matrix
            :dendrite_rec: recurrent (hardware -> hardware) connection gates
            :core_map: the mapping between neuron index to respective core ID (chip, core)
        :rtype: Dict[str, list]
        """

        # Lists to fill
        input_synapses = []
        input_locations = []
        recurrent_synapses = []
        recurrent_locations = []

        # Sort destinations according to tags
        didx = list(range(len(destinations)))
        didx.sort(key=lambda i: destinations[i].tag)
        destinations = list(map(destinations.__getitem__, didx))
        dest_addresses = list(map(dest_addresses.__getitem__, didx))

        # Get detination tags
        dest_tag_list = list(map(lambda d: d.tag, destinations))
        assert dest_tag_list == sorted(dest_tag_list)

        ## Helper functions
        def tag_in_dest(tag: int) -> int:
            """
            tag_in_dest applies binary search on a sorted destination tag list

            :param tag: tag of interest
            :type tag: int
            :return: -1 if it does not exist in the list, else the index if of the tag
            :rtype: int
            """
            if not dest_tag_list:
                return -1
            idx = bisect.bisect_left(dest_tag_list, tag)
            return idx if dest_tag_list[idx] == tag else -1

        def pos_match(didx: int, sidx: int) -> bool:
            """
            pos_match checks if the manhattan distance between the source chip and the destination chip and the
            number of cartesian chip hops (x_hop, y_hop) indicated in the destination object matches

            :param didx: destination list index
            :type didx: int
            :param sidx: _description_
            :type sidx: synapse list index
            :return: true if matches, else false
            :rtype: bool
            """
            __dest = destinations[didx]
            x_hop, y_hop = WeightAllocator.manhattan(
                dest_addresses[didx][0], syn_addresses[sidx][0], self.chip_pos
            )
            return (x_hop == __dest.x_hop) and (y_hop == __dest.y_hop)

        def core_match(didx: int, sidx: int) -> bool:
            """
            core_match checks if the core mask stored in the destination object matches the
            destination core position of the synapse candidate found

            :param didx: destination list index
            :type didx: int
            :param sidx: _description_
            :type sidx: synapse list index
            :return: true if matches, else false
            :rtype: bool
            """
            relative_core_id = syn_addresses[sidx][1]
            core_mask = destinations[didx].core
            return core_mask[relative_core_id]

        # Split recurrent and virtual synapses
        for sidx, syn in enumerate(synapses):
            didx = tag_in_dest(syn.tag)
            if didx >= 0:
                if pos_match(didx, sidx) and core_match(didx, sidx):
                    recurrent_synapses.append(syn)
                    recurrent_locations.append(self.address_hash(syn_addresses[sidx]))

                else:
                    input_synapses.append(syn)
                    input_locations.append(self.address_hash(syn_addresses[sidx]))
            else:
                input_synapses.append(syn)
                input_locations.append(self.address_hash(syn_addresses[sidx]))

        # Deduce network shape
        post_idx_map = self.get_post_idx_map(input_locations, recurrent_locations)
        n_in = len(set(map(lambda syn: syn.tag, input_synapses)))
        n_rec = len(post_idx_map)

        # Compose weight matrices
        weights_in, dendrites_in = self.fill_weight_matrix(
            (n_in, n_rec), input_synapses, input_locations, post_idx_map
        )
        weights_rec, dendrites_rec = self.fill_weight_matrix(
            (n_rec, n_rec), recurrent_synapses, recurrent_locations, post_idx_map
        )

        return {
            "weights_in": weights_in,
            "dendrites_in": dendrites_in,
            "weights_rec": weights_rec,
            "dendrites_rec": dendrites_rec,
            "core_map": self.get_core_map(post_idx_map),
        }

    def get_post_idx_map(
        self, input_locations: List[int], recurrent_locations: List[int]
    ) -> Dict[int, int]:
        """
        get_post_idx_map obtains a post-synaptic neuron index map.
        mapping : encoded neuron address -> matrix column index (axis=1)

        :param input_locations: encoded hardware locations of the neurons receiving virtual connections (see ``class.encode()``)
        :type input_locations: List[int]
        :param recurrent_locations: encoded hardware locations of the neurons receiving recurrent connections (see ``class.encode()``)
        :type recurrent_locations: List[int]
        :return: an index map which maps the encoded locations to post-synaptic dimension indices
        :rtype: Dict[int, int]
        """

        unique_input_loc = set(input_locations)
        unique_rec_loc = set(recurrent_locations)

        # n_rec equals to the maximum of unique input locations and unique recurrent locations
        post_idx_map = dict(zip(unique_input_loc, range(len(unique_input_loc))))
        post_idx_map.update(dict(zip(unique_rec_loc, range(len(unique_rec_loc)))))

        return post_idx_map

    def get_core_map(self, post_idx_map: Dict[int, int]) -> List[Tuple[int]]:
        """
        get_core_map converts the encoded post-synaptic neuron index map to a decoded core map.
        The core map restores only the chip and core dimensions from the encoded location, because the neuron address is irrelevant
        mapping : matrix column index -> core address

        :param post_idx_map: an index map which maps the encoded locations to post-synaptic dimension indices
        :type post_idx_map: Dict[int, int]
        :return: an index to core address mapping
        :rtype: List[Tuple[int]]
        """
        return [self.decode_hash(loc)[0:2] for loc in post_idx_map]

    def fill_weight_matrix(
        self,
        shape: Tuple[int],
        synapses: List[Dynapse2Synapse],
        locations: List[int],
        post_idx_map: Dict[int, int],
    ) -> Tuple[np.ndarray]:
        """
        fill_weight_matrix places the weights and dendrites stored in the synapses stored in CAMs to dense weight matrices

        :param shape: the desired shape of the resulting weigth matrix
        :type shape: Tuple[int]
        :param synapses: list of synapses to be placed
        :type synapses: List[Dynapse2Synapse]
        :param locations: list of encoded hardware locations of the post-synaptic neurons of the synapes
        :type locations: List[int]
        :param post_idx_map: an index map which maps the encoded locations to post-synaptic dimension indices (hardware location -> matrix column)
        :type post_idx_map: Dict[int, int]
        :return: weights, dendrites
            :weights: 4-bit weights matrix
            :dendrites: the synaptic gate matrix allowed entries : (AMPA, GABA, NMDA, SHUNT, NONE)
        :rtype: Tuple[np.ndarray]
        """

        weights = np.zeros(shape, dtype=np.int32)
        dendrites = np.zeros(shape, dtype=np.int32)

        # pre-synaptic dimension is depended on tags
        tag_list = list(set(map(lambda syn: syn.tag, synapses)))
        pre_idx_map = dict(zip(tag_list, range(len(tag_list))))

        # Helper functions
        def __pre(syn: Dynapse2Synapse) -> int:
            """__pre returns pre-synaptic index (axis=0)"""
            return pre_idx_map[syn.tag]

        def __post(loc: int) -> int:
            """__post returns post-synaptic index (axis=1)"""
            return post_idx_map[loc]

        # Place synaptic weights and dendrites on dense weight matrices
        for loc, syn in zip(locations, synapses):
            weights[__pre(syn)][__post(loc)] = int(
                WeightHandler.bit2int_mask(4, np.array(syn.weight))
            )

            dendrites[__pre(syn)][__post(loc)] = syn.dendrite.value

        return weights, dendrites

    @staticmethod
    def address_hash(addr_tuple: Tuple[int], n_bit: int = 8) -> int:
        """
        address_hash implements and encoding method for neuron addresses (chip, core, neuron)
        It makes it possible to use neuron locations as the dictionary keys

        :param addr_tuple: the physical address of a neuron (chip, core, neuron)
        :type addr_tuple: Tuple[int]
        :param n_bit: number of bits allocated for each address component, defaults to 8
        :type n_bit: int, optional
        :return: a unique code encoding the physical location
        :rtype: int
        """
        hash = 0
        for addr in addr_tuple:
            hash = (hash << n_bit) | addr
        return hash

    @staticmethod
    def decode_hash(hash: int, n_bit: int = 8, tuple_length: int = 3) -> int:
        """
        decode_hash restores the full physical address of a neuron from the encoded version

        :param hash: the unique code encoding the physical location
        :type hash: int
        :param n_bit:  number of bits allocated for each address component, defaults to 8
        :type n_bit: int, optional
        :param tuple_length: the number of address components, defaults to 3
        :type tuple_length: int, optional
        :return: the restored 3 component address for a neuron (chip, core, neuron)
        :rtype: int
        """
        mask = 2**n_bit - 1
        restored = []

        for _ in range(tuple_length):
            restored.append(hash & mask)
            hash = hash >> n_bit

        restored.reverse()

        return restored
