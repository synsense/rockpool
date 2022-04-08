"""
Dynap-SE router simulator. Create a CAM matrix using SRAM and CAM content

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
13/09/2021
"""

from typing import Optional, Tuple, Any, Iterable, Dict, Union, List

from rockpool.devices.dynapse.base import (
    DynapSE,
    ArrayLike,
    Numeric,
    NeuronKey,
    NeuronConnection,
    NeuronConnectionSynType,
)

import numpy as np

from rockpool.devices.dynapse.samna_alias.dynapse1 import (
    Dynapse1SynType,
    Dynapse1Neuron,
    Dynapse1Synapse,
    Dynapse1Destination,
    Dynapse1Configuration,
)


class Router(DynapSE):
    @staticmethod
    def get_UID(chipID: np.uint8, coreID: np.uint8, neuronID: np.uint16) -> np.uint16:
        """
        get_ID produce a globally unique id for a neuron on the board

        :param chipID: Unique chip ID
        :type chipID: np.uint8
        :param coreID: Non-unique core ID
        :type coreID: np.uint8
        :param neuronID: Non-unique neuron ID
        :type neuronID: np.uint16
        :raises ValueError: chipID out of bounds
        :raises ValueError: coreID out of bounds
        :raises ValueError: neuronID out of bounds
        :return: Globally unique neuron ID
        :rtype: np.uint16
        """
        if chipID >= Router.NUM_CHIPS or chipID < 0:
            raise ValueError(
                f"chipID out of bounds. 0 <= chipID:{chipID} < {Router.NUM_CHIPS}"
            )

        if coreID >= Router.NUM_CORES or coreID < 0:
            raise ValueError(
                f"coreID out of bounds. 0 <= coreID:{coreID} < {Router.NUM_CORES}"
            )

        if neuronID >= Router.NUM_NEURONS or coreID < 0:
            raise ValueError(
                f"neuronID out of bounds. 0 <= neuronID:{neuronID} < {Router.NUM_NEURONS}"
            )

        uid = (
            Router.NUM_NEURONS * Router.NUM_CORES * chipID
            + Router.NUM_NEURONS * coreID
            + neuronID
        )
        return np.uint16(uid)

    @staticmethod
    def decode_UID(UID: np.uint16) -> NeuronKey:
        """
        decode_UID decodes the uniquie neuron ID to neuron keys (chipID, coreID, neuronID)

        :param UID: Globally unique neuron ID
        :type UID: np.uint16
        :return: the neuron key composed of chipID, coreID and neuronID in order
        :rtype: NeuronKey
        """

        # CHIP
        chipID = UID // (Router.NUM_NEURONS * Router.NUM_CORES)

        # CORE
        UID -= Router.NUM_NEURONS * Router.NUM_CORES * chipID
        coreID = UID // Router.NUM_NEURONS

        # NEURON
        neuronID = UID - Router.NUM_NEURONS * coreID

        ID_tuple = (np.uint8(chipID), np.uint8(coreID), np.uint16(neuronID))
        return ID_tuple

    @staticmethod
    def get_UID_combination(
        chipID: Optional[Union[ArrayLike, Numeric]] = None,
        coreID: Optional[Union[ArrayLike, Numeric]] = None,
        neuronID: Optional[Union[ArrayLike, Numeric]] = None,
    ) -> List[np.uint16]:
        """
        get_UID_combination get_ID_range provides uniquie IDs obtained by all the combinations of
        given chip ID, core ID, and neuron IDs

        :param chipID: It can be provided as a fixed number, an array of different values to be investigated, or None. If None, all possible values are considered. defaults to None
        :type chipID: Optional[Union[ArrayLike, Numeric]], optional
        :param coreID: Handling is the same as the chipID, defaults to None
        :type coreID: Optional[Union[ArrayLike, Numeric]], optional
        :param neuronID: Handling is the same as the chipID, defaults to None
        :type neuronID: Optional[Union[ArrayLike, Numeric]], optional
        :return: List of unique neuron IDs
        :rtype: List[np.uint16]
        """

        def ID_list(
            ID: Union[ArrayLike, Numeric], ID_max: Union[np.uint8, np.uint16]
        ) -> Iterable:
            """
            ID_list Check if given ID type and the values are in the proper range and provides a iterable for traversing.

            :param ID: A fixed number, an array of different values to be investigated, or None. If None, all possible values are considered
            :type ID: Union[ArrayLike, Numeric]
            :param ID_max: The maximum number that the ID can take
            :type ID_max: Union[np.uint8, np.uint16]
            :raises TypeError: Given ID type is not compatible with Optional[Union[ArrayLike, Numeric]]
            :raises ValueError: In the given ID or ID list, the maximum value is greater than the upper limit
            :raises ValueError: In the given ID or ID list, the minimum value is less than 0
            :return: an array like object or a range generator to traverse a list of IDs.
            :rtype: Iterable
            """
            if ID is None:  # Full range
                ID = range(ID_max)

            elif not isinstance(
                ID, (tuple, list, np.ndarray)
            ):  # The ID is a fixed number. Still, it should be iterable.
                try:
                    dtype = type(ID_max)
                    ID = [dtype(ID)]
                except:
                    raise TypeError(
                        "ID can only be an arraylike : [np.ndarray, List, Tuple] or a number [int, float, complex, np.number] "
                    )

            else:
                if np.max(ID) > ID_max:
                    raise ValueError(
                        f"An ID[{np.argmax(ID)}] = {np.max(ID)} should not be greater than {ID_max}"
                    )

                if np.min(ID) < 0:
                    raise ValueError(
                        f"An ID[{np.argmin(ID)}] = {np.min(ID)} should not be less than 0!"
                    )
                ID = np.sort(ID)

            return ID

        chipID = ID_list(chipID, Router.NUM_CHIPS)
        coreID = ID_list(coreID, Router.NUM_CORES)
        neuronID = ID_list(neuronID, Router.NUM_NEURONS)

        id_list = []

        # Traverse the given chip, core and neuron ID iterables
        for chip in chipID:
            for core in coreID:
                for neuron in neuronID:
                    id_list.append(Router.get_UID(chip, core, neuron))

        return id_list

    @staticmethod
    def bitmask_select(bitmask: np.uint8) -> np.ndarray:
        """
        bitmask_select apply 8-bit mask to select bits (or coreIDs in coremask case)
        00000001 -> selected bit: 0
        00001000 -> selected bit: 3
        00000101 -> selected bit 0 and 2

        :param bitmask: Binary mask to select core IDs
        :type bitmask: np.uint8
        :return: an array of indices of selected bits
        :rtype: np.ndarray
        """

        if bitmask > 255 or bitmask < 0:
            raise IndexError(
                "Given bit mask is out of range! 8-bit mask are accepted (at max 255 as integer value)"
            )

        bits = range(8)  # [0,1,2,3,4,5,6,7]
        bit_pattern = lambda n: (1 << n)  # 2^n

        # Indexes of the IDs to be selected in bits list
        idx = np.array([bitmask & bit_pattern(bit) for bit in bits], dtype=bool)
        idx_selected = np.array(bits, dtype=np.uint8)[idx]
        return idx_selected

    @staticmethod
    def pair_uid_to_key(
        connections: Union[NeuronConnection, List[NeuronConnection]]
    ) -> Union[NeuronKey, List[NeuronKey]]:
        """
        pair_uid_to_key convert a single neuron connection pair or a list of neuron connection pairs
        consisting of their uniquie IDs to a neuron key or a list of neuron keys consisting of
        a tuple of chipID, coreID and neuronID

        :param connections: a tuple of UIDs or a list of tuples of UIDs
        :type connections: Union[NeuronConnection, List[NeuronConnection]]
        :raises TypeError: When UID cannot be casted to np.uint16
        :raises TypeError: When connections are not a tuple of UIDs or a list of tuples of UIDs!
        :return: A neuron key with chipID, coreID and neuronID or a list of neuron keys.
        :rtype: Union[NeuronKey, List[NeuronKey]]
        """

        pair_decoder = lambda pair: tuple(map(Router.decode_UID, pair))

        if isinstance(connections, (tuple, list, np.ndarray)):

            # List of pairs
            if isinstance(connections[0], (tuple, list, np.ndarray)):
                connections = list(map(pair_decoder, connections))

            # Single pair
            else:
                try:
                    np.uint16(connections[0])
                    np.uint16(connections[1])
                except:
                    raise TypeError(f"UID should be casted to np.uint16")

                connections = pair_decoder(connections)

        else:
            raise TypeError(
                "Connections can be a tuple of UIDs or a list of tuples of UIDs!"
            )

        return connections

    @staticmethod
    def receiving_connections(
        neuron: Optional[Dynapse1Neuron] = None,
        synapse: Optional[Dynapse1Synapse] = None,
        neuron_UID: Optional[np.uint16] = None,
        listen_core_id: Optional[np.uint8] = None,
        listen_neuron_id: Optional[np.uint8] = None,
        syn_type: Optional[Union[Dynapse1SynType, np.uint8]] = None,
    ) -> List[NeuronConnectionSynType]:
        """
        receiving_connections produce a list of spike receiving connections given
        samna neuron and synapse objects or neuronID and it's event accepting conditions.
        From the device's point of view, in each CAM(Content Addressable Memory) cell
        the neuron can be set to listen to (i.e. receive events from) one other neuron with a specified synapse type.
        In the CAM, the core and neuron IDs can be set, but there is no space to set the chipID.
        Therefore, a post-synaptic neuron listens all the pre-synaptic neurons having the
        same core and neuron ID across different chips.

        Note that samna objects have priority over the ID definitions. For example, if both Dynapse1Neuron
        object and neuronUID is given, funciton considers only the Dynapse1Neuron object.

        :param neuron: The neuron at the post-synaptic side, defaults to None
        :type neuron: Optional[Dynapse1Neuron], optional
        :param synapse: High level content of a CAM cell, defaults to None
                        "syn_type": 2,
                        "listen_neuron_id": 0,
                        "listen_core_id": 0
        :type synapse: Optional[Dynapse1Synapse], optional
        :param neuron_UID: post-synaptic universal neuron ID, defaults to None
        :type neuron_UID: Optional[np.uint16], optional
        :param listen_core_id: the event sending core ID to listen, defaults to None
        :type listen_core_id: Optional[np.uint8], optional
        :param listen_neuron_id: the event sending neuron ID to listen, defaults to None
        :type listen_neuron_id: Optional[np.uint8], optional
        :param syn_type: the type of the synapse to process the events, defaults to None
        :type syn_type: Optional[Union[Dynapse1SynType, np.uint8]], optional
        :return: List of unique IDs of all neuron connection pairs in the (pre, post, syn_type) order.
        :rtype: List[NeuronConnectionSynType]
        """

        # Get the required info from the samna objects or explicit definitions
        chip_id, core_id, neuron_id = (
            Router.decode_UID(neuron_UID)
            if neuron is None
            else (neuron.chip_id, neuron.core_id, neuron.neuron_id)
        )

        listen_core_id = listen_core_id if synapse is None else synapse.listen_core_id
        listen_neuron_id = (
            listen_neuron_id if synapse is None else synapse.listen_neuron_id
        )
        syn_type = syn_type if synapse is None else synapse.syn_type

        # Pre-synaptic neurons to listen across 4 chips
        pre_list = Router.get_UID_combination(
            chipID=None,
            coreID=listen_core_id,
            neuronID=listen_neuron_id,
        )

        # Post-synaptic neuron
        post = Router.get_UID(chip_id, core_id, neuron_id)
        connections = Router.connect_pre_post(pre_list, post, syn_type)

        return connections

    @staticmethod
    def broadcasting_connections(
        neuron: Optional[Dynapse1Neuron] = None,
        destination: Optional[Dynapse1Destination] = None,
        neuron_UID: Optional[np.uint16] = None,
        target_chip_id: Optional[np.uint8] = None,
        core_mask: Optional[np.uint8] = None,
        virtual_core_id: Optional[np.uint8] = None,
    ) -> List[NeuronConnection]:
        """
        broadcasting_connections produce a list of spike boardcasting connections given a neuron and a destination object or
        given exact neuron_UID sending the events and it's target chip ID, core mask, and virtual core ID if exist.
        From device's point of view, in each SRAM(Static Random Access Memory) cell
        the neuron can be set to broadcast it's spikes to one other chip. In the SRAM, one can also
        set a core mask to narrow down the number of neruons receiving the spikes. However, there is no
        space to set the neuronID. Therefore, a pre-synaptic neuron broadcast it's spike output
        to all the neuron in the specified core. The neurons at the post-synaptic side decide on listening or not.

        Note that samna objects have priority over the ID definitions. For example, if both Dynapse1Neuron
        object and neuronUID is given, funciton considers only the Dynapse1Neuron object.

        :param neuron: The neuron at the pre-synaptic side, defaults to None
        :type neuron: Optional[Dynapse1Neuron], optional
        :param destination: High level content of the SRAM cell, defaults to None
                            "targetChipId": 0,
                            "inUse": false,
                            "virtualCoreId": 0,
                            "coreMask": 0,
                            "sx": 0,
                            "sy": 0,
                            "dx": 0,
                            "dy": 0
        :type destination: Optional[Dynapse1Destination], optional
        :param neuron_UID: pre-synaptic universal neuron ID, defaults to None
        :type neuron_UID: Optional[np.uint16], optional
        :param target_chip_id: the chip ID to broadcast the events, defaults to None
        :type target_chip_id: Optional[np.uint8], optional
        :param core_mask: the core mask used while sending the events, defaults to None
            1111 means all 4 cores are on the target
            0010 means events will be arrived at core 2 only
        :type core_mask: Optional[np.uint8], optional
        :param virtual_core_id: virtual core ID of the sending side to pretend. If None, neuron core ID is used instead, defaults to None
        :type virtual_core_id: Optional[np.uint8], optional
        :return: List of unique IDs of all neuron connection pairs in the (pre, post) order.
        :rtype: List[NeuronConnection]
        """

        # If there is no target core, there is no need to calculate the rest!
        core_mask = core_mask if destination is None else destination.core_mask
        cores_to_send = Router.bitmask_select(core_mask)
        if len(cores_to_send) == 0:
            return []

        # Get the required info from the samna objects or explicit definitions
        target_chip_id = (
            target_chip_id if destination is None else destination.target_chip_id
        )
        virtual_core_id = (
            virtual_core_id if destination is None else destination.virtual_core_id
        )
        chip_id, core_id, neuron_id = (
            Router.decode_UID(neuron_UID)
            if neuron is None
            else (neuron.chip_id, neuron.core_id, neuron.neuron_id)
        )

        # No need to define the virtual core id, it's equal to core id if not defined!
        virtual_core_id = core_id if virtual_core_id is None else virtual_core_id

        # Pre-synaptic neurons to broadcast spike events
        post_list = Router.get_UID_combination(
            chipID=target_chip_id,
            coreID=cores_to_send,
            neuronID=None,
        )

        # Pre-synaptic neuron
        pre = Router.get_UID(
            chip_id,
            virtual_core_id,
            neuron_id,
        )  # pretend

        connections = Router.connect_pre_post(pre, post_list)
        return connections

    @staticmethod
    def connect_pre_post(
        preUID: Union[np.uint16, ArrayLike],
        postUID: Union[np.uint16, ArrayLike],
        syn_type: Optional[Union[Dynapse1SynType, np.uint8]] = None,
    ) -> Union[List[NeuronConnection], List[NeuronConnectionSynType]]:
        """
        connect_pre_post produce a list of connections between neurons like List[(preUID, postUID)].
        The pre and post can be given as a list or a single ID. If a single ID is given, a repeating
        UID list having the same shape with the others is created.

        :param preUID: a unique pre-synaptic neuron ID or a list of IDs
        :type preUID: Union[np.uint16, ArrayLike]
        :param postUID: a unique post-synaptic neuron ID or a list of IDs
        :type postUID: Union[np.uint16, ArrayLike]
        :param syn_type: The synapse type of the connection, defaults to None
        :type syn_type: Optional[Union[Dynapse1SynType, np.uint8]], optional
        :raises ValueError: When the size of the preUID and postUID arrays are not the same
        :raises ValueError: When the size of syn_type is different than the number of connections
        :raises TypeError: preUID or postUID is not ArraLike or a type which can casted to np.uint16
        :return: connections between neruons in the form of tuple(preUID, postUID) or tuple(preUID, postUID, synType)
        :rtype: Union[List[NeuronConnection], List[NeuronConnectionSynType]]:
        """

        def to_list(uid: Numeric, dtype: type = np.uint16) -> List[Numeric]:
            """
            to_list creates a repeating list given a single element

            :param uid: a single unique neuron id
            :type uid: Numeric
            :raises TypeError: If the neuron id cannot be casted to dtype.
            :return: a repeating list of given uid with the shape of the second uid list provided to the upper level function.
            :rtype: List[Numeric]
            """
            try:
                uid = dtype(uid)
            except:
                raise TypeError(
                    f"data provided should be casted to data type: {dtype}!"
                )
            uid_list = [uid] * n_connections

            return uid_list

        # Find the number of connections defined implicitly by the size of one of the arrays provided
        n_connections = 1

        if isinstance(preUID, (tuple, list, np.ndarray)):
            n_connections = len(preUID)
            if n_connections == 1:
                preUID = preUID[0]

        if isinstance(postUID, (tuple, list, np.ndarray)):
            if n_connections == 1:
                n_connections = len(postUID)
                if n_connections == 1:
                    postUID = postUID[0]
            else:
                if n_connections != len(postUID):
                    raise ValueError(
                        f"number of pre-synaptic and post-synaptic neurons does not match {len(preUID)} != {len(postUID)}"
                    )

        # syn_type is defined
        if syn_type is not None:
            if isinstance(syn_type, (tuple, list, np.ndarray)):
                if n_connections != len(syn_type):
                    raise ValueError(
                        f"number of synapse types does not match with n_connections {len(syn_type)} != {n_connections}"
                    )
                else:
                    syn_type = list(map(np.uint8, syn_type))
            else:
                syn_type = to_list(syn_type, np.uint8)

        # If both preUID and postUID is numeric
        if n_connections == 1:
            preUID = to_list(preUID)
            postUID = to_list(postUID)

        else:
            if isinstance(preUID, (tuple, list, np.ndarray)):
                postUID = to_list(postUID)

            elif isinstance(postUID, (tuple, list, np.ndarray)):
                preUID = to_list(preUID)

            else:
                raise TypeError(
                    f"preUID and postUID should be an ArraLike or a type which can be casted to uint16"
                )
        if syn_type is not None:
            connections = list(zip(preUID, postUID, syn_type))
        else:
            connections = list(zip(preUID, postUID))

        return connections

    @staticmethod
    def connect_input(
        virtual_connections: List[NeuronConnection],
        virtual_neurons: Optional[List[np.uint16]] = None,
    ) -> Dict[np.uint16, Tuple[np.uint8, np.uint8]]:
        """
        connect_input create a dictionary of FPGA->DynapSE connections to be used in the creation of a list of
        spike boardcasting connections using `Router.broadcasting_connections()`. FPGA neuron are not actual neruons
        with state, instead they generate events given a timestep and necessary credientials. They just forward
        the spiking events to desired locations. Due to the architecture of the router, it's impossible to define a
        unique destination for an FPGA-generated event. Instead, one can define a target chip and a core mask to
        select one core or multiple cores to broadcast an event. Then on the destination side, neurons decide on listening or not.

        :param virtual_connections: a list of tuples of pre-synaptic neuron UID and post-synaptic neuron UID defining the connections from the FPGA neurons to device neurons.
        :type virtual_connections: List[NeuronConnection]
        :param virtual_neurons: explicitly defined virtual neurons, defaults to None
        :type virtual_neurons: Optional[List[np.uint16]], optional
        :raises ValueError: Virtual pre-synaptic neuron is not given in virtual_neurons (in the case virtual neruons defined)
        :raises ValueError: A virtual neuron cannot broadcast to more than one chips!
        :return: a dictionary of virtual(FPGA) pre-synaptic neuron UID mapping to it's destination (chipID, coreMask)
        :rtype: Dict[np.uint16, Tuple[np.uint8, np.uint8]]
        """

        # If a virtual neuron broadcast to multiple cores inside the chip, update the core_mask accordingly
        update_core_mask = lambda core_mask, core_ID: core_mask | (1 << core_ID)

        # If virtual neurons are given explicitly, any other neuron UID encountered will raise a ValueError
        if virtual_neurons is not None:
            target_dict = dict.fromkeys(virtual_neurons)
        else:
            target_dict = {}

        # Traverse the virtual FPGA->device connections
        for pre, post in virtual_connections:

            target_chip, target_core, _ = Router.decode_UID(post)

            # Illegal Key
            if virtual_neurons is not None and pre not in target_dict:
                raise ValueError(
                    f"Virtual neuron {pre} is not given in virtual_neurons : {virtual_neurons}!"
                )

            # First occurance
            elif pre not in target_dict or target_dict[pre] is None:
                target_dict[pre] = (target_chip, update_core_mask(0, target_core))

            # Update the core mask
            else:
                chipID, coreMask = target_dict[pre]

                # ChipID is different than expected!
                if chipID != target_chip:
                    raise ValueError(
                        f"A virtual neuron cannot broadcast to more than one chips!\n"
                        f"Virtual UID:{pre} -> Chip:{chipID}, cores:{Router.bitmask_select(coreMask)} (existing destination)\n"
                        f"Virtual UID:{pre} -> Chip:{target_chip}, core:{target_core}, (post-synaptic neuron:{post})"
                    )

                # Legal update
                else:
                    target_dict[pre] = (
                        chipID,
                        update_core_mask(coreMask, target_core),
                    )

        return target_dict

    @staticmethod
    def virtual_fan_out(
        target_dict: Optional[Dict[np.uint16, Tuple[np.uint8, np.uint8]]] = None,
        virtual_connections: Optional[List[NeuronConnection]] = None,
    ) -> List[NeuronConnection]:
        """
        virtual_fan_out produce a list of spike boardcasting connections specific for FPGA-to-device broadcast
        given a target dictionary or a list of virtual connections. The target dictionary provides us with a virtual
        pre-synaptic neurons and their target chip and core/cores to broadcast the events.
        From device's point of view, in each SRAM(Static Random Access Memory) cell
        the neuron can be set to broadcast it's spikes to one other chip. In the SRAM, one can also
        set a core mask to narrow down the number of neruons receiving the spikes. However, there is no
        space to set the neuronID. Therefore, a pre-synaptic neuron broadcast it's spike output
        to all the neuron in the specified core. The neurons at the post-synaptic side decide on listening or not.

        :param target_dict: a dictionary of targets of the virtual neurons pre_UID: (target_chip_ID, core_mask), defaults to None
        :type target_dict: Optional[Dict[np.uint16, Tuple[np.uint8, np.uint8]]], optional
        :param virtual_connections: [description], defaults to None
        :type virtual_connections: Optional[List[NeuronConnection]], optional
        :raises ValueError: Either target_dict or virtual_connections must be provided!
        :return: List of unique IDs of all virtual-real neuron connection pairs in the (pre, post) order.
        :rtype: List[NeuronConnection]
        """
        if target_dict is None and virtual_connections is None:
            raise ValueError(
                "Either target_dict or virtual_connections must be provided!"
            )

        fpga_out = []

        # First get a dictionary for virtual input connections. pre_UID: (target_chip_ID, core_mask)
        if target_dict is None:
            target_dict = Router.connect_input(virtual_connections)

        # Traverse the virtual connection dictionary
        for virtual_UID, (chip_ID, core_mask) in target_dict.items():
            fpga_out += Router.broadcasting_connections(
                neuron_UID=virtual_UID,
                target_chip_id=chip_ID,
                core_mask=core_mask,
            )

        return fpga_out

    @staticmethod
    def real_synapses(
        fan_in: List[NeuronConnectionSynType],
        fan_out: Optional[List[NeuronConnection]] = None,
    ) -> Dict[NeuronConnectionSynType, int]:
        """
        real_synapses produce a dictionary of synapses indicating the occurance of each synapses between
        the active neurons indicated in the active connection list. If only the fan_in is provided,
        then all the neurons in the fan_in list considered as active neurons.

        :param fan_in: Receving connection indicated in the listening side(CAM cells). list consisting of tuples : (preUID, postUID, syn_type)
        :type fan_in: List[NeuronConnectionSynType]
        :param fan_out: Sending connection indicated in the sending side(SRAM cells). list consisting of tuples : (preUID, postUID, syn_type), defaults to None
        :type fan_out: Optional[List[NeuronConnection]], optional
        :return: a dictionary for number of occurances of each synapses addressed by (preUID, postUID, syn_type) key
        :rtype: Dict[NeuronConnectionSynType, int]
        """

        # Get the number of occurances of the synapses in the fan_in list (preUID, postUID, syn_type)
        synapses, s_count = np.unique(fan_in, axis=0, return_counts=True)

        if fan_out is not None:
            # Skip the synapse type
            fan_in_no_type = np.unique(np.array(fan_in)[:, 0:2], axis=0)
            fan_out = np.unique(fan_out, axis=0)

            # Intersection of connections indicated in the sending side and the connections indicated in the listening side
            connections = list(
                set(map(tuple, fan_in_no_type)) & set(map(tuple, fan_out))
            )

        real_synapses = {}
        # key = preUID, postUID, syn_sype
        for i, key in enumerate(synapses):
            if fan_out is None or (key[0], key[1]) in connections:
                real_synapses[tuple(key)] = s_count[i]

        return real_synapses

    @staticmethod
    def virtual_synapses(
        fan_in: List[NeuronConnectionSynType],
        fan_out: Optional[List[NeuronConnection]] = None,
        real_synapses: Optional[Dict[NeuronConnectionSynType, int]] = None,
    ) -> Dict[NeuronConnectionSynType, int]:
        """
        virtual_synapses finds the pure virtual neurons indicated in the samna configuration object. Pure virtual neurons
        are the ones we can see in the fan_in list but we cannot see any of the 4 peers indicated by one CAM entry in the final connections list.
        One can should provide a fan_in list and choose to provide one of synapse dictionary and fan_out to obtain the virtual synapses.

        Please note that if a device neuron having the same ID with the virtual one is allocated, then this approach cannot find the implied virtual neuron

        :param fan_in: Receving connection indicated in the listening side(CAM cells). list consisting of tuples : (preUID, postUID, syn_type), defaults to None
        :type fan_in: List[NeuronConnectionSynType]
        :param fan_out: Sending connection indicated in the sending side(SRAM cells). list consisting of tuples : (preUID, postUID, syn_type), defaults to None
        :type fan_out: Optional[List[NeuronConnection]], optional
        :param real_synapses: a dictionary for number of occurances of each real(device-device) synapses addressed by(preUID, postUID, syn_type) key, defaults to None
        :type real_synapses: Optional[Dict[NeuronConnectionSynType, int]], optional
        :raises ValueError: Either provide a synanpse_dict & fan_in list or provide a fan_in & fan_out lists
        :return: a dictionary for number of occurances of each virtual(FPGA-DynapSE) synapses addressed by (preUID, postUID, syn_type) key
        :rtype: Dict[NeuronConnectionSynType, int]
        """

        if real_synapses is not None:
            connections = list(real_synapses.keys())
        elif fan_in is not None and fan_out is not None:
            connections = list(Router.real_synapses(fan_in, fan_out).keys())
        else:
            raise ValueError(
                "Either provide a synanpse_dict & fan_in list or provide a fan_in & fan_out lists"
            )

        # Find the all four counterparts of one device-device connection indicated by CAM in the fan_in list
        real_fan_in = []
        for preUID, postUID, syn_type in connections:
            _, coreID, neuronID = Router.decode_UID(preUID)
            real_fan_in += Router.receiving_connections(
                neuron_UID=postUID,
                listen_core_id=coreID,
                listen_neuron_id=neuronID,
                syn_type=syn_type,
            )

        # Virtual connections are the connections exist in the full fan_in list but missing in the real list
        virtual_fan_in = [vc for vc in fan_in if vc not in real_fan_in]

        # Virtual neurons can only have a chipID = 0
        virtual_connections = []
        for connection in virtual_fan_in:
            chipID, coreID, neuronID = Router.decode_UID(connection[0])
            if chipID == 0:
                virtual_connections.append(connection)

        virtual_synapses = Router.real_synapses(virtual_connections)
        return virtual_synapses

    @staticmethod
    def syn_type_map() -> Dict[int, str]:
        """
        syn_type_map creates a dictionary mapping the synapse type index to synapse type name

        :return: a dictionary of integer synapse type index keys and their names
        :rtype: Dict[int, str]
        """

        type_convert = lambda syn_type: (
            Dynapse1SynType(syn_type).value,
            Dynapse1SynType(syn_type).name,
        )

        return dict(map(type_convert, range(4)))

    @staticmethod
    def device_neurons(
        real_synapses: Dict[NeuronConnectionSynType, int],
        virtual_synapses: Optional[Dict[NeuronConnectionSynType, int]] = None,
        decode_UID: bool = True,
    ) -> List[Union[NeuronKey, np.uint16]]:
        """
        device_neurons finds the active neurons, that are either sending events to other neurons or accepting an events from others.
        It uses the real synapse and virtual synapse dictionaries for searching. Virtual synapse dictionary is optional but recommended
        because the post-synaptic neurons indicated there might refer to neurons which are missing in the real_synapse dictionary.
        These neurons are the ones whose only accept events from virtual(FPGA) neurons and does not have any incoming or outgoing connections with the other in-device neurons.

        :param real_synapses: a dictionary for number of occurances of each in-device synapse addressed by (preUID, postUID, syn_type) key
        :type real_synapses: Dict[NeuronConnectionSynType, int]
        :param virtual_synapses: a dictionary for number of occurances of each fpga-to-device synapse addressed by (preUID, postUID, syn_type) key, defaults to None
        :type virtual_synapses: Optional[Dict[NeuronConnectionSynType, int]], optional
        :param decode_UID: return a list of NeuronKey (chipID, coreID, neuronID) if true, return a list of NeuronUID (np.uint16) if false, defaults to True
        :type decode_UID: bool, optional
        :return: a unique list of active neurons in device (either sending or receiving events)
        :rtype: List[Union[NeuronKey, np.uint16]]
        """

        device_pre_post = np.empty(0)
        input_post = np.empty(0)

        # Extract the neurons from the synapse dictionaries
        if real_synapses:
            device_pre_post = np.array(list(real_synapses.keys()))[:, 0:2]
            device_pre_post = device_pre_post.flatten()

        # Destinations of the virtual synapses might indicate other neurons
        if virtual_synapses is not None:
            if virtual_synapses:
                input_post = np.array(list(virtual_synapses.keys()))[:, 1]

            device_neurons = np.unique(np.hstack((device_pre_post, input_post)))
        else:
            device_neurons = np.unique(device_pre_post)

        # Return
        if decode_UID:
            device_neurons = list(map(lambda t: Router.decode_UID(t), device_neurons))

        else:
            device_neurons = list(device_neurons)

        return device_neurons

    @staticmethod
    def fpga_neurons(
        virtual_synapses: Dict[NeuronConnectionSynType, int],
        decode_UID: bool = True,
    ) -> List[Union[NeuronKey, np.uint16]]:
        """
        fpga_neurons finds the active virtual neurons, that are sending events to other neurons. It uses the virtual synapse dictionary for searching.

        :param virtual_synapses: a dictionary for number of occurances of each fpga-to-device synapse addressed by (preUID, postUID, syn_type) key
        :type virtual_synapses: Dict[NeuronConnectionSynType, int]
        :param decode_UID: return a list of NeuronKey (chipID, coreID, neuronID) if true, return a list of NeuronUID (np.uint16) if false, defaults to True
        :type decode_UID: bool, optional
        :return: a unique list of active virtual neurons (which are sending events to in-device neurons)
        :rtype: List[Union[NeuronKey, np.uint16]]
        """

        input_pre = np.empty(0)

        if virtual_synapses:
            input_pre = np.array(list(virtual_synapses.keys()))[:, 0]

        # Obtain the neruons
        fpga_neruons = np.unique(input_pre)

        # Return
        if decode_UID:
            fpga_neruons = list(map(lambda t: Router.decode_UID(t), fpga_neruons))

        else:
            fpga_neruons = list(fpga_neruons)

        return fpga_neruons

    @staticmethod
    def get_CAM_matrix(
        synapses: Dict[NeuronConnectionSynType, int],
        pre_neurons: np.ndarray,
        post_neurons: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[int, NeuronKey]]:
        """
        get_CAM_matrix generates and fills a CAM matrix, the number of CAM defined per connection given the synapse dictionary, pre-synaptic neurons and the post-synaptic neurons.
        If post-synaptic neurons are not given, it's assumed that pre-synaptic and post-synaptic neurons are the same and connections are recurrent.

        :param synapses: a dictionary for number of occurances of each synapses addressed by (preUID, postUID, syn_type) key
        :type synapses: Dict[NeuronConnectionSynType, int]
        :param pre_neurons: a unique list of pre-synaptic neuron UIDs
        :type pre_neurons: np.ndarray
        :param post_neurons: a unique list of post-synaptic neuron UIDs, defaults to None
        :type post_neurons: Optional[np.ndarray], optional
        :return: CAM_matrix, idx_map
                :CAM_matrix: CAM matrix generated using the synapses dictionary
                :idx_map: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :rtype: Tuple[np.ndarray, Dict[int, NeuronKey]]
        """
        pre_idx = dict(zip(pre_neurons, range(len(pre_neurons))))

        if post_neurons is None:
            # Then it's assumed that a recurrent CAM matrix is demanded
            shape = (len(pre_neurons), len(pre_neurons), 4)
            post_idx = pre_idx
        else:
            # Rectangular input CAM matrix
            shape = (len(pre_neurons), len(post_neurons), 4)
            post_idx = dict(zip(post_neurons, range(len(post_neurons))))

        CAM_matrix = np.zeros(shape=shape, dtype=np.uint8)

        for (pre, post, syn_type), count in synapses.items():
            CAM_matrix[pre_idx[pre]][post_idx[post]][syn_type] = count

        idx_map = {pre_idx[n]: Router.decode_UID(n) for n in pre_neurons}

        return CAM_matrix, idx_map

    @staticmethod
    def core_matrix_from_idx_map(idx_map) -> np.ndarray:
        core_matrix = np.empty(len(idx_map), dtype=tuple)
        for idx, (chipID, coreID, _) in idx_map.items():
            core_matrix[idx] = (chipID, coreID)
        return core_matrix

    @staticmethod
    def CAM_in(
        device_synapses: Dict[NeuronConnectionSynType, int],
        input_synapses: Dict[NeuronConnectionSynType, int],
        return_maps: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]:
        """
        CAM_in creates a 3D rectangular shaped input CAM matrix given synapse dictionaries. The dictionary should
        have the number of occurances of each synapses indicated with (preUID, postUID, syn_type) key.
        The third dimension of the CAM matrix holds the different synapse types. For example,
        CAM[:,:,0] stands for the GABA_B connections. CAM[:,:,1] stands for the GABA_A connections and so on.
        The first dimension of the CAM matrix is for pre-synaptic neurons and the second dimension is for
        the post-synaptic neurons. The numbers stored indicate the number of synapses between respected neurons.
        If CAM[1][2][0] == 5, that means that there are 5 GABA_B connections from neuron 1 to neuron 2.

        :param device_synapses: a dictionary for number of occurances of each in-device synapses indicated with (preUID, postUID, syn_type) key
        :type device_synapses: Dict[NeuronConnectionSynType, int]
        :param input_synapses: a dictionary for number of occurances of each fpga-to-device synapses indicated with (preUID, postUID, syn_type) key
        :type input_synapses: Dict[NeuronConnectionSynType, int]
        :param return_maps: return the index-to-key map or not, defaults to True
        :type return_maps: bool, optional
        :return: CAM_in, in_idx
            :CAM_in: input CAM matrix (3D, NinxNrecx4)
            :in_idx: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]
        """

        CAM_in = np.empty(0)
        in_idx = {}

        # Extract the neurons from the synapse dictionaries
        device_neurons = Router.device_neurons(
            device_synapses, input_synapses, decode_UID=False
        )

        fpga_neurons = Router.fpga_neurons(input_synapses, decode_UID=False)
        CAM_in, in_idx = Router.get_CAM_matrix(
            input_synapses, fpga_neurons, device_neurons
        )

        # Return
        if not return_maps:
            return CAM_in

        else:
            return CAM_in, in_idx

    @staticmethod
    def CAM_rec(
        device_synapses: Dict[NeuronConnectionSynType, int],
        input_synapses: Optional[Dict[NeuronConnectionSynType, int]] = None,
        return_map: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]:
        """
        CAM_rec creates a 3D square shaped recurrent CAM matrix given synapse dictionaries. The dictionary should
        have the number of occurances of each synapses indicated with (preUID, postUID, syn_type) key.
        The third dimension of the CAM matrix holds the different synapse types. For example,
        CAM[:,:,0] stands for the GABA_B connections. CAM[:,:,1] stands for the GABA_A connections and so on.
        The first dimension of the CAM matrix is for pre-synaptic neurons and the second dimension is for
        the post-synaptic neurons. The numbers stored indicate the number of synapses between respected neurons.
        If CAM[1][2][0] == 5, that means that there are 5 GABA_B connections from neuron 1 to neuron 2.

        :param device_synapses: a dictionary for number of occurances of each in-device synapses indicated with (preUID, postUID, syn_type) key
        :type device_synapses: Dict[NeuronConnectionSynType, int]
        :param input_synapses: a dictionary for number of occurances of each fpga-to-device synapses indicated with (preUID, postUID, syn_type) key, defaults to None
        :type input_synapses: Optional[Dict[NeuronConnectionSynType, int]], optional
        :param return_maps: return the index-to-key map or not, defaults to True
        :type return_maps: bool, optional
        :return: CAM_rec, rec_idx
            :CAM_rec: recurrent CAM matrix (3D, NrecxNrecx4)
            :rec_idx: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]
        """

        CAM_rec = np.empty(0)
        rec_idx = {}

        # Extract the neurons from the synapse dictionaries
        device_neurons = Router.device_neurons(
            device_synapses, input_synapses, decode_UID=False
        )

        CAM_rec, rec_idx = Router.get_CAM_matrix(device_synapses, device_neurons)

        # Return
        if not return_map:
            return CAM_rec

        else:
            return CAM_rec, rec_idx

    @staticmethod
    def CAM_matrix(
        device_synapses: Dict[NeuronConnectionSynType, int],
        input_synapses: Optional[Dict[NeuronConnectionSynType, int]] = None,
        return_maps: bool = True,
    ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]]:
        """
        CAM_matrix runs `Router.CAM_in()` and `Router.CAM_rec()` together and creates a dictionary storing both the input
        and the recurrent CAM matrices. (For more detail, please look at `Router.CAM_in()` and `Router.CAM_rec()`)

        :param device_synapses: a dictionary for number of occurances of each in-device synapses indicated with (preUID, postUID, syn_type) key
        :type device_synapses: Dict[NeuronConnectionSynType, int]
        :param input_synapses: a dictionary for number of occurances of each fpga-to-device synapses indicated with (preUID, postUID, syn_type) key, defaults to None
        :type input_synapses: Optional[Dict[NeuronConnectionSynType, int]], optional
        :param return_maps: return the index-to-key map or not, defaults to True
        :type return_maps: bool, optional
        :return: a dictionary of tuples of input and recurrent dictionaries and their index maps. (For more detail, please look at Please look at `Router.CAM_in()` and `Router.CAM_rec()`)
        :rtype: Dict[str, Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]]
        """
        CAM_in = np.empty(0)

        if input_synapses is not None:
            CAM_in = Router.CAM_in(device_synapses, input_synapses, return_maps)

        CAM_rec = Router.CAM_rec(device_synapses, input_synapses, return_maps)

        # Return
        CAM_dict = {"CAM_in": CAM_in, "CAM_rec": CAM_rec}
        return CAM_dict

    # --- FROM CONFIG METHODS --- #

    @staticmethod
    def fan_from_config(
        config: Dynapse1Configuration,
    ) -> Tuple[List[NeuronConnectionSynType], List[NeuronConnectionSynType]]:
        """
        fan_from_config traverses the config object for neuron synapses and destinations.
        Produces a fan_in list which consists of all the possible connections indicated by CAM. 4 possibile connections indicated by 1 entry.
        Produces a fan_out list which consists of all the possible connections indicated by SRAM. 256 possible connections indicated by 1 entry.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :return: fan_in, fan_out
            :fan_in: Receving connection indicated in the listening side(CAM cells). list consisting of tuples : (preUID, postUID, syn_type)
            :fan_out: Sending connection indicated in the sending side(SRAM cells). list consisting of tuples : (preUID, postUID, syn_type)
        :rtype: Tuple[List[NeuronConnectionSynType], List[NeuronConnectionSynType]]
        """

        fan_in = []
        fan_out = []

        # Traverse the chip for neruon-neuron connections
        for chip in config.chips:  # 4
            for core in chip.cores:  # 4
                for neuron in core.neurons:  # 256

                    # FAN-IN (64 connections) CAM
                    for syn in neuron.synapses:
                        # An active synapse
                        if syn.listen_neuron_id != 0:
                            fan_in += Router.receiving_connections(neuron, syn)

                    # FAN-OUT (4 chips) SRAM
                    for dest in neuron.destinations:
                        # An active destination
                        if dest.target_chip_id != 16 and dest.target_chip_id != 0:
                            fan_out += Router.broadcasting_connections(neuron, dest)

        return fan_in, fan_out

    @staticmethod
    def real_synapses_from_config(
        config: Dynapse1Configuration,
    ) -> Dict[NeuronConnectionSynType, int]:
        """
        real_synapses_from_config extracts the real(in-device) synapses from the samna config object

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :return: a real synapse dictionary for number of occurances of each device-device synapses addressed by (preUID, postUID, syn_type) keys
        :rtype: Dict[NeuronConnectionSynType, int]
        """

        fan_in, fan_out = Router.fan_from_config(config)
        real_synapses = Router.real_synapses(fan_in, fan_out)

        return real_synapses

    @staticmethod
    def virtual_synapses_from_config(
        config: Dynapse1Configuration,
    ) -> Dict[NeuronConnectionSynType, int]:
        """
        virtual_synapses_from_config extracts the virtual(FPGA-to-device) synapses from the samna config object

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :return: a virtual synapse dictionary for number of occurances of each FPGA-device synapses addressed by (preUID, postUID, syn_type) keys
        :rtype: Dict[NeuronConnectionSynType, int]
        """

        fan_in, fan_out = Router.fan_from_config(config)
        virtual_synapses = Router.virtual_synapses(fan_in=fan_in, fan_out=fan_out)
        return virtual_synapses

    @staticmethod
    def synapses_from_config(
        config: Dynapse1Configuration,
    ) -> Dict[str, Dict[NeuronConnectionSynType, int]]:
        """
        synapses_from_config builts a synapse dictionary by traversing a samna DynapSE1 device configuration object

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :return: a super dictionary of `virtual` and `real` synapse dictionaries for number of occurances of each synapses addressed by (preUID, postUID, syn_type) keys
        :rtype: Dict[str, Dict[NeuronConnectionSynType, int]]
        """

        fan_in, fan_out = Router.fan_from_config(config)
        real_synapses = Router.real_synapses(fan_in, fan_out)
        virtual_synapses = Router.virtual_synapses(
            fan_in=fan_in, real_synapses=real_synapses
        )

        synapses = {
            "real": real_synapses,
            "virtual": virtual_synapses,
        }

        return synapses

    @staticmethod
    def device_neurons_from_config(
        config: Dynapse1Configuration, decode_UID: bool = True
    ) -> List[Union[NeuronKey, np.uint16]]:
        """
        device_neurons_from_config extracts the device neruon ids from the samna configuraiton object. Also look at `Router.device_neurons()`

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param decode_UID: return a list of NeuronKey (chipID, coreID, neuronID) if true, return a list of NeuronUID (np.uint16) if false, defaults to True
        :type decode_UID: bool, optional
        :return: a unique list of active neurons in device (either sending or receiving events)
        :rtype: List[Union[NeuronKey, np.uint16]]
        """
        synapses = Router.synapses_from_config(config)
        return Router.device_neurons(synapses["real"], synapses["virtual"], decode_UID)

    @staticmethod
    def fpga_neurons_from_config(
        config: Dynapse1Configuration, decode_UID: bool = True
    ) -> List[Union[NeuronKey, np.uint16]]:
        """
        fpga_neurons_from_config extracts the virtual FPGA neruon ids from the samna configuraiton object. Also look at `Router.fpga_neurons()`

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param decode_UID: return a list of NeuronKey (chipID, coreID, neuronID) if true, return a list of NeuronUID (np.uint16) if false, defaults to True
        :type decode_UID: bool, optional
        :return: a unique list of active virtual neurons (which are sending events to in-device neurons)
        :rtype: List[Union[NeuronKey, np.uint16]]
        """
        virtual_synapses = Router.virtual_synapses_from_config(config)
        return Router.fpga_neurons(virtual_synapses, decode_UID)

    @staticmethod
    def neurons_from_config(
        config: Dynapse1Configuration, decode_UID: bool = True
    ) -> Dict[str, List[Union[NeuronKey, np.uint16]]]:
        """
        neurons_from_config builts a neurons key dictionary by traversing a samna DynapSE1 device configuration object

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param decode_UID: return a list of NeuronKey (chipID, coreID, neuronID) if true, return a list of NeuronUID (np.uint16) if false, defaults to True
        :type decode_UID: bool, optional
        :return: a unique list of active neurons in device (either sending or receiving events)
        :rtype: Dict[str, List[Union[NeuronKey, np.uint16]]]
        """

        synapses = Router.synapses_from_config(config)

        device_neurons = Router.device_neurons(
            synapses["real"], synapses["virtual"], decode_UID
        )
        fpga_neurons = Router.fpga_neurons(synapses["virtual"], decode_UID)

        neurons = {
            "real": device_neurons,
            "virtual": fpga_neurons,
        }

        return neurons

    @staticmethod
    def CAM_in_from_config(
        config: Dynapse1Configuration, return_maps: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]:
        """
        CAM_in_from_config Use `Router.synapses_from_config()` and `Router.CAM_in()` functions together to extract
        the input CAM matrix from a samna config object. `Router.synapses_from_config()` creates the
        synapse dictionaries from a configuration object and `Router.CAM_in()` converts the dictionary to a CAM matrix.
        For details of the algorithms, please check the functions.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param return_maps: return the index-to-key map or not, defaults to True
        :type return_maps: bool, optional
        :return: CAM_in, in_idx
            :CAM_in: input CAM matrix (3D, NinxNrecx4)
            :in_idx: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]
        """
        syn_dict = Router.synapses_from_config(config)
        return Router.CAM_in(syn_dict["real"], syn_dict["virtual"], return_maps)

    @staticmethod
    def CAM_rec_from_config(
        config: Dynapse1Configuration, return_maps: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]:
        """
        CAM_rec_from_config Use `Router.synapses_from_config()` and `Router.CAM_rec()` functions together to extract
        the input CAM matrix from a samna config object. `Router.synapses_from_config()` creates the
        synapse dictionaries from a configuration object and `Router.CAM_rec()` converts the dictionary to a CAM matrix.
        For details of the algorithms, please check the functions.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param return_maps: return the index-to-key map or not, defaults to True
        :type return_maps: bool, optional
        :return: CAM_rec, rec_idx
            :CAM_rec: recurrent CAM matrix (3D, NrecxNrecx4)
            :rec_idx: a dictionary of the mapping between matrix indexes of the neurons and their neuron keys
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]
        """
        syn_dict = Router.synapses_from_config(config)
        return Router.CAM_rec(syn_dict["real"], syn_dict["virtual"], return_maps)

    @staticmethod
    def CAMs_from_config(
        config: Dynapse1Configuration,
        return_maps: bool = False,
    ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]]:
        """
        CAMs_from_config Use `Router.synapses_from_config()` and `Router.CAM_matrix()` functions together to extract
        the input and recurrent CAM matrices from a samna config object. `Router.synapses_from_config()` creates the
        synapse dictionaries from a configuration object and `Router.CAM_matrix()` converts the dictionary to a CAM matrix.
        For details of the algorithms, please check the functions.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param return_maps: return the index-to-key map or not, defaults to True
        :type return_maps: bool, optional
        :return: a dictionary of tuples of input and recurrent dictionaries and their index maps. (For more detail, please look at Please look at `Router.CAM_in()` and `Router.CAM_rec()`)
        :rtype: Dict[str, Union[np.ndarray, Tuple[np.ndarray, Dict[int, NeuronKey]]]]
        """
        syn_dict = Router.synapses_from_config(config)
        return Router.CAM_matrix(syn_dict["real"], syn_dict["virtual"], return_maps)
