"""
Dynap-SE1 router simulator. Create a weight matrix using SRAM and CAM content

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
13/09/2021
"""


from typing import (
    Dict,
    Iterable,
    Union,
    List,
    Optional,
    Tuple,
)

import numpy as np

ArrayLike = Union[np.ndarray, List, Tuple]
Numeric = Union[int, float, complex, np.number]
NeuronKey = Tuple[np.uint8, np.uint8, np.uint16]
NeuronConnection = Tuple[np.uint16, np.uint16]
NeuronConnectionSynType = Tuple[np.uint16, np.uint16, np.uint8]

_SAMNA_AVAILABLE = True

try:
    from samna.dynapse1 import (
        Dynapse1Configuration,
        Dynapse1Destination,
        Dynapse1Synapse,
        Dynapse1SynType,
        Dynapse1Neuron,
    )
except ModuleNotFoundError as e:
    print(
        e,
        "\nDynapSE1NeuronSynapseJax module can only be used for simulation purposes."
        "Deployment utilities depends on samna!",
    )
    _SAMNA_AVAILABLE = False

_SAMNA_AVAILABLE = True

try:
    from netgen import (
        Network,
        NetworkGenerator,
        convert_incoming_conns_dict2list,
    )
except ModuleNotFoundError as e:
    print(
        e,
        "\nRouter cannot extract the virtual connections from the network generator object!",
    )
    _SAMNA_AVAILABLE = False


class Router:
    NUM_CHIPS = np.uint8(4)
    NUM_CORES = np.uint8(4)
    NUM_NEURONS = np.uint16(256)
    NUM_SYNAPSES = np.uint16(64)
    NUM_DESTINATION_TAGS = np.uint8(4)
    NUM_POISSON_SOURCES = np.uint16(1024)

    f_chip = NUM_NEURONS * NUM_CORES
    f_core = NUM_NEURONS

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

        uid = Router.f_chip * chipID + Router.f_core * coreID + neuronID
        return np.uint16(uid)

    @staticmethod
    def decode_UID(UID: np.uint16) -> NeuronKey:
        """
        decode_UID decodes the uniquie neuron ID

        :param UID: Globally unique neuron ID
        :type UID: np.uint16
        :return: the neuron key composed of chipID, coreID and neuronID in order
        :rtype: NeuronKey
        """

        # CHIP
        chipID = UID // Router.f_chip

        # CORE
        UID -= Router.f_chip * chipID
        coreID = UID // Router.f_core

        # NEURON
        neuronID = UID - Router.f_core * coreID

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
    def select_coreID_with_mask(core_mask: np.uint8) -> np.ndarray:
        """
        select_coreID_with_mask apply bit mask to select cores
        0001 -> selected coreID: 0
        1000 -> selected coreID: 3
        0101 -> selected coreID 0 and 2

        :param core_mask: Binary mask to select core IDs
        :type core_mask: np.uint8
        :return: an array of IDs of selected cores
        :rtype: np.ndarray
        """

        coreID = range(Router.NUM_CORES)  # [3,2,1,0]
        bit_pattern = lambda n: (1 << n)  # 2^n

        # Indexes of the IDs to be selected in coreID list
        idx = np.array([core_mask & bit_pattern(cid) for cid in coreID], dtype=bool)
        coreID_selected = np.array(coreID, dtype=np.uint8)[idx]
        return coreID_selected

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
        cores_to_send = Router.select_coreID_with_mask(core_mask)
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
            virtual_core_id,  # pretend
            neuron_id,
        )

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
            input_dict = dict.fromkeys(virtual_neurons)
        else:
            input_dict = {}

        # Traverse the virtual FPGA->device connections
        for pre, post in virtual_connections:

            target_chip, target_core, _ = Router.decode_UID(post)

            # Illegal Key
            if virtual_neurons is not None and pre not in input_dict:
                raise ValueError(
                    f"Virtual neuron {pre} is not given in virtual_neurons : {virtual_neurons}!"
                )

            # First occurance
            elif pre not in input_dict or input_dict[pre] is None:
                input_dict[pre] = (target_chip, update_core_mask(0, target_core))

            # Update the core mask
            else:
                chipID, coreMask = input_dict[pre]

                # ChipID is different than expected!
                if chipID != target_chip:
                    raise ValueError(
                        f"A virtual neuron cannot broadcast to more than one chips!\n"
                        f"Virtual UID:{pre} -> Chip:{chipID}, cores:{Router.select_coreID_with_mask(coreMask)} (existing destination)\n"
                        f"Virtual UID:{pre} -> Chip:{target_chip}, core:{target_core}, (post-synaptic neuron:{post})"
                    )

                # Legal update
                else:
                    input_dict[pre] = (
                        chipID,
                        update_core_mask(coreMask, target_core),
                    )

        return input_dict

    @staticmethod
    def synapse_dict(
        fan_in: List[NeuronConnectionSynType],
        fan_out: Optional[List[NeuronConnection]] = None,
    ) -> Dict[NeuronConnectionSynType, int]:
        """
        synapse_dict [summary]

        :param fan_in: [description]
        :type fan_in: List[NeuronConnectionSynType]
        :param fan_out: [description], defaults to None
        :type fan_out: Optional[List[NeuronConnection]], optional
        :return: [description]
        :rtype: Dict[NeuronConnectionSynType, int]
        """
        """
        synapse_dict produce a dictionary of synapses indicating the occurance of each synapses between
        the active neurons indicated in the active connection list. If only the fan_in is provided,
        then all the neurons in the fan_in list considered as active neurons.

        :param fan_in: Receving connection indicated in the listening side(CAM cells). list consisting of tuples : (preUID, postUID, syn_type)
        :type fan_in: List[NeuronConnectionSynType]
        :param fan_out: Sending connection indicated in the sending side(SRAM cells). list consisting of tuples : (preUID, postUID, syn_type), defaults to None
        :type fan_out: Optional[List[NeuronConnection]], optional
        :return: a dictionary for number of occurances of each synapses indicated with (preUID, postUID, syn_type) key
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

        synapse_dict = {}
        # key = preUID, postUID, syn_sype
        for i, key in enumerate(synapses):
            if fan_out is None or (key[0], key[1]) in connections:
                synapse_dict[tuple(key)] = s_count[i]

        return synapse_dict

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
        syn_type_map = dict(map(type_convert, range(4)))

        return syn_type_map

    @staticmethod
    def synapses_from_config(
        config: Dynapse1Configuration,
        virtual_connections: Optional[List[NeuronConnection]] = None,
        return_syn_type: bool = False,
    ) -> Union[
        Dict[NeuronConnectionSynType, int],
        Tuple[Dict[NeuronConnectionSynType, int], Dict[int, str]],
    ]:
        """
        synapses_from_config builts a synapse dictionary by traversing a samna DynapSE1 device configuration object

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param virtual_connections: A list of tuples of universal neuron IDs defining the input connections from the FPGA to device neurons, defaults to None
            e.g : [(50,1044),(50,1045)]
        :type virtual_connections: Optional[List[NeuronConnection]], optional
        :param return_syn_type: return a dictionary of synapse types along with the , defaults to False
        :type return_syn_type: bool, optional
        :return: synapses, synapse_type_map
            synapses: a super dictionary of `virtual` and `real` synapse dictionaries for number of occurances of each synapses indicated with (preUID, postUID, syn_type) keys
            synapse_type_map : a dictionary of integer synapse type index keys and their names
        :rtype: Union[Dict[str,Dict[NeuronConnectionSynType, int]], Tuple[Dict[str,Dict[NeuronConnectionSynType, int]], Dict[int,str]]]
        """

        fan_in = []
        fan_out = []
        fpga_out = []

        # Traverse the chip for neruon-neuron connections
        for chip in config.chips:  # 4
            for core in chip.cores:  # 4
                for neuron in core.neurons:  # 256

                    # FAN-IN (64)
                    for syn in neuron.synapses:
                        # An active synapse
                        if syn.listen_neuron_id != 0:
                            fan_in += Router.receiving_connections(neuron, syn)

                    # FAN-OUT (4)
                    for dest in neuron.destinations:
                        # An active destination
                        if dest.target_chip_id != 16 and dest.target_chip_id != 0:
                            fan_out += Router.broadcasting_connections(neuron, dest)

        # Get the virtual input synapses (FPGA neuron -> device neuron)
        if virtual_connections is not None:

            # First create a dictionary for virtual input connections. pre_UID: (target_chip_ID, core_mask)
            input_dict = Router.connect_input(virtual_connections)

            # Traverse the virtual connection dictionary
            for virtual_UID, (chip_ID, core_mask) in input_dict.items():
                fpga_out += Router.broadcasting_connections(
                    neuron_UID=virtual_UID,
                    target_chip_id=chip_ID,
                    core_mask=core_mask,
                )

        # Need target chipID, core mask
        real_synapses = Router.synapse_dict(fan_in, fan_out)
        virtual_synapses = Router.synapse_dict(fan_in, fpga_out)

        synapses = {
            "real": real_synapses,
            "virtual": virtual_synapses,
        }

        if not return_syn_type:
            return synapses
        else:
            # Which synapse type index (the first dimension) correspond to which synapse type
            syn_type_map = Router.syn_type_map()
            return synapses, syn_type_map

    @staticmethod
    def weight_matrix(
        device_synapses: Dict[NeuronConnectionSynType, int],
        input_synapses: Dict[NeuronConnectionSynType, int],
        dtype: type = np.uint8,
        return_maps: bool = True,
        decode_UID: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[
            np.ndarray,
            np.ndarray,
            Union[Dict[int, np.uint16], Dict[int, NeuronKey]],
            Dict[int, str],
        ],
    ]:
        """
        weight_matrix creates a 3D weight matrix given a synapse dictionary. The dictionary should
        have the number of occurances of each synapses indicated with (preUID, postUID, syn_type) key.
        One can use `Router.synapses_from_config()` function to create this dictionary out of a samna config object.
        The third dimension of the weight matrix holds the different synapse types. For example,
        weight[:,:,0] stands for the GABA_B connections. weight[:,:,1] stands for the GABA_A connections and so on.
        The first dimension of the weight matrix is for pre-synaptic neurons and the second dimension is for
        the post-synaptic neurons. The numbers stored indicate the number of synapses between respected neurons.
        If weight[1][2][0] == 5, that means that there are 5 GABA_B connections from neuron 1 to neuron 2.

        :param device_synapses: a dictionary for number of occurances of each in-device synapses indicated with (preUID, postUID, syn_type) key
        :type device_synapses: Dict[NeuronConnectionSynType, int]
        :param input_synapses: a dictionary for number of occurances of each fpga-to-device synapses indicated with (preUID, postUID, syn_type) key
        :type input_synapses: Dict[NeuronConnectionSynType, int]
        :param dtype: numeric type of the weight matrix. For Dynap-SE1, there are at most 64 connections between neurons so dtype defaults to np.uint8
        :type dtype: type, optional
        :param return_maps: return the index-to-UID, and syn-index-to-type maps or not, defaults to True
        :type return_maps: bool, optional
        :param decode_UID: decode the UID to get a key instead or not, defaults to False
        :type decode_UID: bool, optional
        :return: weight, index_UID_map
            w_in: input weight matrix (3D, NinxNrecx4)
            w_rec: recurrent weight matrix (3D, NrecxNrecx4)
            index_UID_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique IDs (or keys)
            syn_type_map:a  dictionary of integer synapse type index keys and their names
        :rtype: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Union[Dict[int, np.uint16], Dict[int, NeuronKey]] , Dict[int, str]]]
        """

        def get_weight_matrix(
            synapses: Dict[NeuronConnectionSynType, int],
            pre_neurons: np.ndarray,
            post_neurons: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, Dict[int, np.uint16]]:
            """
            get_weight_matrix generates and fills a weight matrix given the synapse dictionary, pre-synaptic neurons and the post-synaptic neurons.
            If post-synaptic neurons are not given, it's assumed that pre-synaptic and post-synaptic neurons are the same and connections are recurrent.

            :param synapses: a dictionary for number of occurances of each synapses indicated with (preUID, postUID, syn_type) key
            :type synapses: Dict[NeuronConnectionSynType, int]
            :param pre_neurons: a unique list of pre-synaptic neuron UIDs
            :type pre_neurons: np.ndarray
            :param post_neurons: a unique list of post-synaptic neuron UIDs, defaults to None
            :type post_neurons: Optional[np.ndarray], optional
            :return: weight_matrix, idx_map
                :weight_matrix: weight matrix generated using the synapses dictionary
                :idx_map: index map 
            :rtype: Tuple[np.ndarray, Dict[int, np.uint16]]
            """
            pre_idx = dict(zip(pre_neurons, range(len(pre_neurons))))

            if post_neurons is None:
                # Then it's assumed that a recurrent weight matrix is demanded
                shape = (len(pre_neurons), len(pre_neurons), 4)
                post_idx = pre_idx
            else:
                # Rectangular input weight matrix
                shape = (len(pre_neurons), len(post_neurons), 4)
                post_idx = dict(zip(post_neurons, range(len(post_neurons))))

            weight_matrix = np.zeros(shape=shape, dtype=dtype)

            for (pre, post, syn_type), count in synapses.items():
                weight_matrix[pre_idx[pre]][post_idx[post]][syn_type] = count

            idx_map = {pre_idx[n]: n for n in pre_neurons}

            return weight_matrix, idx_map

        # Extract the neurons from the synapse dictionaries
        device_pre_post = np.array(list(device_synapses.keys()))[:, 0:2]
        input_pre = np.array(list(input_synapses.keys()))[:, 0]
        input_post = np.array(list(input_synapses.keys()))[:, 1]

        # Get the device and virtual neurons seperately
        device_neurons = np.unique(np.hstack((device_pre_post.flatten(), input_post)))
        virtual_neurons = np.unique(input_pre)

        # Get weight matrices
        w_in, in_idx = get_weight_matrix(
            input_synapses, virtual_neurons, device_neurons
        )
        w_rec, rec_idx = get_weight_matrix(device_synapses, device_neurons)

        # Return
        if not return_maps:
            return w_in, w_rec

        else:
            index_UID_map = {"FPGA": in_idx, "Dynap-SE1": rec_idx}

            if decode_UID:
                # Decode the index UID map to represent neurons with NeuronKeys instead of UIDs
                map_decoder = lambda row: (row[0], Router.decode_UID(row[1]))
                traverse = lambda row: (row[0], dict(map(map_decoder, row[1].items())))
                index_UID_map = dict(map(traverse, index_UID_map.items()))

            syn_type_map = Router.syn_type_map()

            return w_in, w_rec, index_UID_map, syn_type_map

    @staticmethod
    def get_weight_from_config(
        config: Dynapse1Configuration,
        virtual_connections: Optional[List[NeuronConnection]] = None,
        dtype: type = np.uint8,
        return_maps: bool = False,
        decode_UID: bool = False,
    ) -> Union[
        np.ndarray,
        Tuple[
            np.ndarray,
            Union[Dict[int, np.uint16], Dict[int, NeuronKey]],
            Dict[int, str],
        ],
    ]:
        """
        get_weight_from_config Use `synapses_from_config` and `weight_matrix` functions together to get a weight matrix
        from a samna config object. `synapses_from_config` creates the synapse dictionaries from a configuration object
        and `weight_matrix` converts the dictionary to a weight matrix. For details of the algorithms, please check the functions.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param virtual_connections: A list of tuples of universal neuron IDs defining the input connections from the FPGA to device neurons, defaults to None
            e.g : [(50,1044),(50,1045)]
        :type virtual_connections: Optional[List[NeuronConnection]], optional
        :param dtype: numeric type of the weight matrix. For Dynap-SE1, there are at most 64 connections between neurons so dtype defaults to np.uint8
        :type dtype: type, optional
        :param return_maps: return the index-to-UID, and syn-index-to-type maps or not, defaults to True
        :type return_maps: bool, optional
        :param decode_UID: decode the UID to get a key instead or not, defaults to False
        :type decode_UID: bool, optional
        :return: weight, index_UID_map, syn_dict
            w_in: input weight matrix (3D, NinxNrecx4)
            w_rec: recurrent weight matrix (3D, NrecxNrecx4)
            index_UID_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique IDs (or keys)
            syn_type_map:a  dictionary of integer synapse type index keys and their names
            syn_dict: a super dictionary of `virtual` and `real` synapse dictionaries for number of occurances of each synapses indicated with (preUID, postUID, syn_type) keys
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Union[Dict[int, np.uint16], Dict[int, NeuronKey]] , Dict[int, str]]]
        """
        syn_dict = Router.synapses_from_config(config, virtual_connections)

        real = syn_dict["real"]
        virtual = syn_dict["virtual"]

        if return_maps:
            return (
                *Router.weight_matrix(real, virtual, dtype, return_maps, decode_UID),
                syn_dict,
            )
        else:
            return Router.weight_matrix(real, virtual, dtype, return_maps, decode_UID)

    @staticmethod
    def get_virtual_connections(
        network: Network, append_syn_type: bool = False
    ) -> Union[List[NeuronConnection], List[NeuronConnectionSynType]]:
        """
        get_virtual_connections generates a neuron connection list between virtual FPGA neurons and real in-device neurons.
        This list then can be used to provide the input_connections to `Router.weight_matrix()` method

        :param network: network object defined in samna/ctxctl_contrib/netgen
        :type network: Network
        :param append_syn_type: specify the synapse type of the connection or not, defaults to False
        :type append_syn_type: bool, optional
        :return: a list of virtual-real neuron connections in either specifying the synapse type or not.
        :rtype: Union[List[NeuronConnection], List[NeuronConnectionSynType]]
        """

        connections = []

        # Traverse the post_neuron dictionary (1, 0): [C1c0n20, C1c0n36]
        for _, post_list in network.post_neuron_dict.items():

            for post in post_list:  # [C1c0n20, C1c0n36]
                incoming_list, _ = convert_incoming_conns_dict2list(
                    post.incoming_connections
                )

                # [(C0c0s50, <Dynapse1SynType.AMPA: 3>), (C3c0n60, <Dynapse1SynType.GABA_B: 0>)]
                for pre, syn_type in incoming_list:

                    # In a virtual connection, the pre-synaptic neurons is a spike-gen on the FPGA
                    if pre.is_spike_gen:

                        # Get universal neuron ids
                        pre_UID = Router.get_UID(
                            pre.chip_id, pre.core_id, pre.neuron_id
                        )
                        post_UID = Router.get_UID(
                            post.chip_id, post.core_id, post.neuron_id
                        )

                        conn = (
                            (pre_UID, post_UID, syn_type.value)
                            if append_syn_type
                            else (pre_UID, post_UID)
                        )

                        connections.append(conn)

        return connections

    @staticmethod
    def get_weight_from_netgen(
        netgen: NetworkGenerator, *args, **kwargs
    ) -> Union[
        np.ndarray,
        Tuple[
            np.ndarray,
            Union[Dict[int, np.uint16], Dict[int, NeuronKey]],
            Dict[int, str],
        ],
    ]:
        """
        get_weight_from_netgen a wrapper function which makes it easier to get a weight matrix using the `NetworkGenerator` object.
        Extract the configuration and virtual connections from the network generator and then
        runs the `Router.get_weight_from_config()` object with given parameter set.

        :param netgen: network generator object defined in samna/ctxctl_contrib/netgen
        :type netgen: NetworkGenerator
        :param dtype: numeric type of the weight matrix. For Dynap-SE1, there are at most 64 connections between neurons so dtype defaults to np.uint8
        :type dtype: type, optional
        :param return_maps: return the index-to-UID, and syn-index-to-type maps or not, defaults to True
        :type return_maps: bool, optional
        :param decode_UID: decode the UID to get a key instead or not, defaults to False
        :type decode_UID: bool, optional
        :return: weight, index_UID_map, syn_dict
            w_in: input weight matrix (3D, NinxNrecx4)
            w_rec: recurrent weight matrix (3D, NrecxNrecx4)
            index_UID_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique IDs (or keys)
            syn_type_map:a  dictionary of integer synapse type index keys and their names
            syn_dict: a super dictionary of `virtual` and `real` synapse dictionaries for number of occurances of each synapses indicated with (preUID, postUID, syn_type) keys
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Union[Dict[int, np.uint16], Dict[int, NeuronKey]] , Dict[int, str]]]
        """

        connections = Router.get_virtual_connections(netgen.network)
        config = netgen.make_dynapse1_configuration()
        return Router.get_weight_from_config(config, connections, *args, **kwargs)
