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
        Dynapse1Neuron,
        Dynapse1Synapse,
        Dynapse1Destination,
        Dynapse1SynType,
        Dynapse1Configuration,
    )
except ModuleNotFoundError as e:
    print(
        e,
        "\nDynapSE1NeuronSynapseJax module can only be used for simulation purposes."
        "Deployment utilities depends on samna!",
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
        neuron: Dynapse1Neuron,
        synapse: Dynapse1Synapse,
    ) -> List[NeuronConnectionSynType]:
        """
        receiving_connections produce a list of spike receiving connections given a neuron and synapse object
        From the device's point of view, in each CAM(Content Addressable Memory) cell
        the neuron can be set to listen to (i.e. receive events from) one other neuron with a specified synapse type.
        In the CAM, the core and neuron IDs can be set, but there is no space to set the chipID.
        Therefore, a post-synaptic neuron listens all the pre-synaptic neurons having the
        same core and neuron ID across different chips.

        :param neuron: The neuron at the post-synaptic side
        :type neuron: Dynapse1Neuron
        :param synapse: High level content of a CAM cell
                        "syn_type": 2,
                        "listen_neuron_id": 0,
                        "listen_core_id": 0
        :type synapse: Dynapse1Synapse
        :return: List of unique IDs of all neuron connection pairs in the (pre, post, syn_type) order.
        :rtype: List[NeuronConnectionSynType]
        """

        # Pre-synaptic neurons to listen across 4 chips
        pre_list = Router.get_UID_combination(
            chipID=None,
            coreID=synapse.listen_core_id,
            neuronID=synapse.listen_neuron_id,
        )

        # Post-synaptic neuron
        post = Router.get_UID(neuron.chip_id, neuron.core_id, neuron.neuron_id)
        connections = Router.connect_pre_post(pre_list, post, synapse.syn_type)

        return connections

    @staticmethod
    def broadcasting_connections(
        neuron: Dynapse1Neuron, destination: Dynapse1Destination
    ) -> List[NeuronConnection]:
        """
        broadcasting_connections produce a list of spike boardcasting connections given a neuron and a destination object.
        From device's point of view, in each SRAM(Static Random Access Memory) cell
        the neuron can be set to broadcast it's spikes to one other chip. In the SRAM, one can also
        set a core mask to narrow down the number of neruons receiving the spikes. However, there is no
        space to set the neuronID. Therefore, a pre-synaptic neuron broadcast it's spike output
        to all the neuron in the specified core. The neurons at the post-synaptic side decide on listening or not.

        :param neuron: The neuron at the pre-synaptic side
        :type neuron: Dynapse1Neuron
        :param destination: High level content of the SRAM cell
                            "targetChipId": 0,
                            "inUse": false,
                            "virtualCoreId": 0,
                            "coreMask": 0,
                            "sx": 0,
                            "sy": 0,
                            "dx": 0,
                            "dy": 0
        :type destination: Dynapse1Destination
        :return: List of unique IDs of all neuron connection pairs in the (pre, post) order.
        :rtype: List[NeuronConnection]
        """

        cores_to_send = Router.select_coreID_with_mask(destination.core_mask)
        if len(cores_to_send) == 0:
            return []

        # Pre-synaptic neurons to broadcast spike events
        post_list = Router.get_UID_combination(
            chipID=destination.target_chip_id,
            coreID=cores_to_send,
            neuronID=None,
        )

        # Pre-synaptic neuron
        pre = Router.get_UID(
            neuron.chip_id,
            destination.virtual_core_id,  # pretend
            neuron.neuron_id,
        )

        connections = Router.connect_pre_post(pre, post_list)
        return connections

    @staticmethod
    def connect_pre_post(
        preUID: Union[np.uint16, ArrayLike],
        postUID: Union[np.uint16, ArrayLike],
        syn_type: Optional[Dynapse1SynType] = None,
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
        :type syn_type: Optional[Dynapse1SynType], optional
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

        if isinstance(postUID, (tuple, list, np.ndarray)):
            if n_connections == 1:
                n_connections = len(postUID)
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
    def synapse_dict(
        fan_in: List[NeuronConnectionSynType], fan_out: List[NeuronConnection]
    ) -> Dict[NeuronConnectionSynType, int]:
        """
        synapse_dict produce a dictionary of synapses indicating the occurance of each synapses between
        the active neurons indicated in the active connection list

        :param fan_in: Receving connection indicated in the listening side(CAM cells). list consisting of tuples : (preUID, postUID, syn_type)
        :type fan_in: List[NeuronConnectionSynType]
        :param fan_out: Sending connection indicated in the sending side(SRAM cells). list consisting of tuples : (preUID, postUID, syn_type)
        :type fan_out: List[NeuronConnection]
        :return: a dictionary for number of occurances of each synapses indicated with (preUID, postUID, syn_type) key
        :rtype: Dict[NeuronConnectionSynType, int]
        """

        # Get the number of occurances of the synapses in the fan_in list (preUID, postUID, syn_type)
        synapses, s_count = np.unique(fan_in, axis=0, return_counts=True)

        # Skip the synapse type
        fan_in_no_type = np.unique(np.array(fan_in)[:, 0:2], axis=0)
        fan_out = np.unique(fan_out, axis=0)

        # Intersection of connections indicated in the sending side and the connections indicated in the listening side
        connections = list(set(map(tuple, fan_in_no_type)) & set(map(tuple, fan_out)))

        synapse_dict = {}
        # key = preUID, postUID, syn_sype
        for i, key in enumerate(synapses):
            if (key[0], key[1]) in connections:
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
        config: Dynapse1Configuration, return_syn_type: bool = False
    ) -> Union[
        Dict[NeuronConnectionSynType, int],
        Tuple[Dict[NeuronConnectionSynType, int], Dict[int, str]],
    ]:
        """
        synapses_from_config builts a synapse dictionary by traversing a samna DynapSE1 device configuration object

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param return_syn_type: return a dictionary of synapse types along with the , defaults to False
        :type return_syn_type: bool, optional
        :return: synapses, synapse_type_map
            synapses: a dictionary for number of occurances of each synapses indicated with (preUID, postUID, syn_type) key
            synapse_type_map : a dictionary of integer synapse type index keys and their names
        :rtype: Union[Dict[NeuronConnectionSynType, int], Tuple[Dict[NeuronConnectionSynType, int], Dict[int,str]]]
        """

        fan_in = []
        fan_out = []

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

        synapses = Router.synapse_dict(fan_in, fan_out)

        if not return_syn_type:
            return synapses
        else:
            # Which synapse type index (the first dimension) correspond to which synapse type
            syn_type_map = Router.syn_type_map()
            return synapses, syn_type_map

    @staticmethod
    def weight_matrix(
        synapse_dict: Dict[NeuronConnectionSynType, int],
        dtype: type = np.uint8,
        return_maps: bool = True,
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
        weight_matrix creates a 3D weight matrix given a synapse dictionary. The dictionary should
        have the number of occurances of each synapses indicated with (preUID, postUID, syn_type) key.
        One can use `Router.synapses_from_config()` function to create this dictionary out of a samna config object.
        The first dimension of the weight matrix holds the different synapse types. For example,
        weight[0] stands for the GABA_B connections. weight[1] stands for the GABA_A connections and so on.
        The second dimension of the weight matrix is for pre-synaptic neurons and the third dimension is for
        the post-synaptic neurons. The numbers stored indicate the number of synapses between respected neurons.
        If weight[0][1][2] == 5, that means that there are 5 GABA_B connections from neuron 1 to neuron 2.

        :param synapse_dict: a dictionary for number of occurances of each synapses indicated with (preUID, postUID, syn_type) key
        :type synapse_dict: Dict[NeuronConnectionSynType, int]
        :param dtype: numeric type of the weight matrix. For Dynap-SE1, there are at most 64 connections between neurons so dtype defaults to np.uint8
        :type dtype: type, optional
        :param return_maps: return the index-to-UID, and syn-index-to-type maps or not, defaults to True
        :type return_maps: bool, optional
        :param decode_UID: decode the UID to get a key instead or not, defaults to False
        :type decode_UID: bool, optional
        :return: weight, index_UID_map
            weight: 3D weight matrix indicating number of synapses between neruons
            index_UID_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique IDs (or keys)
            syn_type_map:a  dictionary of integer synapse type index keys and their names
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Union[Dict[int, np.uint16], Dict[int, NeuronKey]] , Dict[int, str]]]
        """

        # Assign matrix indices to the neurons indicated in synapse_dict [(preUID, postUID, syn_type) : num_connections]
        pre_post = np.array(list(synapse_dict.keys()))[:, 0:2]
        neurons = np.unique(pre_post)
        idx = dict(zip(neurons, range(len(neurons))))

        # weight[synapse_type][pre-synaptic neuron index][post-synaptic neuron index]
        weight = np.zeros((4, len(idx), len(idx)), dtype=dtype)

        # Traverse synapse dictionary
        for (pre, post, syn_type), count in synapse_dict.items():
            weight[syn_type][idx[pre]][idx[post]] = count

        if not return_maps:
            return weight

        else:
            # Which pre,post index (second and third dimensions) correspond to which neuron? universal ID
            if decode_UID:
                index_UID_map = {v: Router.decode_UID(k) for k, v in idx.items()}
            else:
                index_UID_map = {v: k for k, v in idx.items()}

            # Which synapse type index (the first dimension) correspond to which synapse type
            syn_type_map = Router.syn_type_map()

            return weight, index_UID_map, syn_type_map

    @staticmethod
    def get_weight_from_config(
        config: Dynapse1Configuration,
        dtype: type = np.uint8,
        return_maps: bool = True,
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

        Router.get_weight_from_config(config)

        (array([[[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],

                [[0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],

                [[0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],

                [[0, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]], dtype=uint8),
        {0: 1044, 1: 1084, 2: 1176, 3: 3108, 4: 3179},
        {0: 'GABA_B', 1: 'GABA_A', 2: 'NMDA', 3: 'AMPA'})

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param dtype: numeric type of the weight matrix. For Dynap-SE1, there are at most 64 connections between neurons so dtype defaults to np.uint8
        :type dtype: type, optional
        :param return_maps: return the index-to-UID, and syn-index-to-type maps or not, defaults to True
        :type return_maps: bool, optional
        :param decode_UID: decode the UID to get a key instead or not, defaults to False
        :type decode_UID: bool, optional
        :return: weight, index_UID_map
            weight: 3D weight matrix indicating number of synapses between neruons
            index_UID_map: a dictionary of the mapping between matrix indexes of the neurons and their global unique IDs (or keys)
            syn_type_map:a  dictionary of integer synapse type index keys and their names
        :rtype: Union[np.ndarray, Tuple[np.ndarray, Union[Dict[int, np.uint16], Dict[int, NeuronKey]] , Dict[int, str]]]
        """
        syn_dict = Router.synapses_from_config(config)
        return Router.weight_matrix(syn_dict, dtype, return_maps, decode_UID)
