"""
Dynap-SE1 router simulator. Create a weight matrix using SRAM and CAM content

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
13/09/2021
"""

from samna.dynapse1 import (
    Dynapse1Neuron,
    Dynapse1Synapse,
    Dynapse1Destination,
)

from typing import (
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
                    ID = [int(ID)]
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
    def receiving_connections(
        neuron: Dynapse1Neuron, synapse: Dynapse1Synapse
    ) -> List[NeuronConnection]:
        """
        receiving_connections produce a list of spike receiving connections given a neuron and synapse object
        From the device's point of view, in each CAM(Content Addressable Memory) cell
        the neuron can be set to listen to (i.e. receive events from) one other neuron.
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
        :return: List of unique IDs of all neuron connection pairs in the (pre, post) order.
        :rtype: List[NeuronConnection]
        """

        # Pre-synaptic neurons to listen across 4 chips
        pre_list = Router.get_UID_combination(
            chipID=None,
            coreID=synapse.listen_core_id,
            neuronID=synapse.listen_neuron_id,
        )

        # Post-synaptic neuron
        post = Router.get_UID(neuron.chip_id, neuron.core_id, neuron.neuron_id)

        connections = Router.connect_pre_post(pre_list, post)
        return connections

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

    def connect_pre_post(
        preUID: Union[np.uint16, ArrayLike], postUID: Union[np.uint16, ArrayLike]
    ) -> List[NeuronConnection]:
        """
        connect_pre_post produce a list of connections between neurons like List[(preUID, postUID)].
        The pre and post can be given as a list or a single ID. If a single ID is given, a repeating
        UID list having the same shape with the others is created.

        :param preUID: a unique pre-synaptic neuron ID or a list of IDs
        :type preUID: Union[np.uint16, ArrayLike]
        :param postUID: a unique post-synaptic neuron ID or a list of IDs
        :type postUID: Union[np.uint16, ArrayLike]
        :raises ValueError: When the size of the preUID and postUID arrays are not the same
        :raises TypeError: preUID or postUID is not ArraLike or a type which can casted to np.uint16
        :return: connections between neruons in the form of tuple(preUID, postUID)
        :rtype: List[NeuronConnection]
        """

        def to_list(uid: np.uint16) -> List[np.uint16]:
            """
            to_list creates a repeating list given a single element

            :param uid: a single unique neuron id
            :type uid: np.uint16
            :raises TypeError: If the neuron id cannot be casted to uint16.
            :return: a repeating list of given uid with the shape of the second uid list provided to the upper level function.
            :rtype: List[np.uint16]
            """
            try:
                uid = np.uint16(uid)
            except:
                raise TypeError(f"neuron ID should be int or uint!")
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

        connections = list(zip(preUID, postUID))
        return connections
