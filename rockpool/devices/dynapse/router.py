"""
Dynap-SE1 router simulator. Create a weight matrix using SRAM and CAM content

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
13/09/2021
"""

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
            if ID == None:  # Full range
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
