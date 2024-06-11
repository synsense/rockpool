"""
Dynap-SE graph conversion package memory grid implementation

* Non User Facing *
"""

from typing import List, Tuple

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

__all__ = ["NPGrid"]


@dataclass
class NPGrid:
    """
    Addresses a subset of the numpy array that is to be used for reading or writing
    Implements error checking methods
    """

    row: Tuple[int]
    """a two dimensional row slice"""

    col: Tuple[int]
    """a two dimensional column slice"""

    def __post_init__(self):
        """
        __post_init__ checks the row and col limit validity after initialization

        :raises ValueError: row limits (min,max) represented wrong!
        :raises ValueError: col limits (min,max) represented wrong!
        :raises ValueError: row lower end be smaller than 0!
        :raises ValueError: col lower end cannot be smaller than 0!
        :raises ValueError: row indices should be in the non-decreasing order!
        :raises ValueError: col indices should be in the non-decreasing order!
        """
        if len(self.row) > 2:
            raise ValueError("row limits (min,max) represented wrong!")

        if len(self.col) > 2:
            raise ValueError("col limits (min,max) represented wrong!")

        if self.row[0] < 0:
            raise ValueError("row lower end be smaller than 0!")

        if self.col[0] < 0:
            raise ValueError("col lower end cannot be smaller than 0!")

        if self.row[1] < self.row[0]:
            raise ValueError("row indices should be in the non-decreasing order!")

        if self.col[1] < self.col[0]:
            raise ValueError("col indices should be in the non-decreasing order!")

    def place(
        self, destination: np.ndarray, source: np.ndarray, in_place: bool = True
    ) -> np.ndarray:
        """
        place places the smaller array to a bigger array given the grid constraints

        :param destination: the bigger destination array
        :type destination: np.ndarray
        :param source: the smaller source array
        :type source: np.ndarray
        :param in_place: change the destination object in place or deep copy if False, defaults to True
        :type in_place: bool, optional
        :raises ValueError: row indice is beyond the limits of the destination matrix!
        :raises ValueError: col indice is beyond the limits of the destination matrix!
        :raises ValueError: Source array shape is different then mask position!
        :raises ValueError: Target location includes non-zero elements!
        :return: modified destination array
        :rtype: np.ndarray
        """

        if self.row[1] > destination.shape[0]:
            raise ValueError(
                "row indice is beyond the limits of the destination matrix!"
            )
        if self.col[1] > destination.shape[1]:
            raise ValueError(
                "col indice is beyond the limits of the destination matrix!"
            )

        if (self.__diff(self.row), self.__diff(self.col)) != source.shape:
            raise ValueError("Source array shape is different then mask position!")

        if destination[self.row[0] : self.row[1], self.col[0] : self.col[1]].any():
            raise ValueError("Target location includes non-zero elements!")

        if in_place:
            destination[self.row[0] : self.row[1], self.col[0] : self.col[1]] = source
            return destination

        else:
            dest = deepcopy(destination)
            dest[self.row[0] : self.row[1], self.col[0] : self.col[1]] = source
            return dest

    def vplace(
        self, destination: np.ndarray, source: np.ndarray, in_place: bool = True
    ) -> np.ndarray:
        """
        vplace executes vertical placement on column limits

        :param destination: the bigger destination array
        :type destination: np.ndarray
        :param source: the source array
        :type source: np.ndarray
        :param in_place: change the destination object in place or deep copy if False, defaults to True
        :type in_place: bool, optional
        :raises ValueError: col indice is beyond the limits of the 1D destination matrix!
        :raises ValueError: 1D source array shape is different then mask position!
        :raises ValueError: 1D target location includes non-zero elements!
        :return: modified destination array
        :rtype: np.ndarray
        """
        """ """

        if self.col[1] > destination.shape[0]:
            raise ValueError(
                "col indice is beyond the limits of the 1D destination matrix!"
            )

        if (self.__diff(self.col),) != source.shape:
            raise ValueError("1D source array shape is different then mask position!")

        if destination[self.col[0] : self.col[1]].any():
            raise ValueError("1D target location includes non-zero elements!")

        if in_place:
            destination[self.col[0] : self.col[1]] = source
            return destination

        else:
            dest = deepcopy(destination)
            dest[self.col[0] : self.col[1]] = source
            return dest

    @staticmethod
    def __diff(tup: Tuple[int]) -> int:
        """__diff returns the absolute difference between elements of the tuple"""
        return abs(tup[0] - tup[1])
