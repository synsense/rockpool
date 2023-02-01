"""
Dynap-SE2 base simulated property class

* Non User Facing *
"""

from typing import Dict

from dataclasses import dataclass

import numpy as np


@dataclass
class DynapSimProperty:
    """
    DynapSimProperty stands for a commmon base class for all implementations
    """

    def get_full(self, size: int) -> Dict[str, np.ndarray]:
        """
        get_full creates a dictionary with respect to the object, with arrays of current values

        :param size: the lengths of the current arrays
        :type size: int
        :return: the object dictionary with current arrays given the size
        :rtype: Dict[str, np.ndarray]
        """
        __get__ = lambda name: np.full(size, self.__getattribute__(name))
        _dict = {name: __get__(name) for name in self.__dict__.keys()}
        return _dict
