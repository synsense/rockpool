"""
Dynap-SE2 samna alias definitions. Ensures consistency over samna and rockpool

* Non User Facing *
"""

from enum import Enum

__all__ = ["ParameterType", "DvsMode", "Dendrite"]


class ParameterType(int, Enum):
    """
    ParameterType implements the parameter type enumerator to descriminate P type and N type transistor paramters
    """

    p: int = 0
    n: int = 1


class DvsMode(int, Enum):
    """
    DvsMode implements the DVS generation enumerator to describe the model DVS128, Davis240c, or Davis346
    """

    Dvs128: int = 0
    Davis240c: int = 2
    Davis346: int = 4


class Dendrite(int, Enum):
    """
    Dendrite implements the dynapse dendrite types enumerator
    """

    none: int = 0
    ampa: int = 1024
    gaba: int = 512
    nmda: int = 256
    shunt: int = 128
