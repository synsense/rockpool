"""
Module to provide useful types for Rockpool
"""

import numpy as np
from typing import Union, Any
from collections import abc

from rockpool.parameters import ParameterBase

P_float = Union[float, ParameterBase]
P_int = Union[int, ParameterBase]
P_str = Union[str, ParameterBase]
P_bool = Union[bool, ParameterBase]

P_ndarray = Union[np.ndarray, ParameterBase]

Tree = Union[abc.Iterable, abc.MutableMapping, abc.Mapping]
Leaf = Any
Value = Any
Node = Any

P_tree = Union[Tree, ParameterBase]

try:
    from torch import Tensor
except:
    Tensor = Any

P_tensor = Union[Tensor, ParameterBase]

FloatVector = Union[float, np.ndarray, Tensor]
IntVector = Union[int, np.ndarray, Tensor]

JaxRNGKey = Any
JaxTreeDef = Any
