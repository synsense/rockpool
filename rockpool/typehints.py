"""
Module to provide useful types for Rockpool

See :ref:`/reference/params_types.ipynb` for more information on the available types.

"""

import numpy as np
from typing import Union, Any, Callable
from collections import abc

from rockpool.parameters import ParameterBase

__all__ = [
    "P_int",
    "P_str",
    "P_float",
    "P_bool",
    "P_tree",
    "P_tensor",
    "P_ndarray",
    "P_Callable",
    "Tree",
    "Leaf",
    "Value",
    "Node",
    "FloatVector",
    "IntVector",
    "JaxTreeDef",
    "JaxRNGKey",
]

P_float = Union[float, ParameterBase]
P_int = Union[int, ParameterBase]
P_str = Union[str, ParameterBase]
P_bool = Union[bool, ParameterBase]
P_Callable = Union[Callable, ParameterBase]

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

P_Callable = Union[Callable, ParameterBase]

FloatVector = Union[float, np.ndarray, Tensor]
IntVector = Union[int, np.ndarray, Tensor]

JaxRNGKey = Any
JaxTreeDef = Any
