"""
Module to provide useful types for Rockpool

See :ref:`/reference/params_types.ipynb` for more information on the available types.

"""

import numpy as np
from typing import Union, Any, Callable, Dict
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
""" A Parameter or a float """

P_int = Union[int, ParameterBase]
""" A Parameter or an int """

P_str = Union[str, ParameterBase]
""" A Parameter or a string """

P_bool = Union[bool, ParameterBase]
""" A Parameter or a boolean """

P_Callable = Union[Callable, ParameterBase]
""" A Parameter or a Callable """

P_ndarray = Union[np.ndarray, ParameterBase]
""" A Parameter or a numpy array """

Tree = Union[abc.Iterable, abc.MutableMapping, abc.Mapping]
""" A Python tree-like object """

Leaf = Any
""" A leaf node in a tree """

Value = Any
""" The value in a tree leaf node """

Node = Any
""" A node in a tree """

P_tree = Union[Tree, ParameterBase]
""" A Parameter or a Tree """

try:
    from torch import Tensor
except:
    Tensor = Any
    """ A torch tensor """

try:
    from jax.numpy import array as JaxArray
except:
    JaxArray = Any
    """ A Jax array """

P_tensor = Union[Tensor, ParameterBase]
""" A Parameter or a torch tensor """

FloatVector = Union[float, np.ndarray, Tensor, JaxArray]
""" A float scalar or a float vector """

IntVector = Union[int, np.ndarray, Tensor, JaxArray]
""" An int scalar or an int vector """

JaxRNGKey = Any
""" A Jax RNG key """

JaxTreeDef = Any
""" A Jax tree definition """

TreeDef = Dict
""" A Jax-like tree definition """


class DRCError(ValueError):
    """An Error class representing a Design-Rule Check violation"""

    pass


class DRCWarning(Warning, DRCError):
    """A Warning / Error class representing a Design-Rule Check warning"""

    pass
