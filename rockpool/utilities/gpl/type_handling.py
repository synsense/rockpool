"""
type_handling.py - Convenience functions for checking and converting object types
                   as well as objects for type type hints.
"""

from typing import Union, List, Tuple
import numpy as np

# - Configure exports
__all__ = ["to_scalar", "ArrayLike"]

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

### --- Convenience functions


def to_scalar(value, str_type: str = None):
    # - Check the value is a scalar
    assert np.size(value) == 1, "The value must be a scalar"

    if str_type is not None:
        return np.array(value).astype(str_type).item()
    else:
        return np.array(value).item()
