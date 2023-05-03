from functools import wraps
from typing import List

import numpy as np

__all__ = ["type_check"]


def type_check(func):
    """Type-check decorator for IMU python simulation to make sure that all the input data are of type `python.object`.
    This assures that the hardware and software will behave the same for all register sizes.

    Args:
        func (Callable): the function to be decorated.
    """

    def verify(input: List[np.ndarray]) -> None:
        if isinstance(input, list):
            if len(input) != 0:
                for el in input:
                    type_check(el)

        elif isinstance(input, np.ndarray):
            if input.dtype != object or type(input.ravel()[0]) != int:
                raise ValueError(
                    f"The elements of the following variable are not of type `python.object` integer. This may cause mismatch between hardware and python implementation."
                    + f"problem with the follpowing variable:\n{input}\n"
                    + f"To solve the problem make sure that all the arrays have `dtype=object`. You can use `Quantizer` class in `quantizer.py` module."
                )

    @wraps(func)
    def inner_func(*args, **kwargs):
        for arg in args:
            verify(arg)

        for key in kwargs:
            verify(kwargs[key])

        return func(*args, **kwargs)

    return inner_func
