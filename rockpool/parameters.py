from typing import Callable, Iterable, Any

from collections import abc

import numpy as np

# -- Parameter classes
class ParameterBase:
    def __init__(
        self,
        data: Any = None,
        family: str = None,
        init_func: Callable[[Iterable], Any] = None,
        shape: Any = None,
    ):
        # - Assign attributes
        self.family = family
        self.data = data
        self.init_func = init_func
        self.shape = shape

        class_name = type(self).__name__

        # - Check that the initialisation function is callable
        if self.init_func is not None and not callable(self.init_func):
            raise ValueError(
                f"The `init_func` for a {class_name} must be a callable that accepts a shape tuple."
            )

        # - Get the shape from the data, if not provided explicitly
        if self.data is not None:
            if self.shape is not None and self.shape != np.shape(self.data):
                raise ValueError(
                    f"The shape provided for this {class_name} does not match the provided initialisation data."
                )

            if self.shape is None:
                self.shape = np.shape(self.data)

        # - Be generous if a scalar was provided instead of a tuple
        if self.shape is not None:
            if isinstance(self.shape, abc.Iterable):
                self.shape = tuple(self.shape)
            else:
                self.shape = (self.shape,)

        # - Initialise data, if not provided
        if self.data is None:
            if self.init_func is None:
                raise ValueError(
                    f"If concrete initialisation `data` is not provided for a {class_name} then `init_func` must be provided."
                )

            # - Call the `init_func`
            self.data = self.init_func(self.shape)
        else:
            # - If concrete initialisation data is provided, then override the `init_func`
            self.init_func = lambda _: data

    def __repr__(self):
        return f"{type(self).__name__}(data={self.data}, family={self.family}, init_func={self.init_func}, shape={self.shape})"


class Parameter(ParameterBase):
    pass


class State(ParameterBase):
    pass


class SimulationParameter(ParameterBase):
    pass
