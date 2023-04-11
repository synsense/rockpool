from typing import Tuple, Union, Dict, Any

import numpy as np
from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool.typehints import FloatVector


class IMUSim(Module):
    def __init__(
        self,
        shape: Union[Tuple[int], int],
        *args,
        **kwargs,
    ) -> None:
        # - Check shape argument
        if np.size(shape) == 1:
            shape = (1, np.array(shape).item())

        super().__init__(
            shape=shape, spiking_input=False, spiking_output=True, *args, **kwargs
        )

    def evolve(
        self, input_data: FloatVector, record: bool = False
    ) -> Tuple[FloatVector, Dict[str, FloatVector], Dict[str, FloatVector]]:
        return super().evolve(input_data, record)
