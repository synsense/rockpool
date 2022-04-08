"""
Dynap-SE2 samna alias. Replicate the samna data structures

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
08/04/2022
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np

from rockpool.devices.dynapse.lookup.scaling_factor import table as _params

se2_params = list(_params.keys())


@dataclass
class Dynapse2Parameter:
    param_name: str
    coarse_value: np.uint8
    fine_value: np.uint8
    type: Optional[str] = None
    se2_params = se2_params

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the Dynapse2Parameter object

        :raises ValueError: Invalid Parameter Name
        """

        if self.param_name not in se2_params:
            raise ValueError(f"{self.param_name} : Invalid parameter name!")

        self.type = self.param_name[-1]


if __name__ == "__main__":
    pass
