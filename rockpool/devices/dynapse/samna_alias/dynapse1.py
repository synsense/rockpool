"""
Dynap-SE1 samna alias. Replicate the samna data structures

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
08/04/2022
"""
from typing import Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np

from rockpool.devices.dynapse.lookup import param_name as _params

se1_params = list(_params.se1.keys())


class Dynapse1SynType(Enum):
    NMDA = 2
    AMPA = 3
    GABA_B = 0
    GABA_A = 1


@dataclass
class Dynapse1Parameter:
    param_name: str
    coarse_value: np.uint8
    fine_value: np.uint8
    type: Optional[str] = None
    se1_params = se1_params

    def __post_init__(self) -> None:
        """
        __post_init__ runs after __init__ and initializes the Dynapse1Parameter object

        :raises ValueError: Invalid Parameter Name
        """

        if self.param_name not in se1_params:
            raise ValueError(f"{self.param_name} : Invalid parameter name!")

        self.type = self.param_name[-1]


if __name__ == "__main__":
    p = Dynapse1Parameter("PULSE_PWLK_P", 0, 0)
    print(p.param_name)
