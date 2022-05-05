"""
Dynap-SE1 samna alias. Replicate the samna data structures

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
08/04/2022
"""

from typing import Dict, Optional, List
from enum import Enum
from dataclasses import dataclass
import numpy as np

from rockpool.devices.dynapse.lookup import param_name as _params

se1_params = list(_params.se1_bias.keys())


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


@dataclass
class Dynapse1Synapse:
    listen_core_id: np.uint8
    listen_neuron_id: np.uint16
    syn_type: Dynapse1SynType


@dataclass
class Dynapse1Destination:
    target_chip_id: np.uint8

    # indicates if this sram is in use
    in_use: bool

    # id of the sram
    virtual_core_id: np.uint8

    # core mask
    core_mask: np.uint8
    sx: np.uint8
    sy: np.uint8
    dx: np.uint8
    dy: np.uint8


@dataclass
class Dynapse1Neuron:
    chip_id: np.uint8
    core_id: np.uint8
    neuron_id: np.uint16
    destinations: List[Dynapse1Destination]
    synapses: List[Dynapse1Synapse]


@dataclass
class Dynapse1ParameterGroup:
    chip_id: np.uint8
    core_id: np.uint8
    param_map: Dict[str, Dynapse1Parameter]

    def get_parameter_by_name(self, param_name: str) -> Dynapse1Parameter:
        return self.param_map[param_name]

    def get_linear_parameter(self, param_name: str) -> float:
        NotImplementedError("Not implemented for now, use samna")


@dataclass
class Dynapse1Core:
    chip_id: np.uint8
    core_id: np.uint8
    neurons: List[Dynapse1Neuron]
    parameter_group: Dynapse1ParameterGroup


@dataclass
class Dynapse1Chip:
    chip_id: np.uint8
    cores: List[Dynapse1Core]


@dataclass
class Dynapse1Configuration:
    chips: List[Dynapse1Chip]


if __name__ == "__main__":
    p = Dynapse1Parameter("PULSE_PWLK_P", 0, 0)
    print(p.param_name)
