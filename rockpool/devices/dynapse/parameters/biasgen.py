"""
Bias Generator module for Dynap-SE devices used in coarse-fine values to bias current generation

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
02/09/2021
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from rockpool.devices.dynapse.lookup import paramgen_se2, scale_factor_se2
from rockpool.devices.dynapse.samna_alias import Dynapse2Parameter

__all__ = ["BiasGenSE2"]


@dataclass
class BiasGenSE2:
    """
    BiasGenSE2 is a class encapsulating coarse and fine value to bias current conversion utilities.

    :param paramgen_table: parameter bias generator transistor current responses to given coarse and fine value tuples, defaults to paramgen_se2
    :type paramgen_table: Dict[str, Dict[int, List[float]]], optional
    :param scaling_factor_table: a mapping between bias parameter names and their scaling factors, defaults to scale_factor_se2
    :type scaling_factor_table: Dict[str, float], optional
    """

    paramgen_table: Dict[str, Dict[int, List[float]]] = field(
        default_factory=lambda: paramgen_se2
    )
    scaling_factor_table: Dict[str, float] = field(
        default_factory=lambda: scale_factor_se2
    )

    def digital_to_analog(
        self,
        coarse: np.uint8,
        fine: np.uint8,
        scaling_factor: Optional[float] = 1.0,
        type: Optional[str] = "N",
    ) -> np.float64:
        """
        digital_to_analog gets a coarse and a fine value tuple (digital) and returns the corresponding analog current

        :param coarse: integer coarse value :math:`C \\in [0,5]`
        :type coarse: np.uint8
        :param fine: integer fine value :math:`F \\in [0,255]`
        :type fine: np.uint8
        :param scaling_factor: the scaling factor to multiply the current, defaults to 1.0
        :type scaling_factor: Optional[float], optional
        :param type: the transistor type used to fetch from lookup table, defaults to "N"
        :type type: Optional[str], optional
        :return: the bias current in Amperes
        :rtype: np.float64
        """
        bias = self.paramgen_table[type][coarse][fine] * scaling_factor
        return bias

    def param_to_analog(self, param_name: str, param: Dynapse2Parameter) -> float:
        """
        param_to_bias convert samna `Dynapse2Parameter` object to a bias current to be used in the simulator

        :param param_name: the parameter name
        :type param_name: str
        :param param: samna parameter object storing a coarse and fine value tuple and the name of the device bias current
        :type param: Dynapse2Parameter
        :return: corrected bias current value by multiplying a scaling factor
        :rtype: float
        """
        scale = self.scaling_factor_table[param_name]
        bias = self.digital_to_analog(
            param.coarse_value, param.fine_value, scale, param.type
        )
        return bias

    def analog_to_digital(
        self, name: str, current_value: float
    ) -> Tuple[np.uint8, np.uint8]:
        """
        analog_to_digital converts a current value to a coarse and fine tuple representation using the `.__coarse_fine()` method.
        The scaling factor and the transistor type are found using the name of the parameter

        :param name: the name of the parameter
        :type name: str
        :param current_value: the bias current value
        :type current_value: float
        :return: the best matching coarse and fine value tuple
        :rtype: Tuple[np.uint8, np.uint8]
        """
        return self.__coarse_fine(
            current_value, self.scaling_factor_table[name], name[-1]
        )

    def __coarse_fine(
        self,
        current_value: float,
        scaling_factor: Optional[float] = 1.0,
        type: Optional[str] = "N",
    ) -> Tuple[np.uint8, np.uint8]:
        """
        __coarse_fine converts a current value to a coarse and fine tuple given a scale factor and transistor type

        :param current_value: the bias current value
        :type current_value: float
        :param scaling_factor: the parameter specific scale factor, defaults to 1.0
        :type scaling_factor: Optional[float], optional
        :param type: the type of the transistor, defaults to "N"
        :type type: Optional[str], optional
        :return: the best matching coarse and fine value tuple
        :rtype: Tuple[np.uint8, np.uint8]
        """

        # Get the candidates
        candidates = []
        for coarse, base in self.paramgen_table[type].items():
            fine = np.argmin(np.abs((np.array(base) * scaling_factor) - current_value))
            candidates.append((coarse, fine))

        # Find the best candidate
        coarse, fine = min(
            candidates,
            key=lambda key: abs(
                self.digital_to_analog(*key, scaling_factor, type) - current_value
            ),
        )
        return coarse, fine

    @property
    def n_coarse(self) -> int:
        """n_coarse returns the number of coarse bases"""
        assert self.paramgen_table["N"] == self.paramgen_table["P"]
        return len(self.paramgen_table["N"])
