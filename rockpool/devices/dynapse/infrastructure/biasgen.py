"""
Bias Generator module for Dynap-SE devices used in coarse-fine values to bias current generation

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
02/09/2021
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from rockpool.devices.dynapse.lookup import paramgen, scaling_factor
from rockpool.devices.dynapse.samna_alias.dynapse1 import Dynapse1Parameter
from rockpool.devices.dynapse.samna_alias.dynapse2 import Dynapse2Parameter


class BiasGen:
    """
    BiasGen is a static class encapsulating coarse and fine value to bias current conversion utilities.

    :param paramgen_table: parameter bias generator transistor current responses to given coarse and fine value tuples, defaults to None
    :type paramgen_table: Optional[Dict[str, Dict[int, List[float]]]], optional
    :param scaling_factor_table: a mapping between bias parameter names and their scaling factors, defaults to None
    :type scaling_factor_table: optional, Dict[str, float]
    """

    def __init__(
        self,
        paramgen_table: Optional[Dict[str, Dict[int, List[float]]]] = None,
        scaling_factor_table: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        __init__ initializes the common SE1 and SE2 BiasGen module
        """
        self.paramgen_table = paramgen_table
        self.scaling_factor_table = scaling_factor_table

    def get_bias(
        self,
        coarse: np.uint8,
        fine: np.uint8,
        scaling_factor: Optional[float] = 1.0,
        type: Optional[str] = "N",
    ) -> np.float64:
        """
        get_bias overwrites the common get_bias() method and extend it's capabilities such as transistor type
        specific bias current calculation and lookup table usage

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

    def get_coarse_fine(
        self,
        current_value: float,
        scaling_factor: Optional[float] = 1.0,
        type: Optional[str] = "N",
    ) -> Tuple[np.uint8, np.uint8]:
        """
        get_coarse_fine _summary_

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
                self.get_bias(*key, scaling_factor, type) - current_value
            ),
        )
        return coarse, fine

    def param_to_bias(
        self, param_name: str, param: Union[Dynapse1Parameter, Dynapse2Parameter]
    ) -> float:
        """
        param_to_bias convert samna `Dynapse1Parameter` or `Dynapse2Parameter` object to a bias current to be used in the simulator

        :param param_name: the parameter name
        :type param_name: str
        :param param: samna parameter object storing a coarse and fine value tuple and the name of the device bias current
        :type param: Union[Dynapse1Parameter, Dynapse2Parameter]
        :return: corrected bias current value by multiplying a scaling factor
        :rtype: float
        """
        scale = self.scaling_factor_table[param_name]
        bias = self.get_bias(param.coarse_value, param.fine_value, scale, param.type)
        return bias

    @property
    def n_coarse(self) -> int:
        """n_coarse returns the number of coarse bases"""
        return len(self.paramgen_table["N"])


class BiasGenSE1(BiasGen):
    """
    BiasGenSE1 is Dynap-SE1 specific bias generator.
    """

    __doc__ += BiasGen.__doc__

    def __init__(self) -> None:
        """
        __init__ initialize the common module with full list of coarse base currents
        """
        super(BiasGenSE1, self).__init__(
            paramgen_table=paramgen.se1, scaling_factor_table=scaling_factor.se1
        )


class BiasGenSE2(BiasGen):
    """
    BiasGenSE2 is Dynap-SE2 specific bias generator
    """

    __doc__ += BiasGen.__doc__

    def __init__(self) -> None:
        """
        __init__ initialize the common module with shrinked list of coarse base currents
        """
        super(BiasGenSE2, self).__init__(
            paramgen_table=paramgen.se2, scaling_factor_table=scaling_factor.se2
        )


if __name__ == "__main__":
    # print(BiasGenSE1().get_coarse_fine(1e-9, 0.28))
    # print(BiasGenSE2().coarse_base))
    zero_idx = list(paramgen.se2.keys())[0]
    print(len(paramgen.se1[zero_idx]))
