"""
Bias Generator module for Dynap-SE devices used in coarse-fine values to bias current generation

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
02/09/2021
"""

import logging

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rockpool.devices.dynapse.base import ArrayLike
from rockpool.devices.dynapse.lookup import (
    paramgen,
    scaling_factor,
    scaling_factor_se1,
)
from rockpool.devices.dynapse.samna_alias.dynapse1 import Dynapse1Parameter
from rockpool.devices.dynapse.samna_alias.dynapse2 import Dynapse2Parameter

COARSE_BASE_EXT = [
    1.5e-11,  # I0
    1.05e-10,  # I1
    8.2e-10,  # I2
    6.5e-09,  # I3
    5e-08,  # I4
    4e-07,  # I5
    3.2e-06,  # I6
    2.4e-05,  # I7
]


class BiasGen:
    """
    BiasGen is a static class encapsulating coarse and fine value to bias current and linear bias value conversion utilities.

    :param coarse_base: base currents depending on given coarse index
    :type coarse_base: ArrayLike
    :param scaling_factor_table: a mapping between bias parameter names and their scaling factors, defaults to None
    :type scaling_factor_table: optional, Dict[str, float]
    :param f_linear: scaling factor to convert bias currents to linear bias values
    :type f_linear: float, optional
    :param fine_max: the max number that a fine value can get
    :type fine_max: int, optional
    """

    def __init__(
        self,
        coarse_base: ArrayLike,
        scaling_factor_table: Optional[Dict[str, float]] = None,
        f_linear: Optional[float] = np.float32(1e14),
        fine_max: Optional[int] = np.uint8(255),
    ) -> None:
        """
        __init__ initializes the common SE1 and SE2 BiasGen module
        """

        self.coarse_base = np.array(coarse_base, dtype=np.float32)
        self.scaling_factor_table = scaling_factor_table
        self.f_linear = f_linear
        self.fine_max = fine_max

    def _check_coarse_fine_limits(self, coarse: np.uint8, fine: np.uint8) -> None:
        """
        _check_coarse_fine_limits checks if coarse and fine values are within the limits or not.

        :param coarse: integer coarse value :math:`C \\in [0,7]` for SE1 or :math:`C \\in [0,5]` for SE2
        :type coarse: np.uint8
        :param fine: integer fine value :math:`F \\in [0,255]`
        :type fine: np.uint8
        :raises IndexError: The coarse value is out of limits
        :raises IndexError: The fine value is out of limits
        """
        if coarse < 0 or coarse > len(self.coarse_base):
            raise IndexError(f"Coarse Value: {coarse} is out of limits! [0,5]")
        if fine < 0 or fine > self.fine_max:
            raise IndexError(f"Fine Value: {fine} is out of limits! [0,255]")

    def get_bias(
        self,
        coarse: np.uint8,
        fine: np.uint8,
        scaling_factor: Optional[float] = 1.0,
    ) -> np.float64:
        """
        get_bias obtains a bias current value using the theoretical conversion.
        The very large scale integration(VLSI) neurons on DYNAP-SE are controlled by configurable current
        sources called “biases”. For each bias, there is an integer coarse value :math:`C \\in [0,7]` and
        an integer fine value :math:`F \\in [0,255]`, which together determine the amplitude of the current in Amperes.

        :param coarse: integer coarse value :math:`C \\in [0,7]` for SE1 or :math:`C \\in [0,5]` for SE2
        :type coarse: np.uint8
        :param fine: integer fine value :math:`F \\in [0,255]`
        :type fine: np.uint8
        :param scaling_factor: the scaling factor to correct the bias value calculation with multiplication, defaults to 1
        :type scaling_factor: float, optional
        :return: the bias current in Amperes
        :rtype: np.float64
        """
        self._check_coarse_fine_limits(coarse, fine)
        max_current = self.coarse_base[coarse]
        base_current = np.divide(max_current, self.fine_max, dtype=np.float64)
        bias = np.multiply(fine, base_current, dtype=np.float64)
        corrected = bias * scaling_factor
        return corrected

    def get_linear(self, coarse: np.uint8, fine: np.uint8) -> np.float64:
        """
        get_linear calculates the bias current then scales it to the linear bias value
        For more info, look at `BiasGen.bias_to_linear()`.

        :param coarse: integer coarse value :math:`C \\in [0,7]` for SE1 or :math:`C \\in [0,5]` for SE2
        :type coarse: np.uint8
        :param fine: integer fine value :math:`F \\in [0,255]`
        :type fine: np.uint8
        :return: the reciprocal linear bias value obtained from bias current
        :rtype: np.float32
        """
        bias = self.get_bias(coarse, fine)
        linear = self.bias_to_linear(bias)
        return linear

    def bias_to_linear(self, bias: np.float64) -> np.float32:
        """
        bias_to_linear scales and round the bias value using the linearization factor `f_linear`
        It allows expressing all the possible bias values with integers.

        :param bias: the bias current in Amperes
        :type bias: np.float64
        :return: the reciprocal linear bias value obtained from bias current
        :rtype: np.float32
        """
        linear = np.multiply(bias, self.f_linear, dtype=np.float32)
        linear = np.around(linear, 0)
        return linear

    def get_coarse_fine(
        self, linear: np.float32, coarse_smallest: bool = True, exact: bool = True
    ) -> Tuple[np.uint8, np.uint8]:
        """
        get_coarse_fine gives a coarse/fine tuple given a linear bias value

        :param linear: the linear bias value
        :type linear: np.float32
        :param coarse_smallest: Choose the smallest coarse value possible. In this case, fine value would be slightly higher. If False, the function returns the biggest possible coarse value. The smaller the coarse value is the better the resolution is! defaults to True
        :type coarse_smallest: bool, optional
        :param exact: If true, the function returns a corse fine tuple in the case that the exact linear value can be obtained using the ``BiasGen.get_linear()`` function else the function returns None. If false, the function returns the closest possible coarse and fine tuple, defaults to True
        :type exact: bool, optional
        :return: coarse and fine value tuple
        :rtype: Tuple[np.uint8, np.uint8]
        """

        # If linear bias value is 0, no need to calculate!
        if linear == 0:
            return np.uint8(0), np.uint8(0)

        if coarse_smallest:
            couple_idx = 0

        else:  # coarse_biggest
            couple_idx = -1

        def propose_coarse_candidates() -> np.ndarray:
            """
            propose_coarse_candidates gives coarse base currents which can possibly create the linear value desired.
            Multiple coarse base current might generate the same current given a proper fine value!

            :return: an array of coarse value candidate tuples
            :rtype: np.ndarray
            """

            max_linear = np.multiply(self.coarse_base, self.f_linear)
            min_linear = np.divide(max_linear, self.fine_max + 1, dtype=np.float32)

            upper_bound = max_linear >= linear
            lower_bound = min_linear <= linear
            condition = np.logical_and(upper_bound, lower_bound)

            candidates = np.where(condition)[0]
            return candidates.astype(np.uint8)

        def propose_coarse_fine_tuple(coarse: np.uint8) -> Tuple[np.uint8, np.uint8]:
            """
            propose_coarse_fine_tuple finds a fine value which creates the linear bias(exactly or very close to!) desired given the coarse value.

            :param coarse: the coarse index value
            :type coarse: np.uint8
            :return: candidate coarse fine tuple
            :rtype: Tuple[np.uint8, np.uint8]
            """

            fine = np.divide(
                linear * self.fine_max,
                self.coarse_base[coarse] * self.f_linear,
                dtype=np.float32,
            )
            fine = np.uint8(np.around(fine, 0))
            return coarse, fine

        ## -- Coarse Fine Search -- ##

        candidates = propose_coarse_candidates()
        couples = []

        for coarse in candidates:
            coarse, fine = propose_coarse_fine_tuple(coarse)

            if exact:  # Exact linear value should be obtained via the coarse fine tuple
                if linear == self.get_linear(coarse, fine):
                    couples.append((coarse, fine))
                else:
                    continue
            else:
                couples.append((coarse, fine))

        if couples:
            return couples[couple_idx]
        else:
            logging.warning(
                f"Desired linear value is out of limits! {linear}. Max possible returned!"
            )
            return np.uint8(len(self.coarse_base)), np.uint8(self.fine_max)

    def bias_to_coarse_fine(
        self, bias: np.float64, coarse_smallest: bool = True  # Better resolution
    ) -> Tuple[np.uint8, np.uint8]:
        """
        bias_to_coarse_fine find a coarse and fine tuple which represents the desired bias current the best

        :param bias: The bias current in Amperes
        :type bias: np.float64
        :param coarse_smallest: Choose the smallest coarse value possible. In this case, fine value would be slightly higher. If False, the function returns the biggest possible coarse value. The smaller the coarse value is the better the resolution is! defaults to True
        :type coarse_smallest: bool, optional
        :return: coarse and fine value tuple
        :rtype: Tuple[np.uint8, np.uint8]
        """
        linear = self.bias_to_linear(bias)
        coarse, fine = self.get_coarse_fine(linear, coarse_smallest, exact=False)
        return coarse, fine

    def param_to_bias(
        self,
        param_name: str,
        param: Union[Dynapse1Parameter, Dynapse2Parameter],
        scale_lookup: bool = True,
        *args,
        **kwargs,
    ) -> float:
        """
        param_to_bias convert samna `Dynapse1Parameter` or `Dynapse2Parameter` object to a bias current to be used in the simulator

        :param param_name: the parameter name
        :type param_name: str
        :param param: samna parameter object storing a coarse and fine value tuple and the name of the device bias current
        :type param: Union[Dynapse1Parameter, Dynapse2Parameter]
        :param scale_lookup: use the the scaling factor table to scale and correct the bias_current defaults to True
        :type scale_lookup: bool, optional
        :return: corrected bias current value by multiplying a scaling factor
        :rtype: float
        """
        if scale_lookup and self.scaling_factor_table is None:
            raise ValueError("Scale factor lookup table is missing!")
        scale = self.scaling_factor_table[param_name] if scale_lookup else 1.0
        bias = self.get_bias(
            param.coarse_value,
            param.fine_value,
            scale,
            *args,
            **kwargs,
        )
        return bias

    def get_lookup_table(self) -> np.ndarray:
        """
        get_lookup_table provides a lookup table with always increasing linear biases.
        Please note that due to the floating point precision issues, the linear bias values are not
        the same as the C++ samna implementation. However the variation is small. In comparison:
            Unmatched elements: 296 / 5436 (5.45%)
            Max absolute difference: 256.
            Max relative difference: 5.16192974e-06

        The original lookup table can be downloaded from:
            https://gitlab.com/neuroinf/ctxctl_contrib/-/blob/samna-dynapse1-NI-demo/linear_fine_coarse_bias_map.npy

        :return: an array of [coarse, fine, linear bias]
        :rtype: np.ndarray
        """
        lookup = []
        for c in range(len(self.coarse_base)):
            for f in range(self.fine_max + 1):
                linear = self.get_linear(c, f)
                if lookup:
                    if linear > lookup[-1][2] * 1.00001:
                        lookup.append([c, f, linear])
                else:
                    lookup.append([c, f, linear])

        lookup = np.array(lookup, dtype=np.float32)
        return lookup


class BiasGenSE1(BiasGen):
    """
    BiasGenSE1 is Dynap-SE1 specific bias generator. It holds the corrections factors of
    25 different biases and implements a parameter to bias conversion method.
    """

    __doc__ += BiasGen.__doc__

    def __init__(
        self,
        coarse_base: ArrayLike = COARSE_BASE_EXT,
        scaling_factor_table: Dict[str, float] = scaling_factor_se1.table,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ initialize the common module with full list of coarse base currents
        """
        super(BiasGenSE1, self).__init__(
            coarse_base, scaling_factor_table, *args, **kwargs
        )


class BiasGenSE2(BiasGen):
    """
    BiasGenSE2 is Dynap-SE2 specific bias generator. One can convert a coarse, fine value to
    bias current given the transistor type and scaling factor or one can define a bias parameter
    name to use the right transistor type and the scaling factor for conversion
    """

    __doc__ += BiasGen.__doc__

    def __init__(
        self,
        coarse_base: ArrayLike = COARSE_BASE_EXT[1:-1],
        scaling_factor_table: Dict[str, float] = scaling_factor.table,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ initialize the common module with shrinked list of coarse base currents
        """
        self.paramgen_table = paramgen.table
        super(BiasGenSE2, self).__init__(
            coarse_base, scaling_factor_table, *args, **kwargs
        )

    def get_bias(
        self,
        coarse: np.uint8,
        fine: np.uint8,
        scaling_factor: Optional[float] = 1.0,
        type: Optional[str] = "N",
        lookup: Optional[bool] = True,
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
        :param lookup: use lookup tables in bias calculation or not, defaults to True
        :type lookup: Optional[bool], optional
        :return: the bias current in Amperes
        :rtype: np.float64
        """
        self._check_coarse_fine_limits(coarse, fine)
        if lookup:
            bias = self.paramgen_table[f"{type}{coarse}"][fine] * scaling_factor
        else:
            bias = super().get_bias(coarse, fine, scaling_factor)

        return bias

    def param_to_bias(
        self,
        param_name: str,
        param: Dynapse2Parameter,
        scale_lookup: bool = True,
        bias_lookup: bool = True,
    ) -> float:
        """
        param_to_bias convert samna `Dynapse2Parameter` object to a bias current to be used in the simulator

        :param param_name: the parameter name
        :type param_name: str
        :param param: Dynapse2Parameter holding a coarse and fine value tuple and the name of the device bias current
        :type param: Dynapse2Parameter
        :param scale_lookup: use the the scaling factor table to scale and correct the bias_current defaults to True
        :type scale_lookup: bool, optional
        :param bias_lookup: use the transistor specific parameter generator lookup table or not. If not, then theoretical calculations are used. defaults to True
        :type bias_lookup: bool, optional
        :return: corrected bias value by multiplying a scaling factor
        :rtype: float
        """
        bias = super().param_to_bias(
            param_name, param, scale_lookup, param.type, bias_lookup
        )
        return bias
