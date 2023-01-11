"""
Bias Generator module for Dynap-SE2 coarse and fine value to bias current conversion utilities.

* Non User Facing *
"""

from typing import Optional, Tuple

import numpy as np

from rockpool.devices.dynapse.lookup import paramgen_se2, scale_factor_se2
from rockpool.devices.dynapse.samna_alias import Dynapse2Parameter

__all__ = [
    "digital_to_analog",
    "analog_to_digital",
    "param_to_analog",
    "analog_to_param",
]


def digital_to_analog(
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
    bias = paramgen_se2[type][coarse][fine] * scaling_factor
    return bias


def param_to_analog(param_name: str, param: Dynapse2Parameter) -> float:
    """
    param_to_analog convert samna `Dynapse2Parameter` object to a bias current to be used in the simulator

    :param param_name: the parameter name
    :type param_name: str
    :param param: samna parameter object storing a coarse and fine value tuple and the name of the device bias current
    :type param: Dynapse2Parameter
    :return: corrected bias current value by multiplying a scaling factor
    :rtype: float
    """
    scale = scale_factor_se2[param_name]
    bias = digital_to_analog(param.coarse_value, param.fine_value, scale, param.type)
    return bias


def analog_to_param(name: str, current_value: float) -> Tuple[np.uint8, np.uint8]:
    """
    analog_to_param converts a current value to a coarse and fine tuple representation using the `.analog_to_digital()` method.
    The scaling factor and the transistor type are found using the name of the parameter

    :param name: the name of the parameter
    :type name: str
    :param current_value: the bias current value
    :type current_value: float
    :return: the best matching coarse and fine value tuple
    :rtype: Tuple[np.uint8, np.uint8]
    """
    return analog_to_digital(current_value, scale_factor_se2[name], name[-1])


def analog_to_digital(
    current_value: float,
    scaling_factor: Optional[float] = 1.0,
    type: Optional[str] = "N",
) -> Tuple[np.uint8, np.uint8]:
    """
    analog_to_digital converts a current value to a coarse and fine tuple given a scale factor and transistor type

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
    for coarse, base in paramgen_se2[type].items():
        fine = np.argmin(np.abs((np.array(base) * scaling_factor) - current_value))
        candidates.append((coarse, fine))

    # Find the best candidate
    coarse, fine = min(
        candidates,
        key=lambda key: abs(
            digital_to_analog(*key, scaling_factor, type) - current_value
        ),
    )
    return coarse, fine
