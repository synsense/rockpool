"""
Analog device mismatch transformation(jax) implementation

Dynap-SE2 API support project
Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
11/01/2022
"""
from typing import Callable, Dict, Tuple

from jax import random as rand
from jax import numpy as jnp
from rockpool.utilities.jax_tree_utils import tree_map_with_rng

__all__ = ["dynapse_mismatch_generator"]


def dynapse_mismatch_generator(
    percent_deviation: float = 0.30, sigma_rule: float = 3.0
) -> Callable[
    [Tuple[Dict[str, jnp.DeviceArray], jnp.DeviceArray]], Dict[str, jnp.DeviceArray]
]:
    """
    dynapse_mismatch_generator returns a function which simulates the analog device mismatch effect.
    The function deviates the parameter values provided in statistical means.
    The calculation parameters should be based on statistical knowledge obtained from emprical observations.
    That is, if one observes up-to-30% mismatch between the expected current and the measured
    current actually flowing through the transistors: `percent` should be 0.30. Therefore, let's say

        mismatched current / actual current < %30 for 95 percent of the cases

    Using gaussian distribution and statistics, we could obtain this.
    In statistics, the 68-95-99.7 rule, also known as the empirical rule, is a shorthand used to remember
    the percentage of values that lie within an interval estimate in a normal distribution:
    68%, 95%, and 99.7% of the values lie within one, two, and three standard deviations of the mean, respectively.

    .. math ::
        Pr(\\mu -1\\sigma \\leq X\\leq \\mu +1\\sigma ) \\approx 68.27\\%
        Pr(\\mu -2\\sigma \\leq X\\leq \\mu +2\\sigma ) \\approx 95.45\\%
        Pr(\\mu -3\\sigma \\leq X\\leq \\mu +3\\sigma ) \\approx 99.73\\%

    So, to obtain 'mismatched current / actual current < %30 for 95 percent of the cases',
    the sigma should be half of the maximum deviation desired. That is the 30% percent of the theoretical current value.

    :param percent_deviation: the maximum deviation percentage from the theoretical value, defaults to 0.30
    :type percent_deviation: float, optional
    :param sigma_rule: The sigma rule to use. if 1.0, then 68.27% of the values will deviate less then `percent`, if 2.0, then 95.45% of the values will deviate less then `percent` etc., defaults to 3.0
    :type sigma_rule: float, optional
    :return: a function which takes a pytree and computes the mismatch amount accordingly
    :rtype: Callable[ [Tuple[Dict[str, jnp.DeviceArray], jnp.DeviceArray]], Dict[str, jnp.DeviceArray] ]
    """

    sigma_eff = jnp.array(percent_deviation / sigma_rule)

    def regenerate_mismatch(
        params: Dict[str, jnp.DeviceArray], rng_key: jnp.DeviceArray
    ) -> Dict[str, jnp.DeviceArray]:
        """
        regenerate_mismatch takes a parameter dictionary, flattens the tree and applies parameter mismatch to every leaf of the tree.

        :param params: parameter dictionary that is subject to mismatch effect
        :type params: Dict[str, jnp.DeviceArray]
        :param rng_key: the initial jax random number generator seed
        :type rng_key: jnp.DeviceArray
        :return: deviated parameters
        :rtype: Dict[str, jnp.DeviceArray]
        """

        def __atomic_mismatch(
            array: jnp.DeviceArray, rng_key: jnp.DeviceArray
        ) -> jnp.DeviceArray:
            """
            __atomic_mismatch _summary_

            :param array: single parameter to deviate
            :type array: jnp.DeviceArray
            :param rng_key: a single-use random number generator key
            :type rng_key: jnp.DeviceArray
            :return: the deviated parameter
            :rtype: jnp.DeviceArray
            """
            deviation = sigma_eff * rand.normal(rng_key, array.shape)
            return array + deviation * array

        new_params = tree_map_with_rng(params, __atomic_mismatch, rng_key)

        return new_params

    return regenerate_mismatch
