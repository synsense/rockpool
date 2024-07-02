"""
Analog device mismatch transformation (jax) implementation
"""

from typing import Any, Callable, Dict, Tuple
from rockpool.nn.modules.jax import JaxModule

import jax
from jax import random as rand
from jax import numpy as jnp
from rockpool.utilities.jax_tree_utils import tree_map_select_with_rng

__all__ = ["mismatch_generator"]


def mismatch_generator(
    prototype: Dict[str, bool], percent_deviation: float = 0.30, sigma_rule: float = 3.0
) -> Callable[[Tuple[Dict[str, jax.Array], jax.Array]], Dict[str, jax.Array]]:
    """ 
    mismatch_generator returns a function which simulates the analog device mismatch effect.
    The function deviates the parameter values provided in statistical means.
    The calculation parameters should be based on statistical knowledge obtained from emprical observations.
    That is, if one observes up-to-30% mismatch between the expected current and the measured
    current actually flowing through the transistors: ``percent`` should be 0.30. Therefore, let's say

        mismatched current / actual current < %30 for 95 percent of the cases

    Using gaussian distribution and statistics, we could obtain this.
    In statistics, the 68-95-99.7 rule, also known as the empirical rule, is a shorthand used to remember
    the percentage of values that lie within an interval estimate in a normal distribution:
    68%, 95%, and 99.7% of the values lie within one, two, and three standard deviations of the mean, respectively.

    .. math ::
        Pr(\\mu -1\\sigma \\leq X\\leq \\mu +1\\sigma ) \\approx 68.27\\\\
        Pr(\\mu -2\\sigma \\leq X\\leq \\mu +2\\sigma ) \\approx 95.45\\\\
        Pr(\\mu -3\\sigma \\leq X\\leq \\mu +3\\sigma ) \\approx 99.73\\\\

    So, to obtain 'mismatched current / actual current < %30 for 95 percent of the cases',
    the sigma should be half of the maximum deviation desired. That is the 30% percent of the theoretical current value.


    :param prototype: the mismatch prototype. See :py:func:`devices.dynapse.frozen_mismatch_prototype`, and :py:func:`devices.dynapse.dynamic_mismatch_prototype`
    :type prototype: Dict[str, bool]
    :param percent_deviation: the maximum deviation percentage from the theoretical value, defaults to 0.30
    :type percent_deviation: float, optional
    :param sigma_rule: The sigma rule to use. if 1.0, then 68.27% of the values will deviate less then ``percent``, if 2.0, then 95.45% of the values will deviate less then ``percent`` etc., defaults to 3.0
    :type sigma_rule: float, optional
    :return: a function which takes a pytree and computes the mismatch amount accordingly
    :rtype: Callable[ [Tuple[Dict[str, jax.Array], jax.Array]], Dict[str, jax.Array] ]
    """

    sigma_eff = jnp.array(percent_deviation / sigma_rule)

    def regenerate_mismatch(mod: JaxModule, rng_key: jax.Array) -> Dict[str, jax.Array]:
        """
        regenerate_mismatch takes a parameter dictionary, flattens the tree and applies parameter mismatch to every leaf of the tree.

        :param params: parameter dictionary that is subject to mismatch effect
        :type params: Dict[str, jax.Array]
        :param rng_key: the initial jax random number generator seed
        :type rng_key: jax.Array
        :return: deviated parameters
        :rtype: Dict[str, jax.Array]
        """

        def __map_fun(array: jax.Array, rng_key: jax.Array) -> jax.Array:
            """
            __map_fun is the mapping functions that applies the deviation to all leaves of the tree

            :param array: single parameter to deviate
            :type array: jax.Array
            :param rng_key: a single-use random number generator key
            :type rng_key: jax.Array
            :return: the deviated parameter
            :rtype: jax.Array
            """
            deviation = sigma_eff * rand.normal(rng_key, array.shape)
            return jnp.array(array + deviation * array, dtype=jnp.float32)

        params = module_registery(mod)
        new_params = tree_map_select_with_rng(params, prototype, __map_fun, rng_key)

        return new_params

    return regenerate_mismatch


def module_registery(module: JaxModule) -> Dict[str, Any]:
    """
    module_registery traces all the nested module and registered parameters of the JaxModule base given and returns a tree,
    whose leaves includes only the `parameters.SimulationParameters` and `parameters.Parameters`

    [] TODO : embed this into JaxModule base as a class method

    :param module: a self-standing simulation module or a combinator object
    :type module: JaxModule
    :return: a nested parameter tree
    :rtype: Dict[str, Any]
    """
    __attrs, __modules = module._get_attribute_registry()
    __dict = {
        k: jnp.array(getattr(module, k), dtype=jnp.float32)
        for k, v in __attrs.items()
        if (v[1] == "SimulationParameter") or (v[1] == "Parameter")
    }

    # - Append sub-module attributes as nested dictionaries
    __sub_attrs = {}
    for k, m in __modules.items():
        # [0] -> module , [1] -> name
        __sub_attrs[k] = module_registery(m[0])

    __dict.update(__sub_attrs)
    return __dict
