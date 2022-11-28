"""
Device mismatch implementation for DynapSE modules

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
11/01/2022
[] TODO : Jax module
[] TODO : Check again with JIT
[] TODO : Register the attributes as SimulationParameters
"""
from typing import Tuple, Union

from jax import random as rand
from jax import numpy as jnp

from rockpool.utilities.property_arrays import ArrayLike

class MismatchDevice:
    """
    MismatchDevice applies random numerical deviation to the data-frames provided to simulate the effect of analog device mismatch

    :param rng_key: The initial Jax RNG seed. new random number generation keys are produced as many as the parameters provided
    :type rng_key: jnp.DeviceArray
    :param percent: A list including a number of elements equal to the number of parameters provided describing the maximum deviation percentage from the theoretical value. If a float is passed instead of a list, then all parameters share the same maximum deviation percentage, defaults to 0.05
    :type percent: Union[float, ArrayLike], optional
    :param sigma_rule: A list including a number of elements equal to the number of parameters provided describing the sigma rule to use. If 1.0, then 68.27% of the values will deviate less then `percent`, if 2.0, then 95.45% of the values will deviate less then `percent` etc. If a float is passed instead of a list, then all parameters share the same sigma rule, defaults to 2.0
    :type sigma_rule: Union[int, float, ArrayLike], optional
    :raises ValueError: Percent array should have the same length with the number of parameters!
    :raises ValueError: Sigma rule array should have the same length with the number of parameters!

    :Instance Variables:

    :ivar _attr_list: A list of names of attributes whose mismatched version are stored
    :type _attr_list: List[str]
    :ivar rng_key: the random number generation keys of attributes
    :type rng_key: Dict[str, jnp.DeviceArray]
    """

    def __init__(
        self,
        rng_key,
        percent: Union[float, ArrayLike] = 0.05,
        sigma_rule: Union[int, float, ArrayLike] = 2.0,
        **kwargs,
    ) -> None:
        """
        __init__ Initialize ``MismatchDevice`` module. Parameters are explained in the class docstring.
        """

        # Controls
        if isinstance(percent, (float)):
            percent = [percent] * len(kwargs)

        if isinstance(sigma_rule, (int, float)):
            sigma_rule = [sigma_rule] * len(kwargs)

        if len(percent) != len(kwargs):
            raise ValueError(
                "Percent array should have the same length with the number of parameters!"
            )
        if len(sigma_rule) != len(sigma_rule):
            raise ValueError(
                "Sigma rule array should have the same length with the number of parameters!"
            )

        self._attr_list = list(kwargs.keys())
        subkeys = rand.split(rng_key, len(kwargs.keys()))
        self.rng_key = dict(zip(kwargs.keys(), subkeys))

        # Calculate and store both the mismatch ratio and the mismatched current
        for (key, value), p, s_rule in zip(kwargs.items(), percent, sigma_rule):
            mm_param = self.mismatch_ratio(self.rng_key[key], value.shape, p, s_rule)
            param_eff = self.param_eff(value, mm_param)
            self.__setattr__(f"mm_{key}", mm_param)
            self.__setattr__(f"{key}", param_eff)

    @staticmethod
    def mismatch_ratio(
        rng_key: jnp.DeviceArray,
        shape: Tuple[int, int],
        percent: float,
        sigma_rule: float,
    ) -> jnp.DeviceArray:
        """
        mismatch_ratio calculates the percent current difference resulted from the analog device mismatch.
        The calculation parameters should be based on statistical knowledge obtained from emprical observations.
        That is, if one observes up-to-30% mismatch between the expected current and the measured
        current actually flowing through the transistors: `percent` should be 0.30. Therefore, let's say

            mismatched current / actual current < %30 for 95 percenparam_efft of the cases

        Using gaussian distribution and statistics, we could obtain this.
        In statistics, the 68–95–99.7 rule, also known as the empirical rule, is a shorthand used to remember
        the percentage of values that lie within an interval estimate in a normal distribution:
        68%, 95%, and 99.7% of the values lie within one, two, and three standard deviations of the mean, respectively.

        .. math ::
            Pr(\\mu -1\\sigma \\leq X\\leq \\mu +1\\sigma ) \\approx 68.27\\%
            Pr(\\mu -2\\sigma \\leq X\\leq \\mu +2\\sigma ) \\approx 95.45\\%
            Pr(\\mu -3\\sigma \\leq X\\leq \\mu +3\\sigma ) \\approx 99.73\\%

        So, to obtain 'mismatched current / actual current < %30 for 95 percent of the cases',
        the sigma should be half of the maximum deviation desired. That is the 30% percent of the theoretical current value.

        :param param: the current parameter array to deviate
        :type param: jnp.DeviceArray
        :param rng_key: The Jax RNG seed to use in random number generation
        :type rng_key: jnp.DeviceArray
        :param percent: the maximum deviation percentage from the theoretical value, defaults to 0.05
        :type percent: float
        :param sigma_rule: The sigma rule to use. if 1.0, then 68.27% of the values will deviate less then `percent`, if 2.0, then 95.45% of the values will deviate less then `percent` etc., defaults to 2.0
        :type sigma_rule: float
        :return: the percent mismatch ratio array. To obtain the mismatched current do
            `param_effective = param + ratio * param`
        :rtype: jnp.DeviceArray
        """
        sigma = jnp.array(percent / sigma_rule)
        ratio = sigma * rand.normal(rng_key, shape)
        return ratio

    @staticmethod
    def param_base(
        param: jnp.DeviceArray, mm_param: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """
        param_base reverts the effect of the mismatch and returns the base value

        :param param: the mismatched current parameter
        :type param: jnp.DeviceArray
        :param mm_param: percent mismatch ratio applied
        :type mm_param: jnp.DeviceArray
        :return: the original current base before mismatched applied
        :rtype: jnp.DeviceArray
        """
        return param / (jnp.array(1.0) + mm_param)

    @staticmethod
    def param_eff(param: jnp.DeviceArray, mm_param: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        param_eff simulates the effect of analog device mismatch and applies the percent mismatch to the parameter given

        :param param: the base current parameter which is to be disturbed
        :type param: jnp.DeviceArray
        :param mm_param: percent mismatch ratio applied
        :type mm_param: jnp.DeviceArray
        :return: the mismatch disturbed current
        :rtype: jnp.DeviceArray
        """
        return param + mm_param * param

    def param(self, name: str) -> jnp.DeviceArray:
        """
        param reverts the parameter of interest back to its original base

        :param name: the name of the attribute
        :type name: str
        :return: the original current base before mismatched applied
        :rtype: jnp.DeviceArray
        """
        attr = self.__getattribute__(name)
        mm = self.__getattribute__(f"mm_{name}")
        return self.param_base(attr, mm)
