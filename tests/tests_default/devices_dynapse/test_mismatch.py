"""
Tests here make sure that the frozen mismatch generation works as it should work.
The mismatch generator should deviate the parameters differently at each shot
"""

import pytest


def test_mismatch_distribution():
    """
    test_mismatch_distribution checks if the mismatch deviations are applied to the parameters correctly.
    Each parameter should have a different deviation.

    :raises AttributeError: Mismatch generation is identical across different attributes! Randomization is wrong! Each identity should use a different random number seed!
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")

    from rockpool.devices.dynapse import DynapSim, frozen_mismatch_prototype
    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import LinearJax
    from rockpool.transform.mismatch import mismatch_generator
    from jax import numpy as jnp
    from numpy.testing import assert_array_almost_equal, assert_allclose

    # - Construct DynapSim network
    Nin = 64
    Nrec = 256

    net = Sequential(
        LinearJax(shape=(Nin, Nrec), has_bias=False),
        DynapSim((Nrec, Nrec), has_rec=True),
    )

    # - Preliminaries
    rng_key = jnp.array([2021, 2022], dtype=jnp.uint32)
    prototype = frozen_mismatch_prototype(net)

    # - Generate the mismatch
    percent = 0.30
    regenerate_mismatch = mismatch_generator(
        prototype=prototype, percent_deviation=percent
    )
    new_params = regenerate_mismatch(net, rng_key=rng_key)

    # - Check the direction of changes of the different attributes
    sign_prev = None
    counter = 0

    # - Go through the layers
    for layer_name in prototype:
        __layer = net.__getattribute__(layer_name)

        # - Check each attribute
        for attr_name in prototype[layer_name]:
            __attr = __layer.__getattribute__(attr_name)
            if prototype[layer_name][attr_name]:
                # - If the parameter is subject to mismatch, then the parameter values should not be close to the original parameter values
                with pytest.raises(AssertionError):
                    assert_allclose(__attr, new_params[layer_name][attr_name])
            else:
                # - Due to type casting, the parameters might not be exactly equal in every bit
                assert_array_almost_equal(__attr, new_params[layer_name][attr_name])

            # - Record the direction of change
            sign = jnp.sign(__attr - new_params[layer_name][attr_name])

            # - Compare with the previous direction pattern
            if sign_prev is not None:
                if (sign_prev == sign).all():
                    counter += 1
                else:
                    counter = 0

            sign_prev = sign

            # - If the pattern is the same, the randomization is erronous
            if counter > 3:
                raise AttributeError(
                    "Mismatch generation is identical across different attributes! Randomization is wrong! Each identity should use a different random number seed!"
                )


def test_mismatch_statistics():
    """
    test_mismatch_statistics checks the statistical properties of the deviated parameters to make sure that the mismatch generation procedure can keep the mean value
    at the same point and create a standard deviation on parameters.
    """
    import pytest

    pytest.importorskip("samna")
    pytest.importorskip("jax")

    from rockpool.devices.dynapse import DynapSim, frozen_mismatch_prototype
    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import LinearJax
    from rockpool.transform.mismatch import mismatch_generator
    from jax import numpy as jnp
    from numpy.testing import assert_almost_equal

    # - Construct DynapSim network
    Nin = 64
    Nrec = 256

    net = Sequential(
        LinearJax(shape=(Nin, Nrec), has_bias=False),
        DynapSim((Nrec, Nrec), has_rec=True),
    )

    # - Preliminaries
    rng_key = jnp.array([2021, 2022], dtype=jnp.uint32)
    prototype = frozen_mismatch_prototype(net)

    # - Regenerate the mismatch
    regenerate_mismatch = mismatch_generator(prototype=prototype, percent_deviation=0.1)
    new_params = regenerate_mismatch(net, rng_key=rng_key)

    # - Go through the layers
    for layer_name in prototype:
        __layer = net.__getattribute__(layer_name)

        # - Check each attribute
        for attr_name in prototype[layer_name]:
            __attr = __layer.__getattribute__(attr_name)

            if prototype[layer_name][attr_name]:
                # - Even if the parameter is subject to mismatch, the mean value should be fixed
                assert_almost_equal(
                    jnp.mean(new_params[layer_name][attr_name]),
                    jnp.mean(__attr),
                    decimal=2,
                )

                # - Make sure that the standard deviation of the original model is 0.0
                assert_almost_equal(jnp.std(__attr) / jnp.mean(__attr), 0.0)

                # - The standard deviations of the deviated parameters should be a lot bigger than zero
                with pytest.raises(AssertionError):
                    assert_almost_equal(
                        jnp.std(new_params[layer_name][attr_name])
                        / jnp.mean(new_params[layer_name][attr_name]),
                        0.0,
                        decimal=2,
                    )

            # - If the parameters are not subject to mismatch the mean value and the standard deviation should be the same
            else:
                assert_almost_equal(
                    jnp.mean(new_params[layer_name][attr_name]), jnp.mean(__attr)
                )
                assert_almost_equal(
                    jnp.std(new_params[layer_name][attr_name]), jnp.std(__attr)
                )
