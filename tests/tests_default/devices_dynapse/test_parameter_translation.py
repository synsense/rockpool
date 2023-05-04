""" 
The main purpose of the bias generator is converting current values in amperes to a coarse and fine value setting and CF (coarse and fine) settting to current values.
Tests included in this file test if the bias generator implementation is at a healthy state.

Note that the translation does not necessarily need to be one-to-one. If the values are close enough, that's also OK.
"""
import pytest
pytest.importorskip("samna") 

def test_imports():
    """
    test_imports is to first make sure that none of the imports raise any errors
    """
    from rockpool.devices.dynapse.parameters.biasgen import (
        digital_to_analog,
        analog_to_digital,
    )

    from rockpool.devices.dynapse import (
        dynapsim_net_from_config,
        mapper,
        config_from_specification,
        autoencoder_quantization,
        DynapSimCore,
    )

    import samna


def test_digital_to_analog():
    """
    test_digital_to_analog computes an analog current value given a coarse and a fine value setting.
    Then it expects to find the same CF tuple doing the inverse operation (analog current -> digital c&f tuple)
    """
    from rockpool.devices.dynapse.parameters.biasgen import (
        digital_to_analog,
        analog_to_digital,
    )

    # Go over all possible values
    for coarse in range(6):
        for fine in range(256):
            # convert back and forth
            I_val = digital_to_analog(coarse, fine)
            assert (coarse, fine) == analog_to_digital(I_val)


def test_analog_to_digital():
    """
    test_analog_to_digital finds digital bias generator settings given random analog current values in a logarithmic search space
    Then it expects to find a similar current value doing the inverse operation (digital c&f tuple -> analog current)
    """
    import numpy as np
    from rockpool.devices.dynapse.parameters.biasgen import (
        digital_to_analog,
        analog_to_digital,
    )

    # - Create the search space
    space = np.logspace(-13, -5, int(1e3))
    deviation = []

    # - Traverse
    for I_val in space:
        coarse, fine = analog_to_digital(I_val)
        I_val_restored = digital_to_analog(coarse, fine)

        # - Record the percent deviations excluding the ones that are outside the limits
        if (coarse, fine) != (0, 0) and (coarse, fine) != (5, 255):
            val = abs(I_val - I_val_restored) / I_val
            deviation.append(val)

    # - Check the statistical properties of the deviation. One-to-one conversion is not possible!
    assert np.mean(deviation) < 0.01
    assert np.max(deviation) < 0.15


def test_high_level():
    """
    test_high_level obtains a simulation network from a random samna configuration object by doing all current conversions under the hood.
    It obtain another samna config object by processing the network.
    Then compares the current readings of those two configuration objects.
    All bias currents should be the same except for the base weight currents.
    Base weight currents should be different because of the extra quantization step.
    """
    from rockpool.devices.dynapse import (
        dynapsim_net_from_config,
        mapper,
        config_from_specification,
        autoencoder_quantization,
        DynapSimCore,
    )
    import samna

    # - Get a default connfiguration object
    config1 = samna.dynapse2.Dynapse2Configuration()

    # - Create at least one input connection manually to be able to obtain a graph object with input&output nodes
    config1.chips[0].cores[0].neurons[0].synapses[0].dendrite = samna.dynapse2.Dendrite(
        1024
    )
    config1.chips[0].cores[0].neurons[0].synapses[0].weight = [
        True,
        False,
        False,
        False,
    ]
    config1.chips[0].cores[0].neurons[0].synapses[0].tag = 1024

    # - Go through deloyment steps
    # .. seealso ::
    #   :ref:`/devices/DynapSE/post-training.ipynb`
    net = dynapsim_net_from_config(config1)
    spec = mapper(net.as_graph())
    spec.update(autoencoder_quantization(**spec))
    cfg_dict = config_from_specification(**spec)

    # - Read the bias generator settings
    simcore1 = DynapSimCore.from_Dynapse2Core(config1.chips[0].cores[0])
    simcore2 = DynapSimCore.from_Dynapse2Core(cfg_dict["config"].chips[0].cores[0])

    # - Compare the analog current values
    for key in simcore1.__dict__:
        if "Iw" not in key:
            assert simcore1.__dict__[key] == simcore2.__dict__[key]
