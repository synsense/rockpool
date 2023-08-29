""" 
Tests for quantization preprocessing module
"""


def test_import():
    from rockpool.devices.xylo.syns63300.transform import Quantizer


def test_quantization():
    import numpy as np
    from numpy.testing import assert_array_equal
    from rockpool.devices.xylo.syns63300.transform import Quantizer

    # Create quantizer with scale of 1.0 and num_bits of 16
    quantizer = Quantizer(shape=1, scale=1.0, num_bits=16)

    # Quantize input signal
    input_data = np.array([1, -1, 0.00004, 0.00003])
    output, _, _ = quantizer.evolve(input_data)

    # Check that output is equal to expected_output
    expected_output = np.array([[32768, -32768, 1, 0]]).T
    expected_output = np.expand_dims(expected_output, 0)
    assert_array_equal(output, expected_output)


def test_quantization_num_bits():
    import numpy as np
    from numpy.testing import assert_array_equal
    from rockpool.devices.xylo.syns63300.transform import Quantizer

    # Create quantizer with scale of 1.0 and num_bits of 4
    quantizer = Quantizer(shape=1, scale=1.0, num_bits=4)

    # Quantize a synthetic input signal
    input_data = np.array([1, -1, 0.125, 0.1])
    output, _, _ = quantizer.evolve(input_data)

    # Check that output is equal to expected_output
    expected_output = np.array([[8, -8, 1, 0]]).T
    expected_output = np.expand_dims(expected_output, 0)
    assert_array_equal(output, expected_output)
