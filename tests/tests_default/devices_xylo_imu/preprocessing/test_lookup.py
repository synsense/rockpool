# test lookup table


def test_import():
    from rockpool.devices.xylo.imu.preprocessing import RotationLookUpTable


def test_lookup_table():
    import numpy as np
    from numpy.testing import assert_almost_equal

    from rockpool.devices.xylo.imu.preprocessing import RotationLookUpTable

    num_angles = 64
    num_bits = 16

    lut = RotationLookUpTable(num_angles=num_angles, num_bits=num_bits)

    # find a set of parameters
    a = 2
    c = 1
    b = 1.2

    # quantize thse values
    num_bits_quant = 10
    a_quant = int(2 ** (num_bits_quant - 1) * a)
    b_quant = int(2 ** (num_bits_quant - 1) * b)
    c_quant = int(2 ** (num_bits_quant - 1) * c)

    (
        row_index,
        angle_deg,
        angle_rad,
        sin_val,
        cos_val,
        inv_2sin2_val,
        inv_2cos2_val,
        tan2_val,
        cot2_val,
        sin_val_quant,
        cos_val_quant,
        inv_2sin2_val_quantized,
        inv_2cos2_val_quantized,
        tan2_val_quantized,
        cot2_val_quantized,
    ) = lut.find_angle(a_quant, b_quant, c_quant)

    # Compare values against true angle
    true_angle = 0.5 * np.arctan(2 * b / (a - c)) if a != c else np.pi / 4

    assert np.round(angle_deg) == np.round(true_angle * 180 / np.pi)
    assert_almost_equal(angle_rad, true_angle, decimal=1)
    assert_almost_equal(sin_val, np.sin(true_angle), decimal=1)
    assert_almost_equal(cos_val, np.cos(true_angle), decimal=1)
    assert_almost_equal(inv_2sin2_val, 1 / (2 * np.sin(2 * true_angle)), decimal=1)
    assert_almost_equal(cot2_val, 1 / np.tan(2 * true_angle), decimal=1)
