def test_import():
    from rockpool.devices.xylo.imu.imuif import (
        IAFSpikeEncoder,
        ScaleSpikeEncoder,
    )

    assert ScaleSpikeEncoder is not None
    assert IAFSpikeEncoder is not None


def test_scale_spike_encoder():
    import numpy as np
    import pytest
    from numpy.testing import assert_array_compare, assert_array_equal

    from rockpool.devices.xylo.imu.imuif import ScaleSpikeEncoder

    # Prepare input
    np.random.seed(2023)
    input_data = np.random.randint(0, 2**16, size=(10, 100, 15), dtype=np.uint16)

    # Simulate the encoder
    spike_encoder = ScaleSpikeEncoder()
    out, _, _ = spike_encoder(input_data.astype(object))

    # Compare the output against the limit value
    assert_array_compare(lambda x, y: x <= y, out, np.full_like(out, 15))
    assert_array_compare(lambda x, y: x >= y, out, np.zeros_like(out))

    with pytest.raises(AssertionError):
        assert_array_equal(out, np.zeros_like(out))


def test_iaf_spike_encoder():
    import numpy as np
    import pytest
    from numpy.testing import assert_array_compare, assert_array_equal

    from rockpool.devices.xylo.imu.imuif import IAFSpikeEncoder

    # Prepare input
    np.random.seed(2023)
    input_data = np.random.randint(0, 2**16, size=(10, 100, 15), dtype=np.uint16)

    # Simulate the encoder
    spike_encoder = IAFSpikeEncoder(threshold=1000)
    out, _, _ = spike_encoder(input_data.astype(object))

    # Compare the output against the limit value
    assert_array_compare(lambda x, y: x <= y, out, np.ones_like(out))
    assert_array_compare(lambda x, y: x >= y, out, np.zeros_like(out))

    with pytest.raises(AssertionError):
        assert_array_equal(out, np.zeros_like(out))
