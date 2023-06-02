def test_import():
    from rockpool.devices.xylo.imu.preprocessing import (
        IAFSpikeEncoder,
        ScaleSpikeEncoder,
    )

    assert ScaleSpikeEncoder is not None
    assert IAFSpikeEncoder is not None


def test_scale_spike_encoder():
    import numpy as np
    from numpy.testing import assert_array_compare

    from rockpool.devices.xylo.imu.preprocessing import ScaleSpikeEncoder

    # Prepare input
    np.random.seed(2023)
    input_data = np.random.randint(0, 2**16, size=(10, 100, 48), dtype=np.uint16)

    # Simulate the encoder
    spike_encoder = ScaleSpikeEncoder(num_scale_bits=10, num_out_bits=4)
    out, _, _ = spike_encoder(input_data.astype(object))

    # Compare the output against the limit value
    assert_array_compare(lambda x, y: x <= y, out, np.full_like(out, 15))
    assert_array_compare(lambda x, y: x >= y, out, np.zeros_like(out))


def test_iaf_spike_encoder():
    import numpy as np
    from numpy.testing import assert_array_compare

    from rockpool.devices.xylo.imu.preprocessing import IAFSpikeEncoder

    # Prepare input
    np.random.seed(2023)
    input_data = np.random.randint(0, 2**16, size=(10, 100, 48), dtype=np.uint16)

    # Simulate the encoder
    spike_encoder = IAFSpikeEncoder(1000)
    out, _, _ = spike_encoder(input_data.astype(object))

    # Compare the output against the limit value
    assert_array_compare(lambda x, y: x <= y, out, np.ones_like(out))
    assert_array_compare(lambda x, y: x >= y, out, np.zeros_like(out))
