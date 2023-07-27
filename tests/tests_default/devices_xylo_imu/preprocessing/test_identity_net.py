def test_import():
    import pytest

    pytest.importorskip("samna")
    from rockpool.devices.xylo.imu.preprocessing import IdentityNet

    assert IdentityNet is not None


def test_network_operation():
    """
    The identity network should return the same input spike train provided as input
    """
    import numpy as np
    from rockpool.devices.xylo.imu.preprocessing import IdentityNet

    np.random.seed(2023)
    f_rate = 0.2
    n_channels = 15
    T = 10000
    clock_rate = 200
    speed_up_factor = 2
    input_spike_train = np.random.rand(T, n_channels) < f_rate

    # Operate with XyloSim
    mod = IdentityNet(
        device=None,
        n_channel=n_channels,
        clock_rate=clock_rate,
        speed_up_factor=speed_up_factor,
    )

    out, _, _ = mod(input_spike_train)
    assert np.all(out == input_spike_train)
