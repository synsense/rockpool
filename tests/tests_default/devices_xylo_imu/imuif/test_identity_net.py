import pytest

pytest.importorskip("samna")


def test_import():
    from rockpool.devices.xylo.syns63300 import IdentityNet

    assert IdentityNet is not None


@pytest.mark.parametrize("seed", [1995, 2023])
@pytest.mark.parametrize("f_rate,T", [(0.2, 1000), (0.8, 10000)])
def test_network_operation(seed: int, f_rate: float, T: int):
    """
    The identity network should return the same input spike train provided as input

    Args:
        seed (int): The seed for the random number generator
        f_rate (float): The firing rate of the input spike train
        T (int): The length of the input spike train
    """
    import numpy as np
    from rockpool.devices.xylo.syns63300 import IdentityNet

    np.random.seed(seed)
    n_channels = 15
    clock_rate = 200
    input_spike_train = np.random.rand(T, n_channels) < f_rate

    # Operate with XyloSim
    mod = IdentityNet(
        device=None,
        n_channel=n_channels,
        clock_rate=clock_rate,
    )

    out, _, _ = mod(input_spike_train)
    assert np.all(out == input_spike_train)
