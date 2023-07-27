def test_import():
    import pytest

    pytest.importorskip("samna")
    from rockpool.devices.xylo.imu.preprocessing import IdentityNet

    assert IdentityNet is not None
