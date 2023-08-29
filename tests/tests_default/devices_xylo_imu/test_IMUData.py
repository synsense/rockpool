def test_IMUData():
    import pytest

    pytest.importorskip("samna")

    from rockpool.devices.xylo.syns63300 import IMUData, find_xylo_imu_boards
    import numpy as np

    hdks = find_xylo_imu_boards()

    if len(hdks) < 1:
        pytest.skip("A Xylo IMU device is required for this test")

    imu = IMUData(hdks[0])
    data = imu(np.zeros((0, 100, 0)))
