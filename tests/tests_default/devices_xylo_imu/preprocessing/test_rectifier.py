def test_import():
    from rockpool.devices.xylo.imu.preprocessing import (
        FullWaveRectifier,
        HalfWaveRectifier,
    )

    assert FullWaveRectifier is not None
    assert HalfWaveRectifier is not None


def test_half_wave_rectifier():
    from rockpool.devices.xylo.imu.preprocessing import HalfWaveRectifier

    half_wave = HalfWaveRectifier(shape=1)

    out, _, _ = half_wave([5])
    assert out == 5  # Positive input should pass through

    out, _, _ = half_wave([-5])
    assert out == 0  # Negative input should be rectified to 0

    out, _, _ = half_wave([0])
    assert out == 0  # Zero input should be rectified to 0

    out, _, _ = half_wave([10])
    assert out == 10  # Positive input should pass through

    out, _, _ = half_wave([-10])
    assert out == 0  # Negative input should be rectified to 0


def test_full_wave_rectifier():
    from rockpool.devices.xylo.imu.preprocessing import FullWaveRectifier

    full_wave = FullWaveRectifier(shape=1)

    out, _, _ = full_wave([5])
    assert out == 5  # Positive input should pass through

    out, _, _ = full_wave([-5])
    assert out == 5  # Negative input should be rectified to positive

    out, _, _ = full_wave([0])
    assert out == 0  # Zero input should be rectified to 0

    out, _, _ = full_wave([10])
    assert out == 10  # Positive input should pass through

    out, _, _ = full_wave([-10])
    assert out == 10  # Negative input should be rectified to positive


if __name__ == "__main__":
    test_half_wave_rectifier()
