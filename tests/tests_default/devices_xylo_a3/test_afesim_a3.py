import pytest


def test_import() -> None:
    """Test that the AFESim modules can be imported"""

    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESimAGC, AFESimPDM, AFESimExternal

    assert AFESimAGC is not None
    assert AFESimPDM is not None
    assert AFESimExternal is not None


@pytest.mark.parametrize(
    "rate_scale_factor,rate_scale_bitshift",
    [(1, (1, 0)), (16, (5, 4)), (32, (6, 5)), (63, (6, 0))],
)
def test_dn_rate_scale_bitshift_known_feasible(
    rate_scale_factor: int, rate_scale_bitshift: tuple
) -> None:
    """
    Test that the `dn_rate_scale_bitshift` is computed correctly for known feasible values

    Args:
        rate_scale_factor (float): Target `rate_scale_factor` for the `DivisiveNormalization` module.
        rate_scale_bitshift (tuple): Expected `rate_scale_bitshift` calculated given the target `rate_scale_factor`.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESimExternal

    bitshift = AFESimExternal.get_dn_rate_scale_bitshift(
        rate_scale_factor=rate_scale_factor
    )
    assert bitshift == rate_scale_bitshift


@pytest.mark.parametrize("rate_scale_factor", [-10, "a", 0, 5, 61])
def test_dn_rate_scale_bitshift_known_raising_error(rate_scale_factor: int) -> None:
    """
    Test that the `get_dn_rate_scale_bitshift` raises a `ValueError` for known infeasible values

    Args:
        rate_scale_factor (int): Target `rate_scale_factor` for the `DivisiveNormalization` module.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESimExternal

    with pytest.raises(ValueError):
        AFESimExternal.get_dn_rate_scale_bitshift(rate_scale_factor=rate_scale_factor)


@pytest.mark.parametrize(
    "low_pass_averaging_window,low_pass_bitshift",
    [(4e-5, 1), (1e-3, 6), (20e-3, 10), (84e-3, 12)],
)
def test_dn_low_pass_bitshift_known_feasible(
    low_pass_averaging_window: float, low_pass_bitshift: int
) -> None:
    """
    Test that the `dn_low_pass_bitshift` is computed correctly for known feasible values

    Args:
        low_pass_averaging_window (float): Target `low_pass_averaging_window` for the `DivisiveNormalization` module.
        low_pass_bitshift (int): Expected `low_pass_bitshift` calculated given the target `low_pass_averaging_window`.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESimExternal, AUDIO_SAMPLING_RATE

    bitshift = AFESimExternal.get_dn_low_pass_bitshift(
        audio_sampling_rate=AUDIO_SAMPLING_RATE,
        low_pass_averaging_window=low_pass_averaging_window,
    )
    assert bitshift == low_pass_bitshift


@pytest.mark.parametrize("low_pass_averaging_window", [-10, 1e-6, "a", 85e-3])
def test_dn_low_pass_bitshift_known_raising_error(
    low_pass_averaging_window: int,
) -> None:
    """
    Test that the `get_dn_low_pass_bitshift` raises a `ValueError` for known infeasible values

    Args:
        low_pass_averaging_window (int): Target `low_pass_averaging_window` for the `DivisiveNormalization` module.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESimExternal, AUDIO_SAMPLING_RATE

    with pytest.raises(ValueError):
        AFESimExternal.get_dn_low_pass_bitshift(
            audio_sampling_rate=AUDIO_SAMPLING_RATE,
            low_pass_averaging_window=low_pass_averaging_window,
        )


@pytest.mark.parametrize(
    "dt,down_sampling_factor", [(205e-7, 1), (1024e-6, 50), (64 / 48828, 64)]
)
def test_down_sampling_factor_known_feasible(
    dt: float, down_sampling_factor: int
) -> None:
    """
    Test that the `down_sampling_factor` is computed correctly for known feasible `dt` values

    Args:
        dt (float): Sampling period of the audio signal.
        down_sampling_factor (int): Expected `down_sampling_factor` calculated given the target `dt`.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESimExternal, AUDIO_SAMPLING_RATE

    factor = AFESimExternal.get_down_sampling_factor(
        audio_sampling_rate=AUDIO_SAMPLING_RATE, dt=dt
    )
    assert factor == down_sampling_factor


@pytest.mark.parametrize("dt", [-10, 1e-6, "a", 1e-3])
def test_down_sampling_factor_known_raising_error(dt: float) -> None:
    """
    Test that the `get_down_sampling_factor` raises a `ValueError` for known infeasible values

    Args:
        dt (float): Sampling period of the audio signal.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESimExternal, AUDIO_SAMPLING_RATE

    with pytest.raises(ValueError):
        AFESimExternal.get_down_sampling_factor(
            audio_sampling_rate=AUDIO_SAMPLING_RATE, dt=dt
        )
