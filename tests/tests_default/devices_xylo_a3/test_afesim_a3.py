import pytest


def test_import():
    """Test that the AFESim module can be imported"""

    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    assert AFESim is not None


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
    from rockpool.devices.xylo.syns65302 import AFESim

    bitshift = AFESim.get_dn_rate_scale_bitshift(rate_scale_factor=rate_scale_factor)
    assert bitshift == rate_scale_bitshift


@pytest.mark.parametrize("rate_scale_factor", [-10, "a", 0, 5, 61])
def test_dn_rate_scale_bitshift_known_raising_error(rate_scale_factor: int):
    """
    Test that the `get_dn_rate_scale_bitshift` raises a `ValueError` for known infeasible values

    Args:
        rate_scale_factor (int): Target `rate_scale_factor` for the `DivisiveNormalization` module.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    with pytest.raises(ValueError):
        AFESim.get_dn_rate_scale_bitshift(rate_scale_factor=rate_scale_factor)


@pytest.mark.parametrize(
    "low_pass_averaging_window,low_pass_bitshift",
    [(4e-5, 1), (1e-3, 6), (20e-3, 10), (84e-3, 12)],
)
def test_dn_low_pass_bitshift(low_pass_averaging_window: float, low_pass_bitshift: int):
    """
    Test that the `dn_low_pass_bitshift` is computed correctly for known feasible values

    Args:
        low_pass_averaging_window (float): Target `low_pass_averaging_window` for the `DivisiveNormalization` module.
        low_pass_bitshift (int): Expected `low_pass_bitshift` calculated given the target `low_pass_averaging_window`.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    bitshift = AFESim.get_dn_low_pass_bitshift(
        low_pass_averaging_window=low_pass_averaging_window
    )
    assert bitshift == low_pass_bitshift


@pytest.mark.parametrize("low_pass_averaging_window", [-10, 1e-6, "a", 85e-3])
def test_dn_low_pass_bitshift_known_raising_error(low_pass_averaging_window: int):
    """
    Test that the `get_dn_low_pass_bitshift` raises a `ValueError` for known infeasible values

    Args:
        low_pass_averaging_window (int): Target `low_pass_averaging_window` for the `DivisiveNormalization` module.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    with pytest.raises(ValueError):
        AFESim.get_dn_low_pass_bitshift(
            low_pass_averaging_window=low_pass_averaging_window
        )
