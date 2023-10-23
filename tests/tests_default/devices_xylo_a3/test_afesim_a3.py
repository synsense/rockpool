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
    Test that the dn_rate_scale_bitshift is computed correctly for known feasible values

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
    Test that the dn_rate_scale_bitshift raises a ValueError for known infeasible values

    Args:
        rate_scale_factor (int): Target `rate_scale_factor` for the `DivisiveNormalization` module.
    """
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    with pytest.raises(ValueError):
        AFESim.get_dn_rate_scale_bitshift(rate_scale_factor=rate_scale_factor)
