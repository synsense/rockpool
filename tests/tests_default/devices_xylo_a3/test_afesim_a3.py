import pytest


def test_import():
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    assert AFESim is not None


@pytest.mark.parametrize(
    "rate_scale_factor,rate_scale_bitshift",
    [(1, (1, 0)), (16, (5, 4)), (32, (6, 5)), (63, (6, 0))],
)
def test_dn_rate_scale_bitshift_known_feasible(rate_scale_factor, rate_scale_bitshift):
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    bitshift = AFESim.get_dn_rate_scale_bitshift(rate_scale_factor=rate_scale_factor)
    assert bitshift == rate_scale_bitshift


@pytest.mark.parametrize("rate_scale_factor", [-10, "a", 0, 5, 61])
def test_dn_rate_scale_bitshift_known_raising_error(rate_scale_factor):
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    with pytest.raises(ValueError):
        AFESim.get_dn_rate_scale_bitshift(rate_scale_factor=rate_scale_factor)
