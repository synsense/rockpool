import pytest


def test_import():
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    assert AFESim is not None
