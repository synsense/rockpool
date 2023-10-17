import pytest


def test_import():
    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import AFESim

    assert AFESim is not None


def test_valid_filters():
    from rockpool.devices.xylo.syns65302 import AFESim

    select_filters = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert AFESim.validate_filter_selection(select_filters=select_filters)

    select_filters = (15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
    assert AFESim.validate_filter_selection(select_filters=select_filters)

    select_filters = (10, 3, 7, 0, 14)
    assert AFESim.validate_filter_selection(select_filters=select_filters)

    with pytest.raises(TypeError):
        AFESim([0, 1, 2])

    with pytest.raises(ValueError):
        AFESim((0, 1, 2, 16))

    with pytest.raises(ValueError):
        AFESim((0, 1, 2, -1))

    with pytest.raises(TypeError):
        AFESim((0, 1, 2, "a"))

    with pytest.raises(TypeError):
        AFESim((0, 1, 2, 3.5))

    with pytest.raises(ValueError):
        AFESim((0, 0, 1, 2))
