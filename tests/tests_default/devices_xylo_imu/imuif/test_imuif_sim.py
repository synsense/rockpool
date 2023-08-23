def test_import():
    import pytest

    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns63300 import IMUIFSim

    assert IMUIFSim is not None


def test_import_samna_config():
    import pytest

    pytest.importorskip("samna")
    from samna.xyloImu.configuration import InputInterfaceConfig

    assert InputInterfaceConfig is not None


def test_simulate():
    import pytest

    pytest.importorskip("samna")
    import numpy as np
    from numpy.testing import assert_array_compare, assert_array_equal

    from rockpool.devices.xylo.syns63300 import IMUIFSim

    mod_if = IMUIFSim(bypass_jsvd=False)

    data = np.random.randint(0, 2**15, (2, 100, mod_if.size_in)).astype(object)
    out, _, _ = mod_if(data)

    # Compare the output against the limit value
    assert_array_compare(lambda x, y: x <= y, out, np.full_like(out, 15))
    assert_array_compare(lambda x, y: x >= y, out, np.zeros_like(out))

    with pytest.raises(AssertionError):
        assert_array_equal(out, np.zeros_like(out))


def test_samna_import():
    import pytest

    pytest.importorskip("samna")
    import numpy as np
    from numpy.testing import assert_array_compare, assert_array_equal
    from samna.xyloImu.configuration import InputInterfaceConfig

    from rockpool.devices.xylo.syns63300 import IMUIFSim

    default_config = InputInterfaceConfig(enable=True)

    mod_if = IMUIFSim.from_config(default_config)

    data = np.zeros((2, 100, mod_if.size_in)).astype(int).astype(object)
    out, _, _ = mod_if(data)

    assert_array_equal(out, np.zeros_like(out))


def test_samna_export():
    import pytest

    pytest.importorskip("samna")
    from samna.xyloImu.configuration import InputInterfaceConfig

    from rockpool.devices.xylo.syns63300 import IMUIFSim

    default_config = InputInterfaceConfig(enable=True)

    mod_if = IMUIFSim.from_config(default_config)
    config = mod_if.export_config()
    assert config.to_json() == default_config.to_json()

    change_config = IMUIFSim(
        bypass_jsvd=not default_config.bypass_jsvd,
        num_avg_bitshift=default_config.estimator_k_setting + 1,
    ).export_config()

    with pytest.raises(AssertionError):
        assert config.to_json() == change_config.to_json()

    change_config = IMUIFSim(
        bypass_jsvd=not default_config.select_iaf_output,
    ).export_config()

    with pytest.raises(AssertionError):
        assert config.to_json() == change_config.to_json()
