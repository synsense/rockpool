def test_imports():
    from rockpool.devices.xylo.xylo_divisive_normalisation import (
        DivisiveNormalisation,
        DivisiveNormalisationNoLFSR,
        LowPassMode,
        build_lfsr,
    )
    from rockpool.devices.xylo import (
        DivisiveNormalisation,
        DivisiveNormalization,
        DivisiveNormalisationNoLFSR,
        DivisiveNormalizationNoLFSR,
        LowPassMode,
        build_lfsr,
    )


def test_DN():
    from rockpool.devices.xylo import DivisiveNormalisation, LowPassMode

    import numpy as np

    dn = DivisiveNormalisation()
    dn = DivisiveNormalisation(16)

    out, n_s, r_d = dn(np.random.rand(100, dn.size_in) < 0.1, record=True)

    dn = DivisiveNormalisation(16, low_pass_mode=LowPassMode.OVERFLOW_PROTECT)

    out, n_s, r_d = dn(np.random.rand(100, dn.size_in) < 0.1, record=True)


def test_DN_NoLFSR():
    from rockpool.devices.xylo import DivisiveNormalisationNoLFSR, LowPassMode

    import numpy as np

    dn = DivisiveNormalisationNoLFSR()
    dn = DivisiveNormalisationNoLFSR(16)

    out, n_s, r_d = dn(np.random.rand(100, dn.size_in) < 0.1, record=True)

    dn = DivisiveNormalisationNoLFSR(16, low_pass_mode=LowPassMode.OVERFLOW_PROTECT)

    out, n_s, r_d = dn(np.random.rand(100, dn.size_in) < 0.1, record=True)


def test_DN_zero_input():
    from rockpool.devices.xylo import DivisiveNormalisation, LowPassMode

    import numpy as np

    T = 1000
    N = 16
    dn = DivisiveNormalisation(N)

    out, _, _ = dn(np.zeros((T, N)))

    assert out.shape == (T, N)
