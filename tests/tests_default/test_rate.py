import pytest


def test_imports():
    from rockpool.nn.modules import Rate


def test_evolve():
    from rockpool.nn.modules import Rate
    import numpy as np

    # - Test instantiation
    N = 2
    T = 100
    batches = 3
    mod = Rate(N)

    data = np.random.randn(batches, T, N)
    out, _, _ = mod(data)
    assert out.shape == (batches, T, N)

    data = np.random.randn(T, N)
    out, _, _ = mod(data)
    assert out.shape == (1, T, N)


def test_evolve_rec():
    # - Test recurrence
    from rockpool.nn.modules import Rate
    import numpy as np

    # - Test instantiation
    N = 2
    T = 100
    batches = 3
    mod = Rate(N, has_rec=True)
    data = np.random.randn(batches, T, N)
    out, _, _ = mod(data)
    assert out.shape == (batches, T, N)

    data = np.random.randn(T, N)
    out, _, _ = mod(data)
    assert out.shape == (1, T, N)
