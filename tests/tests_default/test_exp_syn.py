def test_imports():
    from rockpool.nn.modules import ExpSyn, ExpSynJax


def test_ExpSyn():
    from rockpool.nn.modules import ExpSyn
    import numpy as np

    batches = 2
    T = 1000
    dt = 1e-3
    N = 4
    p = 0.1

    esMod = ExpSyn(N, tau=[10e-3, 50e-3, 100e-3, 200e-3], dt=dt)

    sp_rand = np.random.rand(batches, T, N) < p

    out, ns, rs = esMod(sp_rand)
    print(out)
