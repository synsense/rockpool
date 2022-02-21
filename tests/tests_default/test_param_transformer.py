import pytest

pytest.importorskip("jax")


def test_quantise():
    from rockpool.transform.quantize import StochasticQuantize
    from rockpool.nn.modules import RateJax
    from rockpool.nn.combinators import FFwdStack

    import numpy as np

    mod = FFwdStack(RateJax(10), RateJax((5, 5)), RateJax(1))

    sq = StochasticQuantize(mod, families="weights")
    p = sq.parameters()
    tfp = sq.transformed_parameters()

    input_data = np.random.rand(100, 10)
    o, ns, r_d = sq.evolve(input_data)
    sq = sq.set_attributes(ns)

    o, ns, r_d = sq(input_data)
    sq = sq.set_attributes(ns)


def test_dropout():
    from rockpool.transform.dropout import Dropout
    from rockpool.nn.modules import RateJax
    from rockpool.nn.combinators import FFwdStack

    import numpy as np

    mod = FFwdStack(RateJax(10), RateJax((5, 5)), RateJax(1))

    sq = Dropout(mod, "weights", 0.5)
    p = sq.parameters()
    tfp = sq.transformed_parameters()

    input_data = np.random.rand(100, 10)
    o, ns, r_d = sq.evolve(input_data)
    sq = sq.set_attributes(ns)

    o, ns, r_d = sq(input_data)
    sq = sq.set_attributes(ns)
