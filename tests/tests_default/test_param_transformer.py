def test_quantise():
    from rockpool.quantize import StochasticQuantize
    from rockpool.rate_jax import RateEulerJax
    from rockpool.ffwd_stack import FFwdStack

    import numpy as np

    mod = FFwdStack(RateEulerJax(10), RateEulerJax((5, 5)), RateEulerJax(1))

    sq = StochasticQuantize(mod, families="weights")
    p = sq.parameters()
    tfp = sq.transformed_parameters()

    input_data = np.random.rand(100, 10)
    o, ns, r_d = sq.evolve(input_data)
    sq = sq.set_attributes(ns)

    o, ns, r_d = sq(input_data)
    sq = sq.set_attributes(ns)


def test_dropout():
    from rockpool.dropout import Dropout
    from rockpool.rate_jax import RateEulerJax
    from rockpool.ffwd_stack import FFwdStack

    import numpy as np

    mod = FFwdStack(RateEulerJax(10), RateEulerJax((5, 5)), RateEulerJax(1))

    sq = Dropout(mod, "weights", 0.5)
    p = sq.parameters()
    tfp = sq.transformed_parameters()

    input_data = np.random.rand(100, 10)
    o, ns, r_d = sq.evolve(input_data)
    sq = sq.set_attributes(ns)

    o, ns, r_d = sq(input_data)
    sq = sq.set_attributes(ns)
