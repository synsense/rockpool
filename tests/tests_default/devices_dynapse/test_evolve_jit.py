"""
Test if a Dynap-SE2 network works the same with jitted and non-jitted version
"""


def test_evolve():
    """
    test_evolve checks if the compiled network and the non-compiled network works the same
    """

    ### --- Preliminaries --- ###
    import os
    import numpy as np
    from rockpool.devices.dynapse import DynapSim
    from jax import jit
    from jax import numpy as jnp
    from numpy.testing import assert_array_equal, assert_array_almost_equal

    # - Hyper-parameters
    np.random.seed(2023)

    T = 1000
    Nrec = 60
    f = 0.01

    # - Build the network
    net = DynapSim(Nrec, has_rec=True)

    # - Random input data
    spike_train = np.random.rand(T, Nrec) < f
    spike_train = spike_train.reshape(1, T, Nrec)

    # - Compile
    net_jit = jit(net)

    # - Regular
    out, state, rec = net(spike_train)

    # - Compiled
    out_jit, state_jit, rec_jit = net_jit(spike_train)

    # make sure that network has activity
    assert jnp.sum(out) > 0

    # - Check the output activity and the records
    assert_array_equal(out, out_jit)

    # Type casting results slight differences, which does not have any practical effects on results
    for key in state:
        assert_array_almost_equal(state[key], state_jit[key])

    for key in rec:
        assert_array_almost_equal(rec[key], rec_jit[key])
