"""
Test back-step spiking layer used for Deneve networks, as well as network implementation
"""

import numpy as np


def test_imports():
    pass


def test_RecFSSpikeEulerBT():
    """Test RecFSSpikeEulerBT"""
    from rockpool import timeseries as ts
    from rockpool.nn.layers import RecFSSpikeEulerBT

    # - Generic parameters
    mfW = 2 * np.random.rand(3, 3) - 1
    vfBias = 2 * np.random.rand(3) - 1
    vtTauN = np.random.rand(3)

    # - Layer generation
    fl0 = RecFSSpikeEulerBT(
        weights_fast=mfW,
        weights_slow=mfW,
        bias=vfBias,
        noise_std=0.1,
        tau_mem=vtTauN,
        tau_syn_r_fast=1e-3,
        tau_syn_r_slow=100e-3,
        v_thresh=-55e-3,
        v_reset=-65e-3,
        v_rest=-65e-3,
        refractory=2e-3,
        spike_callback=None,
        dt=None,
        name="test_RecFSSpikeEulerBT",
    )

    # - Input signal
    tsInCont = ts.TSContinuous(times=np.arange(15) * 0.01, samples=np.ones((15, 3)))

    # - Compare states and time before and after
    vStateBefore = np.copy(fl0.state)
    fl0.evolve(tsInCont, duration=0.1)
    assert fl0.t == 0.1
    assert (vStateBefore != fl0.state).any()

    fl0.reset_all()
    assert fl0.t == 0
    assert (vStateBefore == fl0.state).all()


def FAILING_test_SolveLinearSystem():
    """Test NetworkDeneve.SolveLinearSystem"""
    from rockpool.nn.networks import NetworkDeneve
    import numpy as np
    from rockpool import TSContinuous

    # - Define the dynamical system to be approximated
    nNumVariables = 1
    fLambda_s = 1
    mfA = np.asarray(-fLambda_s * np.identity(nNumVariables))

    # - Generate random decoding weights
    nNetSize = 100
    mfGamma = np.random.randn(nNumVariables, nNetSize)

    # - Generate direct solution
    net = NetworkDeneve.SolveLinearProblem(a=mfA, net_size=nNetSize, gamma=mfGamma)

    # - Generate a command signal
    fhCommand = lambda t: np.asarray(
        np.asarray(t) > np.asarray(200e-3), float
    ) * np.asarray(np.asarray(t) < np.asarray(500e-3), float) - np.asarray(
        np.asarray(t) > np.asarray(800e-3), float
    ) * np.asarray(
        np.asarray(t) < np.asarray(1000e-3), float
    )
    tDuration = 2000e-3
    fCommandAmp = 1000
    fSigmaS = 0.01 * fCommandAmp
    tNoiseEnd = 1200e-3
    tResCommand = 0.1e-3
    fhNoise = (
        lambda t: np.random.randn(np.size(t))
        * fSigmaS
        * np.asarray(np.asarray(t) < np.asarray(tNoiseEnd))
    )
    vtTimeTrace = np.arange(0, tDuration + tResCommand, tResCommand)
    tsCommand = TSContinuous(
        vtTimeTrace,
        fCommandAmp * fhCommand(vtTimeTrace) + fhNoise(vtTimeTrace),
        periodic=True,
        name="Command",
    )

    # - Evolve the network with the command signal
    dResponse = net.evolve(tsCommand)


def FAILING_test_SpecifyNetwork():
    from rockpool.nn.networks import NetworkDeneve
    from rockpool import TSContinuous

    net_size = 10
    in_size = 3
    out_size = 2

    weights_fast = np.random.rand(net_size, net_size)
    weights_slow = np.random.rand(net_size, net_size)
    weights_in = np.random.rand(in_size, net_size)
    weights_out = np.random.rand(net_size, out_size)

    net = NetworkDeneve.SpecifyNetwork(
        weights_fast, weights_slow, weights_in, weights_out
    )

    # - Generate an input signal
    tDuration = 2000e-3
    tResCommand = 0.1e-3
    vtTimeTrace = np.arange(0, tDuration + tResCommand, tResCommand)
    tsCommand = TSContinuous(
        vtTimeTrace, np.random.rand(vtTimeTrace.size), periodic=True, name="Command"
    )

    # - Evolve the network with the command signal
    dResponse = net.evolve(tsCommand)
