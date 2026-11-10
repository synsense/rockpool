import pytest

pytest.importorskip("torch")


def test_LIFTorch_shapes():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.05

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    assert out.shape == (n_batches, T, n_neurons)
    assert ns["isyn"].shape == (n_neurons, n_synapses)
    assert ns["vmem"].shape == (n_neurons,)
    assert rd["isyn"].shape == (n_batches, T, n_neurons, n_synapses)
    assert rd["vmem"].shape == (n_batches, T, n_neurons)


def test_LIFTorch_bias():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 1
    n_neurons = 1
    n_batches = 1
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.02
    bias = torch.ones(n_neurons) * 0.1
    dt = 1e-3

    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=1.0,
        bias=bias,
        dt=dt,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    assert torch.all(ns["isyn"] == 0)
    assert torch.all(rd["isyn"] == 0)
    assert torch.all(rd["vmem"][:, 0] == 0.1)  # match bias in the fist timestep
    assert torch.all(
        rd["vmem"][:, 1] == 0.1 * torch.exp(-dt / tau_mem) + 0.1
    )  # decay one timestep + bias

    # assert bias has gradients
    assert not torch.all(mod.bias.grad == 0)


def test_LIFTorch_recurrent():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 2
    n_neurons = 5
    n_batches = 1
    T = 20
    tau_mem = 0.01
    tau_syn = 0.02

    # more recurrent input to neurons with higher id
    w_rec = torch.zeros(n_neurons, n_neurons, n_synapses)
    w_rec[0, 1, 0] = 1  # neuron 1, synapse 0
    w_rec[0, 1, 1] = 2  # neuron 1, synapse 1
    w_rec[0, 2, 0] = 3  # neuron 2, synapse 0
    w_rec[0, 2, 1] = 4  # neuron 2, synapse 1
    w_rec[0, 3, 0] = 5  # neuron 3, synapse 0
    w_rec[0, 4, 1] = 6  # neuron 4, synapse 1

    w_rec = w_rec.reshape(n_neurons, n_neurons * n_synapses)

    dt = 1e-3

    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_rec=True,
        w_rec=w_rec,
        dt=dt,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)
    with torch.no_grad():
        input_data[:, 0, 0] = 100

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert neurons are increasingly active (per neuron id)
    assert torch.all(out[:, :, 1] <= out[:, :, 2])
    assert torch.all(out[:, :, 3] <= out[:, :, 4])

    # assert w_rec has gradients
    assert not torch.all(mod.w_rec.grad == 0)


def test_LIFTorch_noise():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=dt,
        noise_std=0.1,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # no input but vmem not zero due to noise
    assert not torch.all(rd["vmem"] == 0)


def test_LIFTorch_tau_syn_shape_1():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=dt,
        noise_std=0.1,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.tau_syn.shape == (n_neurons, n_synapses)


def test_LIFTorch_tau_syn_shape_2():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = torch.rand(n_neurons, n_synapses)
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=dt,
        noise_std=0.1,
    )

    # - Generate some data
    input_data = torch.ones(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.tau_syn.shape == (n_neurons, n_synapses)


def test_LIFTorch_threshold_shape_1():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.03
    threshold = 0.5 * torch.ones(n_neurons)
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=dt,
        threshold=threshold,
        noise_std=0.1,
    )

    # - Generate some data
    input_data = torch.zeros(n_batches, T, n_synapses * n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.threshold.shape == (n_neurons,)


def test_LIFTorch_threshold_shape_2():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    n_synapses = 1
    n_neurons = 2
    n_batches = 1
    T = 20
    tau_mem = torch.ones(n_neurons) * 0.05
    tau_syn = 0.03
    threshold = torch.Tensor([0.1, 10.0])
    dt = 1e-3

    # - Test maximal initialisation
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        dt=dt,
        threshold=threshold,
        noise_std=0.0,
    )

    # - Generate some data
    input_data = 2 * torch.ones(
        n_batches, T, n_synapses * n_neurons, requires_grad=True
    )

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)

    out.sum().backward()

    # assert correct shape
    assert mod.threshold.shape == (n_neurons,)

    print(out[:, :, 0] - out[:, :, 1])
    print(out)

    # assert output makes sense (low threshold produces higher activity)
    assert torch.all(out[:, :, 0] >= out[:, :, 1])


def test_LIFTorch_reset():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to test reset to device")

    mod = LIFTorch(10).to("cuda")
    device = mod.isyn.device
    print(device)

    mod.reset_state()
    assert mod.isyn.device == device
    assert mod.vmem.device == device

    mod.reset_parameters()
    assert mod.tau_syn.device == device
    assert mod.tau_mem.device == device
    assert mod.threshold.device == device
    assert mod.bias.device == device


def test_LIFTorch_tc_training():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    mod = LIFTorch(1, bias=Constant(0), threshold=Constant(1.0))
    assert "tau_mem" in mod.parameters().keys()
    assert "tau_syn" in mod.parameters().keys()


def test_LIFTorch_decay_training():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    mod = LIFTorch(1, bias=Constant(0), threshold=Constant(1.0), leak_mode="decays")
    assert "alpha" in mod.parameters().keys()
    assert "beta" in mod.parameters().keys()

    mod = LIFTorch(1, bias=Constant(0), threshold=Constant(1.0), leak_mode="taus")
    assert "alpha" not in mod.parameters().keys()
    assert "beta" not in mod.parameters().keys()


def test_LIFTorch_bitshift_training():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    mod = LIFTorch(
        1,
        bias=Constant(0),
        threshold=Constant(1.0),
        leak_mode="bitshifts",
    )

    assert "dash_mem" in mod.parameters().keys()
    assert "dash_syn" in mod.parameters().keys()

    mod = LIFTorch(
        1,
        bias=Constant(0),
        dash_mem=Constant(0),
        dash_syn=Constant(0.02),
        threshold=Constant(1.0),
        leak_mode="bitshifts",
    )

    assert mod.parameters() == {}


def test_LIFTorch_initialization_taus():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    try:
        tc_mod = LIFTorch(1, alpha=0.8)

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, beta=0.8)

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, dash_mem=2)

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, dash_syn=2)

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )


def test_LIFTorch_initialization_decays():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    try:
        tc_mod = LIFTorch(1, tau_mem=0.02, leak_mode="decays")

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, tau_syn=0.02, leak_mode="decays")

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, dash_mem=2, leak_mode="decays")

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, dash_syn=2, leak_mode="decays")

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )


def test_LIFTorch_initialization_bitshifts():
    from rockpool.nn.modules.torch import LIFTorch
    from rockpool.parameters import Constant

    try:
        tc_mod = LIFTorch(1, tau_mem=0.02, leak_mode="bitshifts")

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, tau_syn=0.02, leak_mode="bitshifts")

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, alpha=0.8, leak_mode="bitshifts")

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )

    try:
        tc_mod = LIFTorch(1, beta=0.8, leak_mode="bitshifts")

    except ValueError:
        pass
    else:
        raise AssertionError(
            "ValueError was not raised for wrong parameter initilization"
        )


def test_cap_spikes_keeps_dtype_and_is_never_nan():
    import torch
    from rockpool.nn.modules.torch.lif_torch import _cap_spikes

    counts = torch.tensor([0.0, 1.0, 5.0, 100.0])
    for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
        n = counts.to(dtype)

        capped = _cap_spikes(n, torch.tensor(3.0))
        assert capped.dtype == dtype
        assert capped.tolist() == [0.0, 1.0, 3.0, 3.0]

        # A cap the dtype cannot represent (2**16 in float16), and an infinite cap,
        # leave the counts unchanged
        for cap in (2.0**16, float("inf")):
            same = _cap_spikes(n, torch.tensor(cap))
            assert same.dtype == dtype
            assert torch.equal(same, n)


def test_LIFTorch_preserves_dtype():
    """After `.to(dtype)` and `reset_state()`, the state, the output and the returned
    state all have that dtype"""
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    for has_rec in (False, True):
        for dtype in (torch.float64, torch.float16):
            mod = LIFTorch(shape=(3,), has_rec=has_rec, dt=1e-3).to(dtype)
            mod.reset_state()
            assert mod.vmem.dtype == dtype
            assert mod.isyn.dtype == dtype
            assert mod.spikes.dtype == dtype

            x = (torch.rand(2, 5, 3) * 2).to(dtype)
            out, state, _ = mod(x)
            assert out.dtype == dtype
            assert state["vmem"].dtype == dtype


@pytest.mark.parametrize("dtype_name", ["float64", "bfloat16", "float16"])
def test_LIFTorch_forward_and_backward_are_finite_in_converted_dtype(dtype_name):
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    dtype = getattr(torch, dtype_name)
    mod = LIFTorch(shape=(3,), has_rec=False, dt=1e-3).to(dtype)
    mod.reset_state()

    x = (torch.rand(2, 5, 3) * 2).to(dtype).requires_grad_()
    out, state, _ = mod(x)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(s).all() for s in state.values())

    out.sum().backward()
    assert x.grad.dtype == dtype
    assert torch.isfinite(x.grad).all()
    assert mod.threshold.grad is not None
    for p in mod.parameters().values():
        if p.grad is not None:
            assert p.grad.dtype == dtype
            assert torch.isfinite(p.grad).all()


def test_LIFTorch_dtype_gradients_still_flow():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch

    mod = LIFTorch(shape=(3,), has_rec=False, dt=1e-3)
    mod.reset_state()
    x = torch.rand(2, 5, 3, requires_grad=True)
    out, _, _ = mod(x)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert mod.threshold.grad is not None


def test_LIFBitshiftTorch_preserves_dtype():
    """LIFBitshiftTorch shares the state initialisation of LIFTorch"""
    from rockpool.nn.modules.torch.lif_bitshift_torch import LIFBitshiftTorch
    import torch

    mod = LIFBitshiftTorch(shape=(3,), has_rec=False, dt=1e-3).double()
    mod.reset_state()
    x = torch.rand(2, 5, 3, dtype=torch.float64)
    out, _, _ = mod(x)
    assert out.dtype == torch.float64


def test_LIFTorch_batch_state_warns_and_keeps_only_sample_0():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import torch
    import warnings

    N = 4
    T = 20
    torch.manual_seed(0)
    sample0 = torch.rand(T, N) * 3.0
    sample1 = torch.rand(T, N) * 3.0 + 2.0

    mod = LIFTorch(N)

    # Final state of each sample, from separate single-sample calls
    finals = []
    for sample in (sample0, sample1):
        mod.reset_state()
        mod.evolve(sample[None])
        finals.append(
            {k: getattr(mod, k).detach().clone() for k in ("vmem", "isyn", "spikes")}
        )

    # The samples must differ, otherwise the test cannot tell them apart
    assert not torch.allclose(finals[0]["vmem"], finals[1]["vmem"])
    assert not torch.allclose(finals[0]["isyn"], finals[1]["isyn"])

    mod.reset_state()
    with pytest.warns(UserWarning, match="only batch index 0"):
        mod.evolve(torch.stack([sample0, sample1]))

    for name in ("vmem", "isyn", "spikes"):
        assert torch.allclose(getattr(mod, name).detach(), finals[0][name])
    assert not torch.allclose(mod.vmem.detach(), finals[1]["vmem"])
    assert not torch.allclose(mod.isyn.detach(), finals[1]["isyn"])

    # A batch of one discards nothing, so it must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LIFTorch(N).evolve(torch.rand(1, T, N))
