import pytest

pytest.importorskip("torch")


def test_Dropout_shapes():
    from rockpool.nn.modules.torch.dropout import UnitDropout, TimeStepDropout
    import torch

    n_neurons = 45
    T = 67
    n_batches = 9

    mod1 = UnitDropout(shape=(n_neurons), p=0.2)
    mod2 = TimeStepDropout(shape=(n_neurons), p=0.2)

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod1(input_data)
    assert out.shape == (n_batches, T, n_neurons)
    out, ns, rd = mod2(input_data)
    assert out.shape == (n_batches, T, n_neurons)


def test_Dropout_all():
    from rockpool.nn.modules.torch.dropout import TimeStepDropout
    import torch

    n_neurons = 35
    T = 27
    n_batches = 54

    mod = TimeStepDropout(shape=(n_neurons), p=1.0)

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_neurons, requires_grad=True)

    # - Test Rockpool interface
    out, ns, rd = mod(input_data)
    assert torch.sum(out) == 0.0


def test_UnitDropout():
    from rockpool.nn.modules.torch.dropout import UnitDropout
    import torch

    n_neurons = 35
    T = 27
    n_batches = 4
    p = 0.5

    mod = UnitDropout(shape=(n_neurons), p=p)
    input_data = torch.rand(n_batches, T, n_neurons, requires_grad=True)

    # the output of individual neurons is either dropped out for all time steps on not at all
    out, ns, rd = mod(input_data)
    for b in range(n_batches):
        for n in range(n_neurons):
            output_dropped = torch.sum(out[b, :, n]) == 0
            output_not_dropped = torch.allclose(
                out[b, :, n], input_data[b, :, n] * 1 / (1 - p)
            )
            assert output_dropped or output_not_dropped


def test_TimeStepDropout():
    from rockpool.nn.modules.torch.dropout import TimeStepDropout
    import torch

    n_neurons = 12
    T = 76
    n_batches = 3
    p = 0.4

    mod = TimeStepDropout(shape=(n_neurons), p=p)
    input_data = torch.rand(n_batches, T, n_neurons, requires_grad=True)

    # the output of time steps that are not dropped are the same as the input
    out, ns, rd = mod(input_data)
    assert torch.allclose(out[out != 0], input_data[out != 0])


def test_dropout_evaluation_mode():
    from rockpool.nn.modules.torch.dropout import TimeStepDropout, UnitDropout
    import torch

    n_neurons = 12
    T = 76
    n_batches = 3
    p = 0.4

    mod = TimeStepDropout(shape=(n_neurons), p=p)
    input_data = torch.rand(n_batches, T, n_neurons)
    mod.eval()
    out_eval1, _, _ = mod(input_data)
    mod.train()
    out_train, _, _ = mod(input_data)
    mod.eval()
    out_eval2, _, _ = mod(input_data)

    # dropout should be disabled in evaluation mode
    assert torch.allclose(out_eval1, out_eval2)
    # dropout is applied in training mode
    assert not torch.allclose(out_eval1, out_train)

    mod = UnitDropout(shape=(n_neurons), p=p)
    input_data = torch.rand(n_batches, T, n_neurons)
    mod.eval()
    out_eval1, _, _ = mod(input_data)
    mod.train()
    out_train, _, _ = mod(input_data)
    mod.eval()
    out_eval2, _, _ = mod(input_data)

    # dropout should be disabled in evaluation mode
    assert torch.allclose(out_eval1, out_eval2)
    # dropout is applied in training mode
    assert not torch.allclose(out_eval1, out_train)
