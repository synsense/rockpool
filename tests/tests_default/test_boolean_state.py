import pytest

pytest.importorskip("torch")


def test_BooleanState():
    from rockpool.nn.modules.torch.bool_state import BooleanState
    import torch

    n_batches = 2
    n_neurons = 3
    T = 5

    # test boolean state with default threshold
    mod = BooleanState(shape=(n_neurons))

    # - Generate some data
    input_data = torch.rand((n_batches, T, n_neurons)) - 0.5
    input_data[input_data < -0.2] = 0
    input_data[input_data > 0.3] = 0

    # - Test Rockpool interface
    eps = 1e-12
    out, ns, rd = mod(input_data, record=True)
    for b in range(n_batches):
        for t in range(T):
            for n in range(n_neurons):
                if input_data[b, t, n] > eps:
                    assert out[b, t, n]
                elif input_data[b, t, n] < -eps:
                    assert not out[b, t, n]
                # elif input_data[b, t, n] == 0 and t > 0:
                elif t > 0:
                    assert out[b, t, n] == out[b, t - 1, n]

    # test boolean state with custom threshold
    thresholds = torch.Tensor([0.1, 0.3, 0.15])
    mod = BooleanState(shape=(n_neurons), threshold=thresholds)

    # - Generate some data
    input_data = torch.rand((n_batches, T, n_neurons)) - 0.5

    # - Test Rockpool interface
    out, ns, rd = mod(input_data, record=True)
    for b in range(n_batches):
        for t in range(T):
            for n in range(n_neurons):
                if input_data[b, t, n] > thresholds[n]:
                    assert out[b, t, n]
                elif input_data[b, t, n] < -thresholds[n]:
                    assert not out[b, t, n]
                elif t > 0:
                    assert out[b, t, n] == out[b, t - 1, n]
