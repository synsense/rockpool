import pytest


def test_FFUpDownTorch():
    from rockpool.nn.modules.torch.updown_torch import FFUpDownTorch
    import torch

    n_channels = 10
    n_batches = 3
    T = 20
    thr_up = torch.rand(n_channels)
    thr_down = torch.rand(n_channels)
    models = []
    # - Test minimal initialisation
    models.append(FFUpDownTorch((n_channels,)))

    # - Test maximal initialisation
    models.append(FFUpDownTorch(
        shape=(n_channels,),
        thr_up=thr_up,
        thr_down=thr_down,
        n_ref_steps = 1,
        repeat_output=1,
        dt=1e-3,
        device=None,
        dtype=None,
    ))

    # - Generate some data
    input_data = torch.rand(n_batches, T, n_channels, requires_grad=True)

    # - Test the different initialisations
    for mod in models:
        # - Test torch interface
        out = mod.forward(input_data)

        out.sum().backward()

        # - Test Rockpool interface
        out, ns, rd = mod(input_data, record=True)

        assert out.shape == (n_batches, T, 2*n_channels)
