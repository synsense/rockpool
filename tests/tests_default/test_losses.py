import pytest


def test_peak_loss():
    import torch
    from rockpool.nn.losses import PeakLoss

    n_classes = 3
    T = 35
    batch_size = 5 * 3
    max_interval = 4
    target_output = 1.5

    mse = torch.nn.MSELoss()

    # create random target signal
    targets = torch.randint(0, n_classes, (batch_size,))

    ## check loss of target neuron

    # pick event time such that the loss window fits within the sample
    pl = PeakLoss(
        max_interval=max_interval, weight_nontarget=1.0, target_output=target_output
    )
    tmax = 5
    vmem = torch.rand((batch_size, T, n_classes))
    vmem[:, tmax, targets] = torch.max(vmem) + 1e-5
    # check loss of target neuron
    vmem_max_target = torch.ones(batch_size) * (torch.max(vmem) + 1e-5)
    for i in range(batch_size):
        vmem_max_target[i] = torch.mean(
            vmem[i, tmax : tmax + max_interval, targets[i]], axis=0
        )
    loss_target = mse(vmem_max_target, target_output * torch.ones(batch_size))
    pl(vmem, targets)
    assert torch.allclose(loss_target, pl.loss_target)

    # pick event time as the second last time point and window size three, thus the output at event time should be
    # counted twice
    pl = PeakLoss(max_interval=3, weight_nontarget=1.0, target_output=target_output)
    vmem = torch.rand((batch_size, T, n_classes))
    vmem[:, -2, targets] = torch.max(vmem) + 1e-5
    vmem_max_target = torch.ones(batch_size) * (torch.max(vmem) + 1e-5)
    for i in range(batch_size):
        vmem_max_target[i] = (
            2 * vmem[i, -2, targets[i]] + vmem[i, -1, targets[i]]
        ) / 3.0
    loss_target = mse(vmem_max_target, target_output * torch.ones(batch_size))
    pl(vmem, targets)
    assert torch.allclose(loss_target, pl.loss_target)

    ## check loss of non-target neurons

    # pick event time such that the loss window fits within the sample
    max_interval = T
    pl = PeakLoss(max_interval=max_interval, weight_nontarget=1.0, target_output=0.0)
    tmax = 0
    vmem = 0.5 * torch.ones((batch_size, T, n_classes))
    # check loss of target neuron
    vmem_max_target = torch.ones(batch_size) * (torch.max(vmem) + 1e-5)
    for i in range(batch_size):
        vmem_max_target[i] = torch.mean(
            vmem[i, tmax : tmax + max_interval, targets[i]], axis=0
        )
    pl(vmem, targets)
    assert torch.allclose(pl.loss_nontarget, pl.loss_target)


def test_binary_peak_loss():
    import torch
    from rockpool.nn.losses import BinaryPeakLoss

    T = 35
    batch_size = 5 * 3
    max_interval = 4
    target_output = 1.5

    mse = torch.nn.MSELoss()

    # create random target signal
    targets = torch.randint(0, 2, (batch_size,))
    n_positives = sum(targets == 1)
    n_negatives = sum(targets == 0)

    ## check loss of positive samples

    # pick event time such that the loss window fits within the sample
    pl = BinaryPeakLoss(
        max_interval=max_interval, weight_nontarget=1.0, target_output=target_output
    )
    tmax = 5
    vmem = torch.rand((batch_size, T, 1))
    vmem[targets == 1, tmax, 0] = torch.max(vmem) + 1e-5
    # check loss of positive samples
    vmem_max_positives = torch.ones(n_positives) * (torch.max(vmem) + 1e-5)
    for i in range(n_positives):
        vmem_max_positives[i] = torch.mean(
            vmem[targets == 1, tmax : tmax + max_interval, 0][i], axis=0
        )
    pl(vmem, targets)
    loss_positives = mse(vmem_max_positives, target_output * torch.ones(n_positives))
    assert torch.allclose(loss_positives, pl.loss_positives)

    # pick event time as the second last time point and window size three, thus the output at event time should be
    # counted twice
    pl = BinaryPeakLoss(
        max_interval=3, weight_nontarget=1.0, target_output=target_output
    )
    vmem = torch.rand((batch_size, T, 1))
    vmem[targets == 1, -2, 0] = torch.max(vmem) + 1e-5
    vmem_max_positives = torch.ones(n_positives) * (torch.max(vmem) + 1e-5)
    for i in range(n_positives):
        vmem_max_positives[i] = (
            2 * vmem[targets == 1, -2, 0][i] + vmem[targets == 1, -1, 0][i]
        ) / 3.0
    loss_positives = mse(vmem_max_positives, target_output * torch.ones(n_positives))
    pl(vmem, targets)
    assert torch.allclose(loss_positives, pl.loss_positives)

    ## check loss of negative samples

    # pick event time such that the loss window fits within the sample
    max_interval = T
    pl = BinaryPeakLoss(
        max_interval=max_interval, weight_nontarget=1.0, target_output=0.0
    )
    tmax = 0
    vmem = 0.5 * torch.ones((batch_size, T, 1))
    # check loss of target neuron
    vmem_max_positives = torch.ones(n_positives) * (torch.max(vmem) + 1e-5)
    for i in range(n_positives):
        vmem_max_positives[i] = torch.mean(
            vmem[targets == 1, tmax : tmax + max_interval, 0][i], axis=0
        )
    pl(vmem, targets)
    assert torch.allclose(pl.loss_negatives, pl.loss_positives)


def test_mse_loss():
    import torch
    from rockpool.nn.losses import MSELoss

    n_classes = 3
    T = 35
    batch_size = 5 * 3

    mse = torch.nn.MSELoss()

    mse_rockpool = MSELoss()
    vmem = torch.rand((batch_size, T, n_classes))
    # create random target signal
    targets = torch.rand(size=vmem.size())
    assert torch.allclose(mse(vmem, targets), mse_rockpool(vmem, targets))
