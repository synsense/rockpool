import pytest

pytest.importorskip("torch")


def test_imports():
    from rockpool.nn.modules.torch.torch_module import TorchModule


def test_torch_to_rockpool():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from rockpool.nn.modules.torch.torch_module import TorchModule

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            # First 2D convolutional layer, taking in 1 input channel (image),
            # outputting 32 convolutional features, with a square kernel size of 3
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            # Second 2D convolutional layer, taking in the 32 input layers,
            # outputting 64 convolutional features, with a square kernel size of 3
            self.conv2 = nn.Conv2d(32, 64, 3, 1)

            # Designed to ensure that adjacent pixels are either all 0s or all active
            # with an input probability
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)

            # First fully connected layer
            self.fc1 = nn.Linear(9216, 128)
            # Second fully connected layer that outputs our 10 labels
            self.fc2 = nn.Linear(128, 10)

        # x represents our data
        def forward(self, x):
            # Pass data through conv1
            x = self.conv1(x)
            # Use the rectified-linear activation function over x
            x = F.relu(x)

            x = self.conv2(x)
            x = F.relu(x)

            # Run max pooling over x
            x = F.max_pool2d(x, 2)
            # Pass data through dropout1
            x = self.dropout1(x)
            # Flatten x with start_dim=1
            x = torch.flatten(x, 1)
            # Pass data through fc1
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)

            # Apply softmax to x
            output = F.log_softmax(x, dim=1)
            return output

    # Equates to one random 28x28 image
    random_data = torch.rand((1, 1, 28, 28))

    # - Generate torch module and test forward
    mod = Net()
    result = mod(random_data)
    print(result)

    # - Convert in-place to Rockpool.TorchModule
    TorchModule.from_torch(mod)

    # - Test Rockpool parameters interface
    p = mod.parameters()
    s = mod.state()
    sp = mod.simulation_parameters()
    print(p, s, sp)

    # - Test Rockpool evolution
    o, ns, rd = mod(random_data)

    print(o, ns, rd)


def test_TorchModule():
    from rockpool.nn.modules.torch.torch_module import TorchModule

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(TorchModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # First 2D convolutional layer, taking in 1 input channel (image),
            # outputting 32 convolutional features, with a square kernel size of 3
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            # Second 2D convolutional layer, taking in the 32 input layers,
            # outputting 64 convolutional features, with a square kernel size of 3
            self.conv2 = nn.Conv2d(32, 64, 3, 1)

            # Designed to ensure that adjacent pixels are either all 0s or all active
            # with an input probability
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)

            # First fully connected layer
            self.fc1 = nn.Linear(9216, 128)
            # Second fully connected layer that outputs our 10 labels
            self.fc2 = nn.Linear(128, 10)

        # x represents our data
        def forward(self, x):
            # Pass data through conv1
            x = self.conv1(x)
            # Use the rectified-linear activation function over x
            x = F.relu(x)

            x = self.conv2(x)
            x = F.relu(x)

            # Run max pooling over x
            x = F.max_pool2d(x, 2)
            # Pass data through dropout1
            x = self.dropout1(x)
            # Flatten x with start_dim=1
            x = torch.flatten(x, 1)
            # Pass data through fc1
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)

            # Apply softmax to x
            output = F.log_softmax(x, dim=1)
            return output

    # Equates to one random 28x28 image
    random_data = torch.rand((1, 1, 28, 28))

    # - Build net as TorchModule from scratch
    mod = Net()

    # - Test Rockpool parameters interface
    p = mod.parameters()
    s = mod.state()
    sp = mod.simulation_parameters()
    print(p, s, sp)

    # - Test evolution interface
    o, ns, rd = mod(random_data)

    print(o, ns, rd)


def test_lowpass():
    from rockpool.nn.modules.torch.lowpass import LowPass
    import torch

    N = 3
    tau_mem = 0.04

    lyr = LowPass(n_neurons=N, tau_mem=tau_mem, dt=0.01)

    inp = torch.rand(50, 1, N).cpu()

    inp.requires_grad = True
    out, states, recs = lyr(inp, record=True)

    out.sum().backward()

    assert out.shape == inp.shape


def test_astorch():
    from rockpool.nn.modules import LIFBitshiftTorch

    import torch

    N = 1
    Nsyn = 2
    tau_mem = 0.04
    tau_syn = torch.Tensor([[0.02, 0.05]])
    threshold = 10.0
    learning_window = 0.5

    lyr = LIFBitshiftTorch(
        shape=(N * Nsyn, N),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
        learning_window=learning_window,
        dt=0.01,
    )

    inp = torch.rand(1, 50, Nsyn * N).cpu()

    params = lyr.parameters()
    params_astorch = params.astorch()

    torch_params = torch.nn.Module.parameters(lyr)

    for r_param, t_param in zip(params_astorch, torch_params):
        assert r_param is t_param, "Rockpool and torch parameters do not match."


def test_API_semantics():
    import torch
    from torch import nn
    from rockpool.nn.modules import TorchModule
    from rockpool.nn.modules import LIFTorch

    from collections import OrderedDict
    from typing import Generator

    class A(TorchModule):
        def __init__(self):
            super().__init__()
            self.slayer = LIFTorch(shape=(1, 1), tau_mem=0.3, tau_syn=0.1)
            self.lin = nn.Linear(1, 1)
            # self.conv = nn.Conv1d(1, 1, 1)

        def forward(self, data):
            x, _, _ = self.slayer(data)
            return self.lin(x)

    class B(TorchModule):
        def __init__(self):
            super().__init__()
            self.a = A()

        def forward(self, data):
            out, _, _ = self.a(data)
            return out

    # - Generate module
    b = B()

    # - Check `modules` call
    assert isinstance(b.modules(), OrderedDict)

    # - Check Rockpool call semantics
    out = b(torch.zeros((1, 1, 1)))
    assert len(out) == 3

    # - Check Rockpool `parameters` call
    assert isinstance(b.parameters(), dict)

    # - Check __repr__
    repr_Rockpool = "B  with shape (None,) {\n    A 'a' with shape (None,) {\n        LIFTorch 'slayer' with shape (1, 1)\n        Linear 'lin' with shape (None,)\n    }\n}"

    assert str(b) == repr_Rockpool

    # - Convert to torch
    b.to_torch()

    # - Check torch.nn call semantics
    out = b(torch.zeros((1, 1, 1)))
    assert len(out) == 1

    # - Check `modules` call
    assert isinstance(b.modules(), Generator)

    # - Check torch.nn `parameters` semantics
    assert isinstance(b.parameters(), Generator)

    # - Check __repr__
    repr_torch = "B(\n  (a): A(\n    (slayer): LIFTorch()\n    (lin): Linear 'lin' with shape (None,)\n  )\n)"
    assert str(b) == repr_torch


def test_torch_device_to():
    from rockpool.nn.modules import TorchModule
    from rockpool.parameters import Parameter, SimulationParameter, State, Constant
    import torch

    if not torch.cuda.is_available():
        return

    class TestModule(TorchModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.test_param = Parameter(torch.zeros((2,)))
            self.test_const_param = Parameter(Constant(torch.zeros((2,))))
            self.test_simparam = SimulationParameter(torch.zeros((2,)))
            self.test_state = State(torch.zeros((2,)))

    mod = TestModule().to_torch().to("cuda")

    print("Checking devices post-initialisation")
    assert mod.test_param.device != "cpu", "Parameter was not moved to GPU"
    assert mod.test_const_param != "cpu", "Constant parameter was not moved to GPU"
    assert mod.test_simparam != "cpu", "SimulationParameter was not moved to GPU"
    assert mod.test_state != "cpu", "State was not moved to GPU"

    mod.reset_state()
    mod.reset_parameters()

    print("Checking devices post-reset")
    assert mod.test_param.device != "cpu", "Parameter was not moved to GPU"
    assert mod.test_const_param != "cpu", "Constant parameter was not moved to GPU"
    assert mod.test_simparam != "cpu", "SimulationParameter was not moved to GPU"
    assert mod.test_state != "cpu", "State was not moved to GPU"
