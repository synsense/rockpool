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



def test_TorchLIF():
    from rockpool.nn.modules.torch.lif_torch import LIFLayer
    import numpy as np
    import torch
    
    N = 10
    Nsyn = 2
    tau_mem = 2 * np.ones(N,)
    tau_syn = torch.Tensor([2, 8])
    tau_syn = tau_syn.view(1, Nsyn).T.repeat(1, N)
    mod = LIFLayer(
        n_neurons=N,
        n_synapses=Nsyn,
        batch_size=1,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=1.0,
        learning_window=0.5,
        device="cpu",
    )
    
    # - Generate some data
    T = 100
    num_batches = 1
    input_data = torch.from_numpy(np.random.rand(T, num_batches, Nsyn, N)).cpu()
    
    # - Test torch interface
    out = mod.forward(input_data)
    
    # - Test Rockpool interface
    out, ns, rd = mod.evolve(input_data)


def test_single_neuron():
    from rockpool.nn.modules.torch.lif_torch import LIFLayer
    import numpy as np
    import torch

    N = 1
    Nsyn = 2
    tau_mem = [0.04]
    tau_syn = [[0.02], [0.03]]
    threshold = [10.0]
    learning_window = [0.5]

    lyr = LIFLayer(
        n_neurons=N,
        n_synapses=Nsyn,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
        learning_window=learning_window,
        batch_size=1,
        dt=0.01,
        device="cpu",
    )

    inp = torch.zeros((10, 1, 2, 1)).cpu()
    inp[1, :, :, :] = 1
    out, states, recs = lyr(inp, record=True)

def test_backward():
    from rockpool.nn.modules.torch.lif_torch import LIFLayer
    import numpy as np
    import torch
    
    N = 1
    Nsyn = 2
    tau_mem = [0.04]
    tau_syn = [[0.02]]
    threshold = [10.0]
    learning_window = [0.5]
    
    lyr = LIFLayer(
        n_neurons=N,
        n_synapses=Nsyn,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
        learning_window=learning_window,
        batch_size=1,
        dt=0.01,
        device="cpu",
    )
    
    inp = torch.rand(50, 1, Nsyn, N).cpu()
    
    inp.requires_grad = True
    out, states, recs = lyr(inp, record=True)
    
    out.sum().backward()


