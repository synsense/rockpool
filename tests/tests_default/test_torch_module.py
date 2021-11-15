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


def test_LIFTorch():
    from rockpool.nn.modules.torch.lif_torch import LIFTorch
    import numpy as np
    import torch

    N = 10
    Nsyn = 2
    tau_mem = 0.01
    tau_syn = torch.Tensor([0.005, 0.015])
    mod = LIFTorch(shape=(N * Nsyn, N),
                   tau_mem=tau_mem,
                   tau_syn=tau_syn,
                   threshold=1.0,
                   has_bias=True,
                   has_rec=True,
                   noise_std=0.1,
                   learning_window=0.5,
                   dt=0.001,
                   device="cpu")
    
    # - Generate some data
    T = 100
    num_batches = 1
    input_data = (torch.rand(num_batches, T, Nsyn * N).cpu() * 100)
    input_data.requires_grad = True
    
    # - Test Rockpool interface
    out, ns, rd = mod.evolve(input_data)
    
    # - Test backward
    out.sum().backward()



def test_LIFBitshiftTorch():
    from rockpool.nn.modules.torch.lif_bitshift_torch import LIFBitshiftTorch
    import numpy as np
    import torch

    N = 10
    Nsyn = 2
    tau_mem = 0.01
    tau_syn = torch.Tensor([0.005, 0.015])
    mod = LIFBitshiftTorch(shape=(N * Nsyn, N),
                           tau_mem=tau_mem,
                           tau_syn=tau_syn,
                           threshold=1.0,
                           has_bias=True,
                           has_rec=True,
                           noise_std=0.1,
                           learning_window=0.5,
                           dt=0.001,
                           device="cpu")
    
    # - Generate some data
    T = 100
    num_batches = 1
    input_data = (torch.rand(num_batches, T, Nsyn * N).cpu() * 100)
    input_data.requires_grad = True
    
    # - Test Rockpool interface
    out, ns, rd = mod.evolve(input_data)
    
    # - Test backward
    out.sum().backward()


def test_lowpass():
    from rockpool.nn.modules.torch.lowpass import LowPass
    import torch

    N = 3
    tau_mem = 0.04

    lyr = LowPass(
        n_neurons=N,
        tau_mem=tau_mem,
        dt=0.01,
    )

    inp = torch.rand(50, 1, N).cpu()

    inp.requires_grad = True
    out, states, recs = lyr(inp, record=True)

    out.sum().backward()

    assert out.shape == inp.shape


def test_astorch():
    from rockpool.nn.modules.torch import LIFBitshiftTorch

    import torch

    N = 1
    Nsyn = 2
    tau_mem = [0.04]
    tau_syn = [0.02, 0.05]
    threshold = 10.0
    learning_window = 0.5

    lyr = LIFBitshiftTorch(shape=(N*Nsyn, N),
                           tau_mem=tau_mem,
                           tau_syn=tau_syn,
                           threshold=threshold,
                           learning_window=learning_window,
                           dt=0.01,
                           device="cpu")

    inp = torch.rand(1, 50, Nsyn * N).cpu()

    params = lyr.parameters()
    params_astorch = params.astorch()

    torch_params = torch.nn.Module.parameters(lyr)

    for (r_param, t_param) in zip(params_astorch, torch_params):
        assert r_param is t_param, "Rockpool and torch parameters do not match."


def test_wavesense_import():
    from rockpool.nn.networks import WaveSenseNet



def test_wavesense_init():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3 
    n_neurons = 4 
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"
    
    # model init
    model = WaveSenseNet(dilations=dilations,
                         n_classes=n_out_neurons,
                         n_channels_in=n_inp_neurons,
                         n_channels_res=n_neurons,
                         n_channels_skip=n_neurons,
                         n_hidden=n_neurons,
                         kernel_size=kernel_size,
                         has_bias=True,
                         smooth_output=True,
                         tau_mem=tau_mem,
                         base_tau_syn=base_tau_syn,
                         tau_lp=tau_lp,
                         threshold=threshold,
                         neuron_model=LIFTorch,
                         dt=dt,
                         device=device)


def test_wavesense_forward():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3 
    n_neurons = 4 
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"
    
    # model init
    model = WaveSenseNet(dilations=dilations,
                         n_classes=n_out_neurons,
                         n_channels_in=n_inp_neurons,
                         n_channels_res=n_neurons,
                         n_channels_skip=n_neurons,
                         n_hidden=n_neurons,
                         kernel_size=kernel_size,
                         has_bias=True,
                         smooth_output=True,
                         tau_mem=tau_mem,
                         base_tau_syn=base_tau_syn,
                         tau_lp=tau_lp,
                         threshold=threshold,
                         neuron_model=LIFTorch,
                         dt=dt,
                         device=device)

    # input params
    n_batches = 2
    T = 20

    # input 
    inp = torch.rand(n_batches, T, n_inp_neurons) * 100

    # forward
    out, state, rec = model(inp)

    assert(len(rec) == 0)
    assert(torch.any(out > 0))


def test_wavesense_record():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3 
    n_neurons = 4 
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"
    
    # model init
    model = WaveSenseNet(dilations=dilations,
                         n_classes=n_out_neurons,
                         n_channels_in=n_inp_neurons,
                         n_channels_res=n_neurons,
                         n_channels_skip=n_neurons,
                         n_hidden=n_neurons,
                         kernel_size=kernel_size,
                         has_bias=True,
                         smooth_output=True,
                         tau_mem=tau_mem,
                         base_tau_syn=base_tau_syn,
                         tau_lp=tau_lp,
                         threshold=threshold,
                         neuron_model=LIFTorch,
                         dt=dt,
                         device=device)

    # input params
    n_batches = 2
    T = 20

    # input 
    inp = torch.rand(n_batches, T, n_inp_neurons) * 10

    # forward
    out, state, rec = model(inp, record=True)

    assert(len(rec) > 0)



def test_wavesense_backward():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch

    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3 
    n_neurons = 4 
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"
    
    # model init
    model = WaveSenseNet(dilations=dilations,
                         n_classes=n_out_neurons,
                         n_channels_in=n_inp_neurons,
                         n_channels_res=n_neurons,
                         n_channels_skip=n_neurons,
                         n_hidden=n_neurons,
                         kernel_size=kernel_size,
                         has_bias=True,
                         smooth_output=True,
                         tau_mem=tau_mem,
                         base_tau_syn=base_tau_syn,
                         tau_lp=tau_lp,
                         threshold=threshold,
                         neuron_model=LIFTorch,
                         dt=dt,
                         device=device)

    # input params
    n_batches = 2
    T = 20

    # input 
    inp = torch.rand(n_batches, T, n_inp_neurons) * 10
    inp.requires_grad = True

    # forward
    out, state, rec = model(inp)

    # backward
    out.sum().backward()

    assert(not torch.all(inp.grad == 0))



def test_wavesense_save_load():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch
    import os
    
    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3 
    n_neurons = 4 
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"
    
    # model init
    model = WaveSenseNet(dilations=dilations,
                         n_classes=n_out_neurons,
                         n_channels_in=n_inp_neurons,
                         n_channels_res=n_neurons,
                         n_channels_skip=n_neurons,
                         n_hidden=n_neurons,
                         kernel_size=kernel_size,
                         has_bias=True,
                         smooth_output=True,
                         tau_mem=tau_mem,
                         base_tau_syn=base_tau_syn,
                         tau_lp=tau_lp,
                         threshold=threshold,
                         neuron_model=LIFTorch,
                         dt=dt,
                         device=device)
    
    model2 = WaveSenseNet(dilations=dilations,
                          n_classes=n_out_neurons,
                          n_channels_in=n_inp_neurons,
                          n_channels_res=n_neurons,
                          n_channels_skip=n_neurons,
                          n_hidden=n_neurons,
                          kernel_size=kernel_size,
                          has_bias=True,
                          smooth_output=True,
                          tau_mem=tau_mem,
                          base_tau_syn=base_tau_syn,
                          tau_lp=tau_lp,
                          threshold=threshold,
                          neuron_model=LIFTorch,
                          dt=dt,
                          device=device)
    
    # input params
    n_batches = 2
    T = 20
    
    # input 
    inp = torch.rand(n_batches, T, n_inp_neurons) * 10
    inp.requires_grad = True
    
    # forward
    out, _, _ = model(inp)
    
    # forward model 2
    out2, _, _ = model2(inp)
    
    
    # assert not all outputs are equal
    assert(not torch.all(out == out2))
    
    # save model
    model.save("tmp.json")
    
    # load parameters to model2
    model2.load("tmp.json")
    
    # forward model 2
    model2.reset_state()
    out2, _, _ = model2(inp)
    
    # assert all outputs are equal
    assert(torch.all(out == out2))
    
    # cleanup 
    os.remove("tmp.json")
    
    
def test_wavesense_reset():
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import LIFTorch
    import torch
    
    # model params
    dilations = [2, 16]
    n_out_neurons = 2
    n_inp_neurons = 3 
    n_neurons = 4 
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"
    
    # model init
    model = WaveSenseNet(dilations=dilations,
                         n_classes=n_out_neurons,
                         n_channels_in=n_inp_neurons,
                         n_channels_res=n_neurons,
                         n_channels_skip=n_neurons,
                         n_hidden=n_neurons,
                         kernel_size=kernel_size,
                         has_bias=True,
                         smooth_output=True,
                         tau_mem=tau_mem,
                         base_tau_syn=base_tau_syn,
                         tau_lp=tau_lp,
                         threshold=threshold,
                         neuron_model=LIFTorch,
                         dt=dt,
                         device=device)
    
    # input params
    n_batches = 2
    T = 20
    
    # input 
    inp = torch.rand(n_batches, T, n_inp_neurons) * 10
    
    # forward
    out, state, rec = model(inp)
    
    # assert first spk layers state is not reset
    assert(not torch.all(state['spk1']['vmem'] == 0))
    
    # assert first spk layers state of first wave block is not reset
    assert(not torch.all(state['wave0']['spk1']['vmem'] == 0))
    
    model.reset_state()
    
    # get state
    state = model.state()
    
    # assert first spk layers state is reset
    assert(torch.all(state['spk1']['vmem'] == 0))
    
    # assert first spk layers state of first wave block is reset
    assert(torch.all(state['wave0']['spk1']['vmem'] == 0))
    
