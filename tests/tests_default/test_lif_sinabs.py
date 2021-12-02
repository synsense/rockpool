
#def test_FF_equality():
import torch

# - parameter
n_synapses = 1
n_neurons = 10
n_batches = 3
T = 20
tau_mem = torch.rand(n_neurons)
tau_syn = 0.05

# - init LIFTorch 

from rockpool.nn.modules.torch.lif_torch import LIFTorch
lif_torch  = LIFTorch(
    shape=(n_synapses * n_neurons, n_neurons),
    tau_mem=tau_mem,
    tau_syn=tau_syn,
    has_bias=False,
    has_rec=False,
    dt=1e-3,
    noise_std=0.0,
    device="cpu",
)


# - init LIFSinabs

from rockpool.nn.modules.torch.lif_sinabs import LIFSinabs
lif_sinabs = LIFSinabs(
    shape=(n_synapses * n_neurons, n_neurons),
    tau_mem=tau_mem,
    tau_syn=tau_syn,
    has_bias=False,
    has_rec=False,
    dt=1e-3,
    noise_std=0.0,
    device="cpu",
)


# - Generate some data
input_data = torch.rand(n_batches, T, n_synapses * n_neurons, requires_grad=True)

# - run LIFTorch and LIFSinabs
out_torch, ns_torch, rd_torch = lif_torch(input_data, record=True)
out_sinabs, ns_sinabs, rd_sinabs = lif_sinabs(input_data, record=True)

assert torch.all(out_torch == out_sinabs)
assert torch.all(ns_torch == ns_sinabs)
assert torch.all(rd_torch == rd_sinabs)
