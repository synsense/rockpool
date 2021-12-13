import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch
import numpy as np


class TemporalXOR(Dataset):
    def __init__(
        self,
        T_total=100,
        T_stim=20,
        T_delay=40,
        T_randomize=20,
        max_num_spikes=15,
        num_channels=16,
    ):
        self.T_total = T_total
        self.T_stim = T_stim
        self.T_delay = T_delay
        self.T_randomize = T_randomize
        self.max_num_spikes = max_num_spikes
        self.num_channels = num_channels

        # two different stimuli
        self.inp_A = torch.randint(
            0, self.max_num_spikes + 1, (self.T_stim, self.num_channels)
        )
        self.inp_A[:, : self.num_channels // 2] = 0
        self.inp_B = torch.randint(
            0, self.max_num_spikes + 1, (self.T_stim, self.num_channels)
        )
        self.inp_B[:, self.num_channels // 2 :] = 0

        # stimuli sequence for logical XOR
        self.key_stim_map = {
            0: [self.inp_A, self.inp_A],
            1: [self.inp_A, self.inp_B],
            2: [self.inp_B, self.inp_A],
            3: [self.inp_B, self.inp_B],
        }

        # supervision signal for logical XOR
        self.key_target_map = {0: 0, 1: 1, 2: 1, 3: 0}

    def __getitem__(self, key):
        # generate input sample as
        # [FIRST STIM, ... silence ..., SECOND STIM, ... silence ...]
        inp = torch.zeros(self.T_total, self.num_channels)
        inp[: self.T_stim] = self.key_stim_map[key][0]
        T_second_stim = (
            self.T_stim
            + self.T_delay
            - np.random.randint(-self.T_randomize, self.T_randomize)
        )
        inp[T_second_stim : T_second_stim + self.T_stim] = self.key_stim_map[key][1]

        # supervision signal
        tgt = torch.Tensor([self.key_target_map[key]]).long()

        return inp, tgt

    def __len__(self):
        return len(self.key_stim_map)


from torch.utils.data import DataLoader

data = TemporalXOR(
    T_total=100,
    T_stim=20,
    T_delay=30,
    T_randomize=20,
    max_num_spikes=15,
    num_channels=16,
)
dataloader = DataLoader(data, batch_size=len(data), shuffle=True)

from rockpool.nn.modules import LIFBitshiftTorch, LIFTorch
from rockpool.nn.networks.wavesense import WaveSenseNet

dilations = [2, 16]
n_out_neurons = 2
n_inp_neurons = data.num_channels
n_neurons = 16
kernel_size = 2
tau_mem = 0.002
base_tau_syn = 0.002
tau_lp = 0.01
threshold = 1.0
dt = 0.001
device = "cpu"  # feel free to use cuda if you have a GPU

model = WaveSenseNet(
    dilations=dilations,
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
    device=device,
)

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

crit = CrossEntropyLoss()
opt = Adam(model.parameters().astorch(), lr=5e-3)

from sklearn.metrics import accuracy_score

num_epochs = 500

# save loss and accuracy over epochs
losses = []
accs = []

# loop over epochs
for epoch in range(num_epochs):

    # read one batch of the data
    for inp, tgt in dataloader:

        # reset states and gradients
        model.reset_state()
        opt.zero_grad()

        # forward path
        out, _, rec = model(inp.to(device), record=True)
        #         print(np.max(out[:, -1].detach().cpu().numpy()))

        # get the last timestep of the output
        out_at_last_timestep = out[:, -1]

        # pass the last timestep of the output and the target through CE
        loss = crit(out_at_last_timestep, tgt.squeeze().to(device))

        # backward
        loss.backward()

        # apply gradients
        opt.step()

        # save loss and accuracy
        with torch.no_grad():
            pred = out_at_last_timestep.argmax(1)
            accs.append(accuracy_score(tgt.squeeze().cpu().numpy(), pred.cpu().numpy()))
            losses.append(loss.item())
        #         print("loss", loss)
        #         print('acc', accuracy_score(tgt.squeeze().cpu().numpy(), pred.cpu().numpy()))
        # print loss and accuracy every 10th epoch
        if epoch % 10 == 0:
            print("epoch", epoch, "loss: ", losses[-1], "acc: ", accs[-1])
