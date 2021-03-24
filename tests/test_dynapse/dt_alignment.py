# ----
# dt_alignment.py - Test whether badly aligned time steps in Dynapse (fpga_isibase)
#                   and input timeseries result in wrongly timed input events due to
#                   rounding. If not, the plot in the end should be well aligned.
# ----

import numpy as np
from matplotlib import pyplot as plt

from rockpool.devices import rectangular_neuron_arrangement, DynapseControlExtd
from rockpool import TSEvent

plt.ion()

# Path for loading circuit biases (which define neuron and synapse characteristics)
bias_path = "files/biases.py"

# Input weights
weights_in = np.zeros((4, 128))
for i in range(4):
    weights_in[i, 32 * i : 32 * (i + 1)] = 8

# - Reservoir neuron arangement
neuron_ids = rectangular_neuron_arrangement(first_neuron=4, num_neurons=128, width=8)

# - 'Virtual' neurons (input neurons)
virtual_neuron_ids = [1, 2, 3, 12]

# - Set up DynapseControl

controller = DynapseControlExtd(fpga_isibase=0.0007)

# Circuit biases
controller.load_biases(bias_path)

# - Connections
controller.set_connections_from_weights(
    weights_in,
    neuron_ids=virtual_neuron_ids,
    neuron_ids_post=neuron_ids,
    virtual_pre=True,
)

# - Input spikes
spike_times = np.arange(5000) * 0.001  # kwargs_reservoir["dt"]
spike_channels_single_rep = 250 * [0] + 250 * [1] + 250 * [2] + 250 * [3]
spike_channels = 5 * spike_channels_single_rep
ts_input = TSEvent(spike_times, spike_channels)

# - Send stimulus
rec = controller.send_TSEvent(
    ts_input,
    virtual_neur_ids=virtual_neuron_ids,
    record_neur_ids=neuron_ids,
    record=True,
    return_ts=True,
)

# - Plot stimulus and activation -> should be aligned in spite of mismatch in dt
fig, ax = plt.subplots()
rec.plot(target=ax, s=1)
ts_input.plot(target=ax, s=1, label="Input")
plt.legend()
