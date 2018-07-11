from matplotlib import pyplot as plt
plt.ion()

import numpy as np

from brian2 import second, amp, farad
import brian2 as b2

from teili.core.groups import Neurons, Connections
from teili import teiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.synapse_models import DPISyn as syn_model
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

tDt = 0.0001 * second

nNumInputSamples = 0
tInputDuration = 1

# - Corrected constant parameters
dParamsNeuron = {
    'Io' : 1.5e-12 * amp,
    'Cmem' : 2e-12 * farad,
    'Ispkthr' : 1e-5 * amp,
    'Ireset' : 0 * amp,
    'Ith' : 2e-9 * amp,
    'Iagain' : 1e-5 * amp,
    'Iconst' : 7e-9 * amp,
    'Iath' : 1e-5 * amp,
    # 'Ica' : 10e-12 * amp,
}

neuron_model_param.update(Iconst=4.375e-9 * amp)
corrected_neuron_model = NeuronEquationBuilder.import_eq(
    'teili/models/equations/DPIexp'
)
# - Neuron
neuron = Neurons(N=1, equation_builder=neuron_model(num_inputs=1),
                       name="testNeuron")
neuron1 = Neurons(N=1, equation_builder=neuron_model(num_inputs=1),
                       name="testNeuron1")
neuron2 = Neurons(N=1, equation_builder=corrected_neuron_model,
                       name="testNeuron2", model='rk4')

# Update parameters
neuron.set_params(neuron_model_param)
neuron1.set_params(dParamsNeuron)
neuron2.set_params(dParamsNeuron)

# Monitors
stm = b2.StateMonitor(neuron, ['Imem'], record=True)
spm = b2.SpikeMonitor(neuron, record=True)
stm1 = b2.StateMonitor(neuron1, ['Imem', 'Iahp', 'Ia_clip'], record=True)
spm1 = b2.SpikeMonitor(neuron1, record=True)
stm2 = b2.StateMonitor(neuron2, ['Imem', 'Iahp', 'Ia_clip'], record=True)
spm2 = b2.SpikeMonitor(neuron2, record=True)

# Network
net = b2.Network(neuron, neuron1, neuron2, stm, spm, stm1, spm1, stm2, spm2)

net.run(tInputDuration * second)

# - Plot
fig, axes = plt.subplots(3, sharex=True, figsize=(10,12))
axes[0].plot(stm.t/second, stm.Imem.T/amp)
axes[0].grid()
axes[0].set_xlim(0.055472177471279305, 0.15942424331515304)
axes[0].set_ylim(-2.7992480428019219e-10, 1.5852835169217604e-09)
ylims = axes[0].get_ylim()
for t in spm.t/second:
    axes[0].plot([t,t], ylims, 'k--')

axes[1].plot(stm1.t/second, stm1.Imem.T/amp)
axes[1].set_xlim(0.055472177471279305, 0.15942424331515304)
axes[1].set_ylim(-2.7992480428019219e-10, 1.5852835169217604e-09)
axes[1].grid()
for t in spm1.t/second:
    axes[1].plot([t,t], ylims, 'k--')

axes[2].plot(stm2.t/second, stm2.Imem.T/amp)
axes[2].set_xlim(0.055472177471279305, 0.15942424331515304)
axes[2].set_ylim(-2.7992480428019219e-10, 1.5852835169217604e-09)
axes[2].grid()
for t in spm2.t/second:
    axes[2].plot([t,t], ylims, 'k--')

# log of Imem
# plt.figure()
# plt.plot(stm1.t/second, np.log(np.clip(stm1.Imem.T/amp, 1e-12, None)))
# plt.plot(stm2.t/second, np.log(np.clip(stm2.Imem.T/amp, 1e-12, None)))