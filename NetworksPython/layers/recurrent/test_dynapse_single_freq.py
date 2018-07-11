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

tInputDuration = 2

lfIconst = np.array([4.375, 4.5, 4.75, 5, 5.25, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 12, 12.5]) 
lfIconst = np.hstack((lfIconst, np.arange(15, 205, 5))) * 1e-9*amp

def freq(fIconst):
    # - Neuron
    neuron = Neurons(N=1, equation_builder=neuron_model(num_inputs=1),
                           name="testNeuron")

    # Update parameters
    neuron_model_param.update(Iconst=fIconst)
    neuron.set_params(neuron_model_param)

    # Monitors
    # stm = b2.StateMonitor(neuron, ['Imem'], record=True)
    spm = b2.SpikeMonitor(neuron, record=True)

    # Network
    net = b2.Network(neuron, spm)

    net.run(tInputDuration * second)

    # Firing rates from inter-spike intervals
    vfRates = 1./np.diff(spm.t/second)  
    fFreqMean = np.mean(vfRates)
    fFreqStd = np.std(vfRates, ddof=1)

    # return frequency
    return fFreqMean, fFreqStd

lfFreq = []
lfStd = []
for n, fIconst in enumerate(lfIconst):
    print("Step {} of {}".format(n+1, len(lfIconst)), end="\r")
    fFreq, fStd = freq(fIconst)
    lfFreq.append(fFreq)
    lfStd.append(fStd)
print('')

# Plot
plt.figure()
plt.errorbar(lfIconst, lfFreq, lfStd)
ax = plt.gca()
ax.grid()
ax.set_title("Dynapse single neuron response")
ax.set_xlabel("Iconst in amp")
ax.set_ylabel("Firing rate in Hz")