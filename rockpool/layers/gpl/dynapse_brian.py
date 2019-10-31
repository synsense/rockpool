###
# dynapse_brian.py - Class implementing a recurrent layer in Brian using Dynap equations from teili lib
###

# - Imports
from warnings import warn

import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

import sys

strNetworkPath = sys.path[0] + "../../.."
sys.path.insert(1, strNetworkPath)

from ...timeseries import TSContinuous, TSEvent

from ..layer import Layer

from .. import TimedArray as TAShift

# - Teili
from teili import Neurons as teiliNG, Connections as teiliSyn, teiliNetwork
from teili.models.neuron_models import DPI as teiliDPIEqts
from teili.models.synapse_models import DPISyn as teiliDPISynEqts
from teili.models.parameters.dpi_neuron_param import parameters as dTeiliNeuronParam

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["RecDynapseBrian"]


## - RecIAFBrian - Class: define a spiking recurrent layer based on Dynap equations
class RecDynapseBrian(Layer):
    """ RecIAFBrian - Class: define a spiking recurrent layer based on Dynap equations
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        weights_in: np.ndarray,
        dt: float = 0.1 * ms,
        noise_std: float = 0 * mV,
        refractory=0 * ms,
        neuron_params=None,
        syn_params=None,
        integrator_name: str = "rk4",
        name: str = "unnamed",
    ):
        """
        RecIAFBrian - Construct a spiking recurrent layer with IAF neurons, with a Brian2 back-end

        :param weights:             np.array NxN weight matrix
        :param weights_in:             np.array 1xN input weight matrix.

        :param refractory: float Refractory period after each spike. Default: 0ms

        :param neuron_params:    dict Parameters to over overwriting neuron defaulst

        :param syn_params:    dict Parameters to over overwriting synapse defaulst

        :param integrator_name:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'
        """
        warn("RecDynapseBrian: This layer is deprecated.")

        # - Call super constructor
        super().__init__(
            weights=weights,
            dt=np.asarray(dt),
            noise_std=np.asarray(noise_std),
            name=name,
        )

        # - Input weights must be provided
        assert weights_in is not None, "weights_in must be provided."

        # - Warn that nosie is not implemented
        if noise_std != 0:
            print("WARNING: Noise is currently not implemented in this layer.")

        # - Set up spike source to receive spiking input
        self._input_generator = b2.SpikeGeneratorGroup(
            self.size, [0], [0 * second], dt=np.asarray(dt) * second
        )

        # - Handle unit of dt: if no unit provided, assume it is in seconds
        dt = np.asscalar(np.array(dt)) * second

        ### --- Neurons

        # - Set up reservoir neurons
        self._neuron_group = teiliNG(
            N=self.size,
            equation_builder=teiliDPIEqts(num_inputs=2),
            name="reservoir_neurons",
            refractory=refractory,
            method=integrator_name,
            dt=dt,
        )

        # - Overwrite default neuron parameters
        if neuron_params is not None:
            self._neuron_group.set_params(dict(dTeiliNeuronParam, **neuron_params))
        else:
            self._neuron_group.set_params(dTeiliNeuronParam)

        ### --- Synapses

        # - Add recurrent synapses (all-to-all)
        self._rec_synapses = teiliSyn(
            self._neuron_group,
            self._neuron_group,
            equation_builder=teiliDPISynEqts,
            method=integrator_name,
            dt=dt,
            name="reservoir_recurrent_synapses",
        )
        self._rec_synapses.connect()

        # - Add source -> reservoir synapses (one-to-one)
        self._inp_synapses = teiliSyn(
            self._input_generator,
            self._neuron_group,
            equation_builder=teiliDPISynEqts,
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="receiver_synapses",
        )
        # Each spike generator neuron corresponds to one reservoir neuron
        self._inp_synapses.connect("i==j")

        # - Overwrite default synapse parameters
        if syn_params is not None:
            self._rec_synapses.set_params(neuron_params)
            self._inp_synapses.set_params(neuron_params)

        # - Add spike monitor to record layer outputs
        self._spike_monitor = b2.SpikeMonitor(
            self._neuron_group, record=True, name="layer_spikes"
        )

        # - Call Network constructor
        self._net = b2.Network(
            self._neuron_group,
            self._rec_synapses,
            self._input_generator,
            self._inp_synapses,
            self._spike_monitor,
            name="recurrent_spiking_layer",
        )

        # - Record neuron / synapse parameters
        # automatically sets weights  via setters
        self.weights = weights
        self.weights_in = weights_in

        # - Store "reset" state
        self._net.store("reset")

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._neuron_group.i_mem = 0 * amp
        self._neuron_group.i_ahp = 0.5 * pamp
        self._rec_synapses.Ie_syn = 0.5 * pamp
        self._rec_synapses.Ii_syn = 0.5 * pamp
        self._inp_synapses.Ie_syn = 0.5 * pamp
        self._inp_synapses.Ii_syn = 0.5 * pamp

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        # - Save state variables
        i_mem = np.copy(self._neuron_group.i_mem) * amp
        i_ahp = np.copy(self._neuron_group.i_ahp) * amp
        i_ex_recur = np.copy(self._rec_synapses.Ie_syn) * amp
        i_inh_recur = np.copy(self._rec_synapses.Ii_syn) * amp
        i_ex_inp = np.copy(self._inp_synapses.Ie_syn) * amp
        i_inh_inp = np.copy(self._inp_synapses.Ii_syn) * amp

        # - Save parameters
        weights = np.copy(self.weights)
        weights_in = np.copy(self.weights_in)

        # - Reset Network
        self._net.restore("reset")
        self._timestep = 0

        # - Restore state variables
        self._neuron_group.i_mem = i_mem
        self._neuron_group.i_ahp = i_ahp
        self._rec_synapses.Ie_syn = i_ex_recur
        self._rec_synapses.Ii_syn = i_inh_recur
        self._inp_synapses.Ie_syn = i_ex_inp
        self._inp_synapses.Ii_syn = i_inh_inp

        # - Restore parameters
        self.weights = weights
        self.weights_in = weights_in

    ### --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        time_base, input_steps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Set spikes for spike generator
        if ts_input is not None:
            event_times, event_channels, _ = ts_input.find(
                [time_base[0], time_base[-1] + self.dt]
            )
            self._input_generator.set_spikes(
                event_channels, event_times * second, sorted=False
            )
        else:
            self._input_generator.set_spikes([], [] * second)

        # - Perform simulation
        self._net.run(num_timesteps * self.dt * second, level=0)
        self._timestep += num_timesteps

        # - Build response TimeSeries
        use_event = self._spike_monitor.t_ >= time_base[0]
        event_time_out = self._spike_monitor.t[use_event]
        event_channel_out = self._spike_monitor.i[use_event]

        return TSEvent(event_time_out, event_channel_out, name="Layer spikes")

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @property
    def input_type(self):
        return TSEvent

    @property
    def weights(self):
        if hasattr(self, "_rec_synapses"):
            return np.reshape(self._rec_synapses.weight, (self.size, -1))
        else:
            return self._weights

    @weights.setter
    def weights(self, new_w):
        assert np.size(new_w) == self.size ** 2, (
            "`new_w` must have [" + str(self.size ** 2) + "] elements."
        )

        self._weights = new_w

        if hasattr(self, "_rec_synapses"):
            # - Assign recurrent weights
            new_w = np.asarray(new_w).reshape(self.size, -1)
            self._rec_synapses.weight = new_w.flatten()

    @property
    def weights_in(self):
        if hasattr(self, "_inp_synapses"):
            return np.reshape(self._inp_synapses.weight, (self.size, -1))
        else:
            return self._weights

    @weights_in.setter
    def weights_in(self, new_weights):
        assert np.size(new_weights) == self.size, (
            "`new_w` must have [" + str(self.size) + "] elements."
        )

        self._weights = new_weights

        if hasattr(self, "_inp_synapses"):
            # - Assign input weights
            self._inp_synapses.weight = new_weights.flatten()

    @property
    def state(self):
        return self._neuron_group.Imem_

    @state.setter
    def state(self, new_state):
        self._neuron_group.i_mem = (
            np.asarray(self._expand_to_net_size(new_state, "new_state")) * volt
        )

    @property
    def t(self):
        return self._net.t_

    @Layer.dt.setter
    def dt(self, _):
        raise ValueError("The `dt` property cannot be set for this layer")
