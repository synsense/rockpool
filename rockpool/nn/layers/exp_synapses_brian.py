"""
Spike-to-current layer with exponential synapses, with a Brian2 backend
"""


# - Imports
from warnings import warn

import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

from rockpool.timeseries import TSContinuous, TSEvent
from rockpool.nn.layers.layer import Layer
from rockpool.utilities.timedarray_shift import TimedArray as TAShift
from rockpool.nn.modules.timed_module import astimedmodule

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = ["FFExpSynBrian", "eqSynapseExp"]

# - Equations for an exponential synapse
eqSynapseExp = b2.Equations(
    """
    dI_syn/dt = (-I_syn + I_inp(t, i)) / tau_s  : amp                       # Synaptic current
    tau_s                                       : second                    # Synapse time constant
"""
)


## - FFExpSynBrian - Class: define an exponential synapse layer (spiking input)
@astimedmodule(
    parameters=["weights", "tau_syn"],
    states=["state"],
    simulation_parameters=["dt", "noise_std"],
)
class FFExpSynBrian(Layer):
    """Define an exponential synapse layer (spiking input), with a Brian2 backend"""

    ## - Constructor
    def __init__(
        self,
        weights: Union[np.ndarray, int] = None,
        dt: float = 0.1 * ms,
        noise_std: float = 0 * mV,
        tau_syn: float = 5 * ms,
        synapse_eq=eqSynapseExp,
        integrator_name: str = "rk4",
        name: str = "unnamed",
    ):
        """
        Construct an exponential synapse layer (spiking input), with a Brian2 backend

        :param weights:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param dt:             float Time step for state evolution. Default: 0.1 ms
        :param noise_std:       float Std. dev. of noise added to this layer. Default: 0

        :param tau_syn:         float Output synaptic time constants. Default: 5ms
        :param synapse_eq:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param integrator_name:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'
        """
        warn(
            "FFExpSynBrian - This layer is deprecated. You can use FFExpSyn or FFExpSynTorch instead."
        )

        # - Provide default dt
        if dt is None:
            dt = 0.1 * ms

        # - Provide default weight matrix for one-to-one conversion
        if isinstance(weights, int):
            weights = np.identity(weights, "float")

        # - Call super constructor
        super().__init__(weights=weights, dt=dt, noise_std=noise_std, name=name)

        # - Set up spike source to receive spiking input
        self._input_generator = b2.SpikeGeneratorGroup(
            self.size_in, [0], [0 * second], dt=np.asarray(dt) * second
        )

        # - Set up layer receiver nodes
        self._neuron_group = b2.NeuronGroup(
            self.size,
            synapse_eq,
            refractory=False,
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="receiver_neurons",
        )

        # - Add source -> receiver synapses
        self._inp_synapses = b2.Synapses(
            self._input_generator,
            self._neuron_group,
            model="w : 1",
            on_pre="I_syn_post += w*amp",
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="receiver_synapses",
        )
        self._inp_synapses.connect()

        # - Add current monitors to record reservoir outputs
        self._state_monitor = b2.StateMonitor(
            self._neuron_group, "I_syn", True, name="receiver_synaptic_currents"
        )

        # - Call Network constructor
        self._net = b2.Network(
            self._input_generator,
            self._neuron_group,
            self._inp_synapses,
            self._state_monitor,
            name="ff_spiking_to_exp_layer",
        )

        # - Record layer parameters, set weights
        self.weights = weights
        self.tau_syn = tau_syn

        # - Store "reset" state
        self._net.store("reset")

    def reset_state(self):
        """Reset the internal state of the layer"""
        self._neuron_group.I_syn = 0 * amp

    def randomize_state(self):
        """Randomize the internal state of the layer"""
        self.reset_state()

    def reset_time(self):
        """
        Reset the internal clock of this layer
        """

        # - Sotre state variables
        syn_inp = np.copy(self._neuron_group.I_syn) * amp

        # - Store parameters
        tau_syn = np.copy(self.tau_syn)
        weights = np.copy(self.weights)

        # - Reset network
        self._net.restore("reset")
        self._timestep = 0

        # - Restork parameters
        self.tau_syn = tau_syn
        self.weights = weights

        # - Restore state variables
        self._neuron_group.I_syn = syn_inp

    ### --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        Function to evolve the states of this layer given an input

        :param Optional[TSEvent] ts_input:      TSEvent  Input spike trian
        :param Optional[float] duration:           Simulation/Evolution time
        :param Optional[int] num_timesteps:       Number of evolution time steps
        :param bool verbose:            Currently no effect, just for conformity
        :return TSContinuous:              output spike series

        """

        # - Prepare time base
        time_base, __, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Set spikes for spike generator
        if ts_input is not None:
            event_times, event_channels = ts_input(
                t_start=time_base[0], t_stop=time_base[-1] + self.dt
            )
            self._input_generator.set_spikes(
                event_channels, event_times * second, sorted=False
            )
        else:
            self._input_generator.set_spikes([], [] * second)

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(time_base), self.size)
            * self.noise_std
            * np.sqrt(2 * self.tau_syn / self.dt)
        )
        # noise_step = np.zeros((np.size(time_base), self.size))
        # noise_step[0,:] = self.noise_std

        # - Specifiy noise input currents, construct TimedArray
        inp_noise = TAShift(
            np.asarray(noise_step) * amp,
            self.dt * second,
            tOffset=self.t * second,
            name="noise_input",
        )

        # - Perform simulation
        self._net.run(
            num_timesteps * self.dt * second, namespace={"I_inp": inp_noise}, level=0
        )
        self._timestep += num_timesteps

        # - Build response TimeSeries
        time_base_out = self._state_monitor.t_
        use_time = self._state_monitor.t_ >= time_base[0]
        time_base_out = time_base_out[use_time]
        a = self._state_monitor.I_syn_.T
        a = a[use_time, :]

        # - Return the current state as final time point
        if time_base_out[-1] != self.t:
            time_base_out = np.concatenate((time_base_out, [self.t]))
            a = np.concatenate((a, np.reshape(self.state, (1, self.size))))

        return TSContinuous(time_base_out, a, name="Receiver current")

    ### --- Properties

    @property
    def input_type(self):
        return TSEvent

    @property
    def weights(self):
        if hasattr(self, "_inp_synapses"):
            return np.reshape(self._inp_synapses.w, (self.size, -1))
        else:
            return self._weights

    @weights.setter
    def weights(self, new_w):
        assert np.size(new_w) == self.size * self.size_in, (
            "`new_w` must have [" + str(self.size * self.size_in) + "] elements."
        )

        self._weights = new_w

        if hasattr(self, "_inp_synapses"):
            # - Assign recurrent weights
            new_w = np.asarray(new_w).reshape(self.size, -1)
            self._inp_synapses.w = new_w.flatten()

    @property
    def state(self):
        return self._neuron_group.I_syn_

    @state.setter
    def state(self, new_state):
        self._neuron_group.I_syn = (
            np.asarray(self._expand_to_net_size(new_state, "new_state")) * amp
        )

    @property
    def tau_syn(self):
        return self._neuron_group.tau_s_[0]

    @tau_syn.setter
    def tau_syn(self, new_tau_syn):
        self._neuron_group.tau_s = np.asarray(new_tau_syn) * second

    @property
    def t(self):
        return self._net.t_

    @Layer.dt.setter
    def dt(self, _):
        warn("The `dt` property cannot be set for this layer")

    def to_dict(self):
        d = super().to_dict()
        d["tau_syn"] = self.tau_syn

        return d
