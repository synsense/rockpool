###
# iaf_brian.py - Classes implementing recurrent and feedforward layers consisting of standard IAF neurons in brian2
###


# - Imports
from warnings import warn

import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

from ...timeseries import TSContinuous, TSEvent

from ..layer import Layer
from rockpool.utilities import TimedArray as TAShift

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = [
    "FFIAFBrian",
    "FFIAFSpkInBrian",
    "RecIAFBrian",
    "RecIAFSpkInBrian",
    "eqNeuronIAFFF",
    "eqNeuronIAFSpkInFF",
    "eqNeuronIAFRec",
    "eqNeuronIAFSpkInRec",
    "eqSynapseExp",
    "eqSynapseExpSpkInRec",
]

# - Equations for an integrate-and-fire neuron, ff-layer, analogue external input
eqNeuronIAFFF = b2.Equations(
    """
    dv/dt = (v_rest - v + r_m * I_total) / tau_m    : volt (unless refractory)  # Neuron membrane voltage
    I_total = I_inp(t, i) + I_bias                  : amp                       # Total input current
    I_bias                                          : amp                       # Per-neuron bias current
    v_rest                                          : volt                      # Rest potential
    tau_m                                           : second                    # Membrane time constant
    r_m                                             : ohm                       # Membrane resistance
    v_thresh                                        : volt                      # Firing threshold potential
    v_reset                                         : volt                      # Reset potential
"""
)

# - Equations for an integrate-and-fire neuron, ff-layer, analogue external input, constant leak
eqNeuronCLIAFFF = b2.Equations(
    """
    dv/dt = (r_m * I_total) / tau_m                 : volt (unless refractory)  # Neuron membrane voltage
    I_total = I_inp(t, i) + I_bias                  : amp                       # Total input current
    I_bias                                          : amp                       # Per-neuron bias current
    v_rest                                          : volt                      # Rest potential
    tau_m                                           : second                    # Membrane time constant
    r_m                                             : ohm                       # Membrane resistance
    v_thresh                                        : volt                      # Firing threshold potential
    v_reset                                         : volt                      # Reset potential
"""
)

# - Equations for an integrate-and-fire neuron, ff-layer, spiking external input
eqNeuronIAFSpkInFF = b2.Equations(
    """
    dv/dt = (v_rest - v + r_m * I_total) / tau_m    : volt (unless refractory)  # Neuron membrane voltage
    I_total = I_syn + I_bias + I_inp(t, i)          : amp                       # Total input current
    dI_syn/dt = -I_syn / tau_s                      : amp                       # Synaptic input current
    I_bias                                          : amp                       # Per-neuron bias current
    v_rest                                          : volt                      # Rest potential
    tau_m                                           : second                    # Membrane time constant
    tau_s                                           : second                    # Membrane time constant
    r_m                                             : ohm                       # Membrane resistance
    v_thresh                                        : volt                      # Firing threshold potential
    v_reset                                         : volt                      # Reset potential
"""
)

# - Equations for an integrate-and-fire neuron, recurrent layer, analogue external input
eqNeuronIAFRec = b2.Equations(
    """
   dv/dt = (v_rest - v + r_m * I_total) / tau_m    : volt (unless refractory)  # Neuron membrane voltage
   I_total = I_inp(t, i) + I_syn + I_bias          : amp                       # Total input current
   I_bias                                          : amp                       # Per-neuron bias current
   v_rest                                          : volt                      # Rest potential
   tau_m                                           : second                    # Membrane time constant
   r_m                                             : ohm                       # Membrane resistance
   v_thresh                                        : volt                      # Firing threshold potential
   v_reset                                         : volt                      # Reset potential
"""
)

# - Equations for an integrate-and-fire neuron, recurrent layer, spiking external input
eqNeuronIAFSpkInRec = b2.Equations(
    """
    dv/dt = (v_rest - v + r_m * I_total) / tau_m    : volt (unless refractory)  # Neuron membrane voltage
    I_total = I_inp(t, i) + I_syn + I_bias          : amp                       # Total input current
    I_syn = I_syn_inp + I_syn_rec                   : amp                       # Synaptic currents
    I_bias                                          : amp                       # Per-neuron bias current
    v_rest                                          : volt                      # Rest potential
    tau_m                                           : second                    # Membrane time constant
    r_m                                             : ohm                       # Membrane resistance
    v_thresh                                        : volt                      # Firing threshold potential
    v_reset                                         : volt                      # Reset potential
"""
)

# - Equations for an exponential synapse - used for RecIAFBrian
eqSynapseExp = b2.Equations(
    """
    dI_syn/dt = -I_syn / tau_s                      : amp                       # Synaptic current
    tau_s                                           : second                    # Synapse time constant
"""
)

# - Equations for two exponential synapses (spiking external input and recurrent) for RecIAFSpkInBrian
eqSynapseExpSpkInRec = b2.Equations(
    """
    dI_syn_inp/dt = -I_syn_inp / tau_syn_inp        : amp                       # Synaptic current, input synapses
    dI_syn_rec/dt = -I_syn_rec / tau_syn_rec        : amp                       # Synaptic current, recurrent synapses
    tau_syn_inp                                     : second                    # Synapse time constant, input
    tau_syn_rec                                     : second                    # Synapse time constant, recurrent
"""
)

## - FFIAFBrian - Class: define a spiking feedforward layer with spiking outputs
class FFIAFBrian(Layer):
    """ FFIAFBrian - Class: define a spiking feedforward layer with spiking outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: Union[float, np.ndarray] = 15 * mA,
        dt: float = 0.1 * ms,
        noise_std: float = 0 * mV,
        tau_mem: Union[float, np.ndarray] = 20 * ms,
        v_thresh: Union[float, np.ndarray] = -55 * mV,
        v_reset: Union[float, np.ndarray] = -65 * mV,
        v_rest: Union[float, np.ndarray] = -65 * mV,
        refractory: float = 0 * ms,
        neuron_eq: str = eqNeuronIAFFF,
        integrator_name: str = "rk4",
        name: str = "unnamed",
        record: bool = False,
    ):
        """
        FFIAFBrian - Construct a spiking feedforward layer with IAF neurons, with a Brian2 back-end
                     Inputs are continuous currents; outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param refractory: float Refractory period after each spike. Default: 0ms

        :param neuron_eq:       Brian2.Equations set of neuron equations. Default: IAF equation set

        :param integrator_name:   str Integrator to use for simulation. Default: 'rk4'

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions
        """
        
        warn("FFIAFBrian: This layer is deprecated. You can use FFIAFTorch or FFIAFNest instead.")

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            weights=np.asarray(weights),
            dt=np.asarray(dt),
            noise_std=np.asarray(noise_std),
            name=name,
        )

        # - Set up layer neurons
        self._neuron_group = b2.NeuronGroup(
            self.size,
            neuron_eq,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=np.asarray(refractory) * second,
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="spiking_ff_neurons",
        )
        self._neuron_group.v = v_rest
        self._neuron_group.r_m = 1 * ohm

        # - Add monitors to record layer outputs
        self._layer = b2.SpikeMonitor(
            self._neuron_group, record=True, name="layer_spikes"
        )

        # - Call Network constructor
        self._net = b2.Network(self._neuron_group, self._layer, name="ff_spiking_layer")

        if record:
            # - Monitor for recording network potential
            self.state_monitor = b2.StateMonitor(
                self._neuron_group, ["v"], record=True, name="layer_potential"
            )
            self._net.add(self.state_monitor)

        # - Record neuron parameters
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.bias = bias
        self.weights = weights

        # - Store "reset" state
        self._net.store("reset")

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._neuron_group.v = self.v_rest * volt

    def randomize_state(self):
        """ .randomize_state() - arguments:: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        v_range = abs(self.v_thresh - self.v_reset)
        self._neuron_group.v = (
            np.random.rand(self.size) * v_range + self.v_reset
        ) * volt

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        # - Sotre state variables
        v_state = np.copy(self._neuron_group.v) * volt

        # - Store parameters
        v_thresh = np.copy(self.v_thresh)
        v_reset = np.copy(self.v_reset)
        v_rest = np.copy(self.v_rest)
        tau_mem = np.copy(self.tau_mem)
        bias = np.copy(self.bias)
        weights = np.copy(self.weights)

        # - Reset network
        self._net.restore("reset")
        self._timestep = 0

        # - Restork parameters
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.bias = bias
        self.weights = weights

        # - Restore state variables
        self._neuron_group.v = v_state

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

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        time_base, input_steps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Weight inputs
        neuron_inp_step = input_steps @ self.weights

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(time_base), self.size)
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.noise_std
            * np.sqrt(2.0 * self.tau_mem / self.dt)
            * 1.63
        )

        # - Specifiy network input currents, construct TimedArray
        inp_current = TAShift(
            np.asarray(neuron_inp_step + noise_step) * amp,
            self.dt * second,
            tOffset=self.t * second,
            name="external_input",
        )

        # - Perform simulation
        self._net.run(
            num_timesteps * self.dt * second, namespace={"I_inp": inp_current}, level=0
        )

        # - Start and stop times for output time series
        t_start = self._timestep * np.asscalar(self.dt)
        t_stop = (self._timestep + num_timesteps) * np.asscalar(self.dt)

        # - Update layer time step
        self._timestep += num_timesteps

        # - Build response TimeSeries
        use_event = self._layer.t_ >= time_base[0]
        event_time_out = self._layer.t_[use_event]
        event_channel_out = self._layer.i[use_event]

        return TSEvent(
            np.clip(event_time_out, t_start, t_stop),
            event_channel_out,
            name="Layer spikes",
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

    def stream(
        self, duration: float, dt: float, verbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        stream - Stream data through this layer
        :param duration:   float Total duration for which to handle streaming
        :param dt:         float Streaming time step
        :param verbose:    bool Display feedback

        :yield: (event_times, event_channels)

        :return: Final (event_times, event_channels)
        """

        # - Initialise simulation, determine how many dt to evolve for
        if verbose:
            print("Layer: I'm preparing")
        time_trace = np.arange(0, duration + dt, dt)
        num_steps = np.size(time_trace) - 1

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(time_trace), self.size)
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.noise_std
            * np.sqrt(2.0 * self.tau_mem / self.dt)
            * 1.63
        )

        # - Generate a TimedArray to use for step-constant input currents
        inp_current = TAShift(
            np.zeros((1, self._size_in)) * amp, self.dt * second, name="external_input"
        )

        if verbose:
            print("Layer: Prepared")

        # - Loop over dt steps
        for step in range(num_steps):
            if verbose:
                print("Layer: Yielding from internal state.")
            if verbose:
                print("Layer: step", step)
            if verbose:
                print("Layer: Waiting for input...")

            # - Yield current output spikes, receive input for next time step
            use_events = self._layer.t_ >= time_trace[step]
            if verbose:
                print("Layer: Yielding {} spikes".format(np.sum(use_events)))
            inp = (yield self._layer.t_[use_events], self._layer.i_[use_events])

            # - Specify network input currents for this streaming step
            if inp is None:
                inp_current.values = noise_step[step, :]
            else:
                inp_current.values = (
                    np.reshape(inp[1][0, :], (1, -1)) + noise_step[step, :]
                )

            # - Reinitialise TimedArray
            inp_current._init_2d()

            if verbose:
                print("Layer: Input was: ", inp)

            # - Evolve layer (increments time implicitly)
            self._net.run(dt * second, namespace={"I_inp": inp_current}, level=0)

        # - Return final spikes, if any
        use_events = self._layer.t_ >= time_trace[-2]  # Should be duration - dt
        return self._layer.t_[use_events], self._layer.i_[use_events]

    def to_dict(self) -> dict:
        """
        to_dict - Convert parameters of `self` to a dict if they are relevant for
                  reconstructing an identical layer.
        """
        config = super().to_dict()
        config["bias"] = self.bias.tolist()
        config["tau_mem"] = self.tau_mem.tolist()
        config["v_thresh"] = self.v_thresh.tolist()
        config["v_reset"] = self.v_reset.tolist()
        config["v_rest"] = self.v_rest.tolist()
        config["refractory"] = self.refractory
        config["neuron_eq"] = self._neuron_group.equations
        config["integrator_name"] = self._neuron_group.method
        config["record"] = self.hasattr("state_monitor")

        return config

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @property
    def refractory(self):
        return self._neuron_group._refractory

    @property
    def state(self):
        return self._neuron_group.v_

    @state.setter
    def state(self, new_state):
        self._neuron_group.v = (
            np.asarray(self._expand_to_net_size(new_state, "new_state")) * volt
        )

    @property
    def tau_mem(self):
        return self._neuron_group.tau_m_

    @tau_mem.setter
    def tau_mem(self, new_tau_mem):
        self._neuron_group.tau_m = (
            np.asarray(self._expand_to_net_size(new_tau_mem, "new_tau_mem")) * second
        )

    @property
    def bias(self):
        return self._neuron_group.I_bias_

    @bias.setter
    def bias(self, new_bias):
        self._neuron_group.I_bias = (
            np.asarray(self._expand_to_net_size(new_bias, "new_bias")) * amp
        )

    @property
    def v_thresh(self):
        return self._neuron_group.v_thresh_

    @v_thresh.setter
    def v_thresh(self, new_v_thresh):
        self._neuron_group.v_thresh = (
            np.asarray(self._expand_to_net_size(new_v_thresh, "new_v_thresh")) * volt
        )

    @property
    def v_rest(self):
        return self._neuron_group.v_rest_

    @v_rest.setter
    def v_rest(self, new_v_rest):
        self._neuron_group.v_rest = (
            np.asarray(self._expand_to_net_size(new_v_rest, "new_v_rest")) * volt
        )

    @property
    def v_reset(self):
        return self._neuron_group.v_reset_

    @v_reset.setter
    def v_reset(self, new_v_reset):
        self._neuron_group.v_reset = (
            np.asarray(self._expand_to_net_size(new_v_reset, "new_v_reset")) * volt
        )

    @property
    def t(self):
        return self._net.t_

    @Layer.dt.setter
    def dt(self, _):
        raise ValueError("The `dt` property cannot be set for this layer")


# - FFIAFSpkInBrian - Class: Spiking feedforward layer with spiking in- and outputs
class FFIAFSpkInBrian(FFIAFBrian):
    """ FFIAFSpkInBrian - Class: Spiking feedforward layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray = 10 * mA,
        dt: float = 0.1 * ms,
        noise_std: float = 0 * mV,
        tau_mem: np.ndarray = 20 * ms,
        tau_syn: np.ndarray = 20 * ms,
        v_thresh: np.ndarray = -55 * mV,
        v_reset: np.ndarray = -65 * mV,
        v_rest: np.ndarray = -65 * mV,
        refractory: float = 0 * ms,
        neuron_eq: str = eqNeuronIAFSpkInFF,
        integrator_name: str = "rk4",
        name: str = "unnamed",
        record: bool = False,
    ):
        """
        FFIAFSpkInBrian - Construct a spiking feedforward layer with IAF neurons, with a Brian2 back-end
                          in- and outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param tau_syn:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param refractory: float Refractory period after each spike. Default: 0ms

        :param neuron_eq:       Brian2.Equations set of neuron equations. Default: IAF equation set

        :param integrator_name:   str Integrator to use for simulation. Default: 'rk4'

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions
        """

        warn("FFIAFSpkInBrian: This layer is deprecated. You can use FFIAFSpkInTorch instad.")

        # - Call Layer constructor
        Layer.__init__(
            self,
            weights=weights,
            dt=np.asarray(dt),
            noise_std=np.asarray(noise_std),
            name=name,
        )

        # - Set up spike source to receive spiking input
        self._input_generator = b2.SpikeGeneratorGroup(
            self.size_in, [0], [0 * second], dt=np.asarray(dt) * second
        )
        # - Set up layer neurons
        self._neuron_group = b2.NeuronGroup(
            self.size,
            neuron_eq,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=np.asarray(refractory) * second,
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="spiking_ff_neurons",
        )
        self._neuron_group.v = v_rest
        self._neuron_group.r_m = 1 * ohm

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

        # - Add monitors to record layer outputs
        self._layer = b2.SpikeMonitor(
            self._neuron_group, record=True, name="layer_spikes"
        )

        # - Call Network constructor
        self._net = b2.Network(
            self._input_generator,
            self._inp_synapses,
            self._neuron_group,
            self._layer,
            name="ff_spiking_layer",
        )

        if record:
            # - Monitor for recording network potential
            self.state_monitor = b2.StateMonitor(
                self._neuron_group, ["v"], record=True, name="layer_potential"
            )
            self._net.add(self.state_monitor)

        # - Record neuron parameters
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.bias = bias
        self.weights = weights

        # - Store "reset" state
        self._net.store("reset")

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
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
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)
        time_base = self.t + np.arange(num_timesteps) * self.dt

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
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.noise_std
            * np.sqrt(2.0 * self.tau_mem / self.dt)
            * 1.63
        )

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

        # - Start and stop times for output time series
        t_start = self._timestep * np.asscalar(self.dt)
        t_stop = (self._timestep + num_timesteps) * np.asscalar(self.dt)

        # - Update layer time
        self._timestep += num_timesteps

        # - Build response TimeSeries
        use_event = self._layer.t_ >= time_base[0]
        event_time_out = self._layer.t_[use_event]
        event_channel_out = self._layer.i[use_event]

        return TSEvent(
            np.clip(event_time_out, t_start, t_stop),
            event_channel_out,
            name="Layer spikes",
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

    def reset_time(self):

        # - Store state variables
        v_state = np.copy(self._neuron_group.v) * volt
        syn_inp = np.copy(self._neuron_group.I_syn) * amp

        # - Store parameters
        v_thresh = np.copy(self.v_thresh)
        v_reset = np.copy(self.v_reset)
        v_rest = np.copy(self.v_rest)
        tau_mem = np.copy(self.tau_mem)
        tau_syn = np.copy(self.tau_syn)
        bias = np.copy(self.bias)
        weights = np.copy(self.weights)

        self._net.restore("reset")
        self._timestep = 0

        # - Restork parameters
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.bias = bias
        self.weights = weights

        # - Restore state variables
        self._neuron_group.v = v_state
        self._neuron_group.I_syn = syn_inp

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._neuron_group.v = self.v_rest * volt
        self._neuron_group.I_syn = 0 * amp

    def reset_all(self, keep_params=True):
        if keep_params:
            # - Store parameters
            v_thresh = np.copy(self.v_thresh)
            v_reset = np.copy(self.v_reset)
            v_rest = np.copy(self.v_rest)
            tau_mem = np.copy(self.tau_mem)
            tau_syn = np.copy(self.tau_syn)
            bias = np.copy(self.bias)
            weights = np.copy(self.weights)

        self.reset_state()
        self._net.restore("reset")
        self._timestep = 0

        if keep_params:
            # - Restork parameters
            self.v_thresh = v_thresh
            self.v_reset = v_reset
            self.v_rest = v_rest
            self.tau_mem = tau_mem
            self.tau_syn = tau_syn
            self.bias = bias
            self.weights = weights

    def randomize_state(self):
        """ .randomize_state() - arguments:: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        v_range = abs(self.v_thresh - self.v_reset)
        self._neuron_group.v = (
            np.random.rand(self.size) * v_range + self.v_reset
        ) * volt
        self._neuron_group.I_syn = np.random.rand(self.size) * amp

    def pot_kernel(self, t):
        """ pot_kernel - response of the membrane potential to an
                         incoming spike at a single synapse with
                         weight 1*amp (not considering v_rest)
        """
        t = t.reshape(-1, 1)
        fConst = (
            self.tau_syn / (self.tau_syn - self.tau_mem) * self._neuron_group.r_m * amp
        )
        return fConst * (np.exp(-t / self.tau_syn) - np.exp(-t / self.tau_mem))

    def train(
        self,
        ts_target: None,
        ts_input: TSContinuous,
        is_first: bool,
        is_last: bool,
        method: str = "mst",
        **kwargs,
    ):
        """
        train - Wrapper to standardize training syntax across layers. Use
                specified training method to train layer for current batch.
        :param ts_target: Target time series for current batch. Can be skipped for `mst` method.
        :param ts_input:  Input to the layer during the current batch.
        :param is_first:  Set `True` to indicate that this batch is the first in training procedure.
        :param is_last:   Set `True` to indicate that this batch is the last in training procedure.
        :param method:    String indicating which training method to choose.
                          Currently only multi-spike tempotron ("mst") is supported.
        kwargs will be passed on to corresponding training method.
        For 'mst' method, kwargs `duration` and `t_start` must be provided.
        """
        # - Choose training method
        if method in {"mst", "multi-spike tempotron"}:
            if "duration" not in kwargs.keys():
                raise TypeError(
                    f"FFIAFSpkInBrian `{self.name}`: For multi-spike tempotron, argument "
                    + "`duration` must be provided."
                )
            if "t_start" not in kwargs.keys():
                raise TypeError(
                    f"FFIAFSpkInBrian `{self.name}`: For multi-spike tempotron, argument "
                    + "`t_start` must be provided."
                )
            self.train_mst_simple(
                ts_input=ts_input, is_first=is_first, is_last=is_last, **kwargs
            )
        else:
            raise ValueError(
                f"FFIAFSpkInBrian `{self.name}`: Training method `{method}` is currently "
                + "not supported. Use `mst` for multi-spike tempotron."
            )

    def train_mst_simple(
        self,
        duration: float,
        t_start: float,
        ts_input: TSEvent,
        target_counts: np.ndarray = None,
        lambda_: float = 1e-5,
        eligibility_ratio: float = 0.1,
        momentum: float = 0,
        is_first: bool = True,
        is_last: bool = False,
        verbose: bool = False,
    ):
        """
        train_mst_simple - Use the multi-spike tempotron learning rule
                           from Guetig2017, in its simplified version,
                           where no gradients are calculated
        """

        assert hasattr(self, "state_monitor"), (
            "Layer needs to be instantiated with record=True for "
            + "this learning rule."
        )

        # - End time of current batch
        t_stop = t_start + duration

        if ts_input is not None:
            event_times, event_channels = ts_input(t_start=t_start, t_stop=t_stop)
        else:
            print("No ts_input defined, assuming input to be 0.")
            event_times, event_channels = [], []

        # - Prepare target
        if target_counts is None:
            target_counts = np.zeros(self.size)
        else:
            assert (
                np.size(target_counts) == self.size
            ), "Target array size must match layer size ({}).".format(self.size)

        ## -- Determine eligibility for each neuron and synapse
        eligibility = np.zeros((self.size_in, self.size))

        # - Iterate over source neurons
        for source_id in range(self.size_in):
            if verbose:
                print(
                    "\rProcessing input {} of {}".format(source_id + 1, self.size_in),
                    end="",
                )
            # - Find spike timings
            event_time_source = event_times[event_channels == source_id]
            # - Sum individual correlations over input spikes, for all synapses
            for t_spike_in in event_time_source:
                # - Membrane potential between input spike time and now (transform to v_rest at 0)
                v_mem = (
                    self.state_monitor.v.T[self.state_monitor.t_ >= t_spike_in]
                    - self.v_rest * volt
                )
                # - Kernel between input spike time and now
                kernel = self.pot_kernel(
                    self.state_monitor.t_[self.state_monitor.t_ >= t_spike_in]
                    - t_spike_in
                )
                # - Add correlations to eligibility matrix
                eligibility[source_id, :] += np.sum(kernel * v_mem)

        ## -- For each neuron sort eligibilities and choose synapses with largest eligibility
        eligible = int(eligibility_ratio * self.size_in)
        # - Mark eligible neurons
        is_eligible = np.argsort(eligibility, axis=0)[:eligible:-1]

        ##  -- Compare target number of events with spikes and perform weight updates for chosen synapses
        # - Numbers of (output) spike times for each neuron
        use_out_events = (self._layer.t_ >= t_start) & (self._layer.t_ <= t_stop)
        spikes_out_neurons = self._layer.i[use_out_events]
        spike_counts = np.array(
            [np.sum(spikes_out_neurons == n_id) for n_id in range(self.size)]
        )

        # - Updates to eligible synapses of each neuron
        updates = np.zeros(self.size)
        # - Negative update if spike count too high
        updates[spike_counts > target_counts] = -lambda_
        # - Positive update if spike count too low
        updates[spike_counts < target_counts] = lambda_

        # - Reset previous weight changes that are used for momentum heuristic
        if is_first:
            self._dw_previous = np.zeros_like(self.weights)

        # - Accumulate updates to me made to weights
        dw_current = np.zeros_like(self.weights)

        # - Update only eligible synapses
        for target_id in range(self.size):
            dw_current[is_eligible[:, target_id], target_id] += updates[target_id]

        # - Include previous weight changes for momentum heuristic
        dw_current += momentum * self._dw_previous

        # - Perform weight update
        self.weights += dw_current
        # - Store weight changes for next iteration
        self._dw_previous = dw_current

    def to_dict(self) -> dict:
        """
        to_dict - Convert parameters of `self` to a dict if they are relevant for
                  reconstructing an identical layer.
        """
        config = super().to_dict()
        config["tau_syn"] = self.tau_syn.tolist()
        return config

    @property
    def input_type(self):
        return TSEvent

    @property
    def refractory(self):
        return self._neuron_group._refractory

    @property
    def weights(self):
        return np.array(self._inp_synapses.w).reshape(self.size_in, self.size)

    @weights.setter
    def weights(self, new_w):
        assert (
            new_w.shape == (self.size_in, self.size)
            or new_w.shape == self._inp_synapses.w.shape
        ), "weights must be of dimensions ({}, {}) or flat with size {}.".format(
            self.size_in, self.size, self.size_in * self.size
        )

        self._inp_synapses.w = np.array(new_w).flatten()

    @property
    def tau_syn(self):
        return self._neuron_group.tau_s_

    @tau_syn.setter
    def tau_syn(self, new_tau_syn):
        self._neuron_group.tau_s = (
            np.asarray(self._expand_to_net_size(new_tau_syn, "new_tau_syn")) * second
        )


## - RecIAFBrian - Class: define a spiking recurrent layer with exponential synaptic outputs
class RecIAFBrian(Layer):
    """ RecIAFBrian - Class: define a spiking recurrent layer with exponential synaptic outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray = None,
        bias: Union[float, np.ndarray] = 10.5 * mA,
        dt: float = 0.1 * ms,
        noise_std: float = 0 * mV,
        tau_mem: Union[float, np.ndarray] = 20 * ms,
        tau_syn_r: Union[float, np.ndarray] = 50 * ms,
        v_thresh: Union[float, np.ndarray] = -55 * mV,
        v_reset: Union[float, np.ndarray] = -65 * mV,
        v_rest: Union[float, np.ndarray] = -65 * mV,
        refractory: float = 0 * ms,
        neuron_eq: str = eqNeuronIAFRec,
        rec_syn_eq: str = eqSynapseExp,
        integrator_name: str = "rk4",
        name: str = "unnamed",
        record: bool = False,
    ):
        """
        RecIAFBrian - Construct a spiking recurrent layer with IAF neurons, with a Brian2 back-end

        :param weights:             np.array NxN weight matrix. Default: [100x100] unit-lambda matrix
        :param bias:          np.array Nx1 bias vector. Default: 10.5mA

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20 ms
        :param tau_syn_r:       np.array NxN vector of recurrent synaptic time constants. Default: 50 ms

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param refractory: float Refractory period after each spike. Default: 0ms

        :param neuron_eq:       Brian2.Equations set of neuron equations. Default: IAF equation set
        :param rec_syn_eq:  Brian2.Equations set of synapse equations for recurrent connects. Default: exponential

        :param integrator_name:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions
        """

        warn("RecIAFBrian: This layer is deprecated. You can use RecIAFTorch instad.")
        assert (
            np.atleast_2d(weights).shape[0] == np.atleast_2d(weights).shape[1]
        ), "Layer `{}`: weights must be a square matrix.".format(name)
        # - Call super constructor
        super().__init__(
            weights=weights,
            dt=np.asarray(dt),
            noise_std=np.asarray(noise_std),
            name=name,
        )

        # - Set up reservoir neurons
        self._neuron_group = b2.NeuronGroup(
            self.size,
            neuron_eq + rec_syn_eq,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=np.asarray(refractory) * second,
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="reservoir_neurons",
        )
        self._neuron_group.v = v_rest
        self._neuron_group.r_m = 1 * ohm

        # - Add recurrent weights (all-to-all)
        self._rec_synapses = b2.Synapses(
            self._neuron_group,
            self._neuron_group,
            model="w : 1",
            on_pre="I_syn_post += w*amp",
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="reservoir_recurrent_synapses",
        )
        self._rec_synapses.connect()

        # - Add spike monitor to record layer outputs
        self._spike_monitor = b2.SpikeMonitor(
            self._neuron_group, record=True, name="layer_spikes"
        )

        # - Call Network constructor
        self._net = b2.Network(
            self._neuron_group,
            self._rec_synapses,
            self._spike_monitor,
            name="recurrent_spiking_layer",
        )

        if record:
            # - Monitor for recording network potential
            self._v_monitor = b2.StateMonitor(
                self._neuron_group,
                ["v", "I_syn", "I_total"],
                record=True,
                name="layer_neurons",
            )
            self._net.add(self._v_monitor)

        # - Record neuron / synapse parameters
        self.weights = weights
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.tau_syn_r = tau_syn_r
        self.bias = bias

        self._neuron_eq = neuron_eq
        self._rec_syn_eq = rec_syn_eq

        # - Store "reset" state
        self._net.store("reset")

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._neuron_group.v = self.v_rest * volt
        self._neuron_group.I_syn = 0 * amp

    def randomize_state(self):
        """ .randomize_state() - arguments:: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        v_range = abs(self.v_thresh - self.v_reset)
        self._neuron_group.v = (
            np.random.rand(self.size) * v_range + self.v_reset
        ) * volt
        self._neuron_group.I_syn = np.random.rand(self.size) * amp

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        # - Store state variables
        v_state = np.copy(self._neuron_group.v) * volt
        syn_inp = np.copy(self._neuron_group.I_syn) * amp

        # - Store parameters
        v_thresh = np.copy(self.v_thresh)
        v_reset = np.copy(self.v_reset)
        v_rest = np.copy(self.v_rest)
        tau_mem = np.copy(self.tau_mem)
        tau_syn_r = np.copy(self.tau_syn_r)
        bias = np.copy(self.bias)
        weights = np.copy(self.weights)

        # - Reset network
        self._net.restore("reset")
        self._timestep = 0

        # - Restork parameters
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.tau_syn_r = tau_syn_r
        self.bias = bias
        self.weights = weights

        # - Restore state variables
        self._neuron_group.v = v_state
        self._neuron_group.I_syn = syn_inp

    def to_dict(self) -> dict:
        """
        to_dict - Convert parameters of `self` to a dict if they are relevant for
                  reconstructing an identical layer.
        """
        config = super().to_dict()
        config["rec_syn_eq"] = self._rec_syn_eq
        config["neuron_eq"] = self._neuron_eq
        config["tau_mem"] = self.tau_mem.tolist()
        config["tau_syn_r"] = self.tau_syn_r.tolist()
        config["v_thresh"] = self.v_thresh.tolist()
        config["v_reset"] = self.v_reset.tolist()
        config["v_rest"] = self.v_rest.tolist()
        config["refractory"] = self.refractory
        config["integrator_name"] = self._neuron_group.method
        config["record"] = self.hasattr("state_monitor")

        return config

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

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        time_base, input_steps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Store stuff for debugging
        self.time_base = time_base
        self.input_steps = input_steps
        self.num_timesteps = num_timesteps

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(time_base), self.size)
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.noise_std
            * np.sqrt(2.0 * self.tau_mem / self.dt)
            * 1.63
        )

        # - Specifiy network input currents, construct TimedArray
        inp_current = TAShift(
            np.asarray(input_steps + noise_step) * amp,
            self.dt * second,
            tOffset=self.t * second,
            name="external_input",
        )

        # - Perform simulation
        self._net.run(
            num_timesteps * self.dt * second, namespace={"I_inp": inp_current}, level=0
        )

        # - Start and stop times for output time series
        t_start = self._timestep * np.asscalar(self.dt)
        t_stop = (self._timestep + num_timesteps) * np.asscalar(self.dt)

        # - Update layer time step
        self._timestep += num_timesteps

        # - Build response TimeSeries
        use_event = self._spike_monitor.t_ >= time_base[0]
        event_time_out = self._spike_monitor.t_[use_event]
        event_channel_out = self._spike_monitor.i[use_event]

        return TSEvent(
            np.clip(event_time_out, t_start, t_stop),
            event_channel_out,
            name="Layer spikes",
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @property
    def weights(self):
        if hasattr(self, "_rec_synapses"):
            return np.reshape(self._rec_synapses.w, (self.size, -1))
        else:
            return self._weights

    @weights.setter
    def weights(self, new_w):
        assert new_w is not None, "Layer `{}`: weights must not be None.".format(
            self.name
        )

        assert np.size(new_w) == self.size ** 2, (
            "Layer `{}`: `new_w` must have ["
            + str(self.size ** 2)
            + "] elements.".format(self.name)
        )

        self._weights = new_w

        if hasattr(self, "_rec_synapses"):
            # - Assign recurrent weights (need to transpose)
            new_w = np.asarray(new_w).reshape(self.size, -1)
            self._rec_synapses.w = new_w.flatten()

    @property
    def state(self):
        return self._neuron_group.v_

    @state.setter
    def state(self, new_state):
        self._neuron_group.v = (
            np.asarray(self._expand_to_net_size(new_state, "new_state")) * volt
        )

    @property
    def refractory(self):
        return self._neuron_group._refractory

    @property
    def tau_mem(self):
        return self._neuron_group.tau_m_

    @tau_mem.setter
    def tau_mem(self, new_tau_mem):
        self._neuron_group.tau_m = (
            np.asarray(self._expand_to_net_size(new_tau_mem, "new_tau_mem")) * second
        )

    @property
    def tau_syn_r(self):
        return self._neuron_group.tau_s_

    @tau_syn_r.setter
    def tau_syn_r(self, vtNewTauSynR):
        self._neuron_group.tau_s = (
            np.asarray(self._expand_to_net_size(vtNewTauSynR, "vtNewTauSynR")) * second
        )

    @property
    def bias(self):
        return self._neuron_group.I_bias_

    @bias.setter
    def bias(self, new_bias):
        self._neuron_group.I_bias = (
            np.asarray(self._expand_to_net_size(new_bias, "new_bias")) * amp
        )

    @property
    def v_thresh(self):
        return self._neuron_group.v_thresh_

    @v_thresh.setter
    def v_thresh(self, new_v_thresh):
        self._neuron_group.v_thresh = (
            np.asarray(self._expand_to_net_size(new_v_thresh, "new_v_thresh")) * volt
        )

    @property
    def v_rest(self):
        return self._neuron_group.v_rest_

    @v_rest.setter
    def v_rest(self, new_v_rest):
        self._neuron_group.v_rest = (
            np.asarray(self._expand_to_net_size(new_v_rest, "new_v_rest")) * volt
        )

    @property
    def v_reset(self):
        return self._neuron_group.v_reset_

    @v_reset.setter
    def v_reset(self, new_v_reset):
        self._neuron_group.v_reset = (
            np.asarray(self._expand_to_net_size(new_v_reset, "new_v_reset")) * volt
        )

    @property
    def t(self):
        return self._net.t_

    @Layer.dt.setter
    def dt(self, _):
        raise ValueError(
            "Layer `{}`: The `dt` property cannot be set for this layer".format(
                self.name
            )
        )


# - RecIAFSpkInBrian - Class: Spiking recurrent layer with spiking in- and outputs
class RecIAFSpkInBrian(RecIAFBrian):
    """ RecIAFSpkInBrian - Class: Spiking recurrent layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        bias: np.ndarray = 10.5 * mA,
        dt: float = 0.1 * ms,
        noise_std: float = 0 * mV,
        tau_mem: np.ndarray = 20 * ms,
        tau_syn_inp: np.ndarray = 50 * ms,
        tau_syn_rec: np.ndarray = 50 * ms,
        v_thresh: np.ndarray = -55 * mV,
        v_reset: np.ndarray = -65 * mV,
        v_rest: np.ndarray = -65 * mV,
        refractory=0 * ms,
        neuron_eq=eqNeuronIAFSpkInRec,
        synapse_eq=eqSynapseExpSpkInRec,
        integrator_name: str = "rk4",
        name: str = "unnamed",
        record: bool = False,
    ):
        """
        RecIAFSpkInBrian - Construct a spiking recurrent layer with IAF neurons, with a Brian2 back-end
                           in- and outputs are spiking events

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10.5mA

        :param dt:             float Time-step. Default: 0.1 ms
        :param noise_std:       float Noise std. dev. per second. Default: 0

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param tau_syn_inp:       np.array Nx1 vector of synapse time constants. Default: 20ms
        :param tau_syn_rec:       np.array Nx1 vector of synapse time constants. Default: 20ms

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param refractory: float Refractory period after each spike. Default: 0ms

        :param neuron_eq:       Brian2.Equations set of neuron equations. Default: IAF equation set
        :param synapse_eq:      Brian2.Equations set of synapse equations for recurrent connects. Default: exponential

        :param integrator_name:   str Integrator to use for simulation. Default: 'rk4'

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions
        """

        warn("RecIAFSpkInBrian: This layer is deprecated. You can use RecIAFSpkInTorch or RecIAFSpkInNest instad.")
        # - Call Layer constructor
        Layer.__init__(
            self,
            weights=weights_in,
            dt=np.asarray(dt),
            noise_std=np.asarray(noise_std),
            name=name,
        )

        # - Set up spike source to receive spiking input
        self._input_generator = b2.SpikeGeneratorGroup(
            self.size_in, [0], [0 * second], dt=np.asarray(dt) * second
        )
        # - Set up layer neurons
        self._neuron_group = b2.NeuronGroup(
            self.size,
            neuron_eq + synapse_eq,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=np.asarray(refractory) * second,
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="spiking_ff_neurons",
        )
        self._neuron_group.v = v_rest
        self._neuron_group.r_m = 1 * ohm

        # - Add source -> receiver synapses
        self._inp_synapses = b2.Synapses(
            self._input_generator,
            self._neuron_group,
            model="w : 1",
            on_pre="I_syn_inp_post += w*amp",
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="receiver_synapses",
        )
        self._inp_synapses.connect()

        # - Add recurrent synapses
        self._rec_synapses = b2.Synapses(
            self._neuron_group,
            self._neuron_group,
            model="w : 1",
            on_pre="I_syn_rec_post += w*amp",
            method=integrator_name,
            dt=np.asarray(dt) * second,
            name="recurrent_synapses",
        )
        self._rec_synapses.connect()

        # - Add monitors to record layer outputs
        self._spike_monitor = b2.SpikeMonitor(
            self._neuron_group, record=True, name="layer_spikes"
        )

        # - Call Network constructor
        self._net = b2.Network(
            self._input_generator,
            self._inp_synapses,
            self._rec_synapses,
            self._neuron_group,
            self._spike_monitor,
            name="rec_spiking_layer",
        )

        if record:
            # - Monitor for recording network potential
            self._v_monitor = b2.StateMonitor(
                self._neuron_group,
                ["v", "I_syn_inp", "I_syn_rec"],
                record=True,
                name="layer_neurons",
            )
            self._net.add(self._v_monitor)

        # - Record neuron parameters
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.tau_syn_inp = tau_syn_inp
        self.tau_syn_rec = tau_syn_rec
        self.bias = bias
        self.weights_in = weights_in
        self.weights_rec = weights_rec

        self._neuron_eq = neuron_eq
        self._synapse_eq = synapse_eq

        # - Store "reset" state
        self._net.store("reset")

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
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
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)
        time_base = self.t + np.arange(num_timesteps) * self.dt

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
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.noise_std
            * np.sqrt(2.0 * self.tau_mem / self.dt)
            * 1.63
        )

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

        # - Start and stop times for output time series
        t_start = self._timestep * np.asscalar(self.dt)
        t_stop = (self._timestep + num_timesteps) * np.asscalar(self.dt)

        # - Update layer time step
        self._timestep += num_timesteps

        # - Build response TimeSeries
        use_event = self._spike_monitor.t_ >= time_base[0]
        event_time_out = self._spike_monitor.t_[use_event]
        event_channel_out = self._spike_monitor.i[use_event]

        return TSEvent(
            np.clip(event_time_out, t_start, t_stop),
            event_channel_out,
            name="Layer spikes",
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

    def reset_time(self):

        # - Store state variables
        v_state = np.copy(self._neuron_group.v) * volt
        v_syn_rec = np.copy(self._neuron_group.I_syn_rec) * amp
        v_syn_inp = np.copy(self._neuron_group.I_syn_inp) * amp

        # - Store parameters
        v_thresh = np.copy(self.v_thresh)
        v_reset = np.copy(self.v_reset)
        v_rest = np.copy(self.v_rest)
        tau_mem = np.copy(self.tau_mem)
        tau_syn_inp = np.copy(self.tau_syn_inp)
        tau_syn_rec = np.copy(self.tau_syn_rec)
        bias = np.copy(self.bias)
        weights_in = np.copy(self.weights_in)
        weights_rec = np.copy(self.weights_rec)

        self._net.restore("reset")
        self._timestep = 0

        # - Restork parameters
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.tau_mem = tau_mem
        self.tau_syn_inp = tau_syn_inp
        self.tau_syn_rec = tau_syn_rec
        self.bias = bias
        self.weights_in = weights_in
        self.weights_rec = weights_rec

        # - Restore state variables
        self._neuron_group.v = v_state
        self._neuron_group.I_syn_inp = v_syn_inp
        self._neuron_group.I_syn_rec = v_syn_rec

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._neuron_group.v = self.v_rest * volt
        self._neuron_group.I_syn_inp = 0 * amp
        self._neuron_group.I_syn_rec = 0 * amp

    def reset_all(self, keep_params=True):
        if keep_params:
            # - Store parameters
            v_thresh = np.copy(self.v_thresh)
            v_reset = np.copy(self.v_reset)
            v_rest = np.copy(self.v_rest)
            tau_mem = np.copy(self.tau_mem)
            tau_syn_rec = np.copy(self.tau_syn_rec)
            tau_syn_inp = np.copy(self.tau_syn_inp)
            bias = np.copy(self.bias)
            weights_in = np.copy(self.weights_in)
            weights_rec = np.copy(self.weights_rec)

        self.reset_state()
        self._net.restore("reset")
        self._timestep = 0

        if keep_params:
            # - Restork parameters
            self.v_thresh = v_thresh
            self.v_reset = v_reset
            self.v_rest = v_rest
            self.tau_mem = tau_mem
            self.tau_syn_inp = tau_syn_inp
            self.tau_syn_rec = tau_syn_rec
            self.bias = bias
            self.weights_in = weights_in
            self.weights_rec = weights_rec

    def randomize_state(self):
        """ .randomize_state() - arguments:: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        v_range = abs(self.v_thresh - self.v_reset)
        self._neuron_group.v = (
            np.random.rand(self.size) * v_range + self.v_reset
        ) * volt
        self._neuron_group.I_syn_inp = (
            np.random.randn(self.size) * np.mean(np.abs(self.weights_in)) * amp
        )
        self._neuron_group.I_syn_rec = (
            np.random.randn(self.size) * np.mean(np.abs(self.weights_rec)) * amp
        )

    def to_dict(self) -> dict:
        """
        to_dict - Convert parameters of `self` to a dict if they are relevant for
                  reconstructing an identical layer.
        """
        config = super().to_dict()
        config.pop("weights")
        config.pop("tau_syn_r")
        config.pop("rec_syn_eq")
        config["weights_in"] = self.weights_in
        config["weights_rec"] = self.weights_rec
        config["tau_syn_inp"] = self.tau_syn_inp.tolist()
        config["tau_syn_rec"] = self.tau_syn_rec.tolist()
        config["synapse_eq"] = self._synapse_eq

        return config

    @property
    def input_type(self):
        return TSEvent

    @property
    def refractory(self):
        return self._neuron_group._refractory

    @property
    def weights(self):
        return self.weights_rec

    @weights.setter
    def weights(self, new_w):
        self.weights_rec = new_w

    @property
    def weights_in(self):
        return np.array(self._inp_synapses.w).reshape(self.size_in, self.size)

    @weights_in.setter
    def weights_in(self, new_w):
        assert new_w is not None, "Layer `{}`: weights_in must not be None.".format(
            self.name
        )

        assert (
            new_w.shape == (self.size_in, self.size)
            or new_w.shape == self._inp_synapses.w.shape
        ), "Layer `{}`: weights must be of dimensions ({}, {}) or flat with size {}.".format(
            self.name, self.size_in, self.size, self.size_in * self.size
        )

        self._inp_synapses.w = np.array(new_w).flatten()

    @property
    def weights_rec(self):
        return np.array(self._rec_synapses.w).reshape(self.size, self.size)

    @weights_rec.setter
    def weights_rec(self, new_w):
        assert new_w is not None, "Layer `{}`: weights_rec must not be None.".format(
            self.name
        )

        assert (
            new_w.shape == (self.size, self.size)
            or new_w.shape == self._inp_synapses.w.shape
        ), "Layer `{}`: weights_rec must be of dimensions ({}, {}) or flat with size {}.".format(
            self.name, self.size, self.size, self.size * self.size
        )

        self._rec_synapses.w = np.array(new_w).flatten()

    @property
    def tau_syn_inp(self):
        return self._neuron_group.tau_syn_inp

    @tau_syn_inp.setter
    def tau_syn_inp(self, new_tau_syn):
        self._neuron_group.tau_syn_inp = (
            np.asarray(self._expand_to_net_size(new_tau_syn, "tau_syn_inp")) * second
        )

    @property
    def tau_syn_rec(self):
        return self._neuron_group.tau_syn_rec

    @tau_syn_rec.setter
    def tau_syn_rec(self, new_tau_syn):
        self._neuron_group.tau_syn_rec = (
            np.asarray(self._expand_to_net_size(new_tau_syn, "tau_syn_rec")) * second
        )

    @property
    def tau_syn_r(self):
        print(
            "Layer {}: This layer has no attribute `tau_syn_r`. ".format(self.name)
            + "You might want to consider `tau_syn_rec` or `tau_syn_inp`."
        )

    @tau_syn_r.setter
    def tau_syn_r(self, *args, **kwargs):
        print(
            "Layer {}: This layer has no attribute `tau_syn_r`. ".format(self.name)
            + "You might want to consider `tau_syn_rec` or `tau_syn_inp`."
        )
