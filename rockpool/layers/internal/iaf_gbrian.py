###
# iaf_gfbrian.py - Classes implementing recurrent and feedforward layers consisting of standard IAF neurons in brian2genn
###


# - Imports
import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

# - Use GeNN
import brian2genn

b2.set_device("genn")

from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer
from .timedarray_shift import TimedArray as TAShift

from typing import Optional, Union, Tuple, List

# - Configure exports
__all__ = [
    "FFIAFBrian",
    "FFIAFSpkInBrian",
    "eqNeuronIAFSpkInFF",
    "eqNeuronIAFSpkInRec",
    "eqSynapseExp",
    "eqSynapseExpSpkInRec",
]

# - Equations for an integrate-and-fire neuron, ff-layer, analogue external input
eqNeuronIAFFF = b2.Equations(
    """
    dv/dt = (v_rest - v + r_m * I_total) / tau_m    : volt (unless refractory)  # Neuron membrane voltage
    I_total = I_bias                  : amp                       # Total input current
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
    I_total = I_syn + I_bias          : amp                       # Total input current
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
        refractory=0 * ms,
        neuron_eq=eqNeuronIAFFF,
        integrator_name: str = "rk4",
        name: str = "unnamed",
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
        """

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
        self.time_step = 0

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
        self.time_step += num_timesteps

        # - Build response TimeSeries
        use_event = self._layer.t_ >= time_base[0]
        event_time_out = self._layer.t_[use_event]
        event_channel_out = self._layer.i[use_event]

        return TSEvent(event_time_out, event_channel_out, name="Layer spikes")

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
        refractory=0 * ms,
        neuron_eq=eqNeuronIAFSpkInFF,
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

        b2.defaultclock.dt = dt

        # - Call Layer constructor
        Layer.__init__(
            self,
            weights=weights,
            dt=np.asarray(dt),
            noise_std=np.asarray(noise_std),
            name=name,
        )

        # - Set up spike source to receive spiking input
        self._input_generator = b2.SpikeGeneratorGroup(self.size_in, [0], [0 * second])
        # - Set up layer neurons
        self._neuron_group = b2.NeuronGroup(
            self.size,
            neuron_eq,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=np.asarray(refractory) * second,
            method=integrator_name,
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
        # self._net.store("reset")

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
        time_base, __, num_timesteps = self._prepare_input(
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
        # inp_noise = TAShift(
        #     np.asarray(noise_step) * amp,
        #     self.dt * second,
        #     tOffset=self.t * second,
        #     name="noise_input",
        # )

        # - Perform simulation
        self._net.run(num_timesteps * self.dt * second, level=0)
        self.time_step += num_timesteps

        # - Build response TimeSeries
        use_event = self._layer.t_ >= time_base[0]
        event_time_out = self._layer.t_[use_event]
        event_channel_out = self._layer.i[use_event]

        return TSEvent(event_time_out, event_channel_out, name="Layer spikes")

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
        self.time_step = 0

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
        self.time_step = 0

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
            event_times, event_channels, _ = ts_input.find([t_start, t_stop])
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
