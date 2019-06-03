###
# exp_synapses_brian.py - Class implementing a spike-to-current layer with exponential synapses
###


# - Imports
import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

from ...timeseries import TSContinuous, TSEvent
from ..layer import Layer
from .timedarray_shift import TimedArray as TAShift

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
class FFExpSynBrian(Layer):
    """ FFExpSynBrian - Class: define an exponential synapse layer (spiking input)
    """

    ## - Constructor
    def __init__(
        self,
        weights: Union[np.ndarray, int] = None,
        dt: float = 0.1 * ms,
        noise_std: float = 0 * mV,
        tTauSyn: float = 5 * ms,
        eqSynapses=eqSynapseExp,
        strIntegrator: str = "rk4",
        name: str = "unnamed",
    ):
        """
        FFExpSynBrian - Construct an exponential synapse layer (spiking input)

        :param weights:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param dt:             float Time step for state evolution. Default: 0.1 ms
        :param noise_std:       float Std. dev. of noise added to this layer. Default: 0

        :param tTauSyn:         float Output synaptic time constants. Default: 5ms
        :param eqSynapses:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'
        """

        # - Provide default dt
        if dt is None:
            dt = 0.1 * ms

        # - Provide default weight matrix for one-to-one conversion
        if isinstance(weights, int):
            weights = np.identity(weights, "float")

        # - Call super constructor
        super().__init__(
            weights=weights,
            dt=np.asarray(dt),
            noise_std=np.asarray(noise_std),
            name=name,
        )

        # - Set up spike source to receive spiking input
        self._sggInput = b2.SpikeGeneratorGroup(
            self.size_in, [0], [0 * second], dt=np.asarray(dt) * second
        )

        # - Set up layer receiver nodes
        self._ngReceiver = b2.NeuronGroup(
            self.size,
            eqSynapses,
            refractory=False,
            method=strIntegrator,
            dt=np.asarray(dt) * second,
            name="receiver_neurons",
        )

        # - Add source -> receiver synapses
        self._sgReceiver = b2.Synapses(
            self._sggInput,
            self._ngReceiver,
            model="w : 1",
            on_pre="I_syn_post += w*amp",
            method=strIntegrator,
            dt=np.asarray(dt) * second,
            name="receiver_synapses",
        )
        self._sgReceiver.connect()

        # - Add current monitors to record reservoir outputs
        self._stmReceiver = b2.StateMonitor(
            self._ngReceiver, "I_syn", True, name="receiver_synaptic_currents"
        )

        # - Call Network constructor
        self._net = b2.Network(
            self._sggInput,
            self._ngReceiver,
            self._sgReceiver,
            self._stmReceiver,
            name="ff_spiking_to_exp_layer",
        )

        # - Record layer parameters, set weights
        self.weights = weights
        self.tTauSyn = tTauSyn

        # - Store "reset" state
        self._net.store("reset")

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._ngReceiver.I_syn = 0 * amp

    def randomize_state(self):
        """ .randomize_state() - Method: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        self.reset_state()

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        # - Sotre state variables
        vfIsyn = np.copy(self._ngReceiver.I_syn) * amp

        # - Store parameters
        tTauSyn = np.copy(self.tTauSyn)
        weights = np.copy(self.weights)

        # - Reset network
        self._net.restore("reset")
        self._timestep = 0

        # - Restork parameters
        self.tTauSyn = tTauSyn
        self.weights = weights

        # - Restore state variables
        self._ngReceiver.I_syn = vfIsyn

    ### --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSContinuous:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSContinuous  output spike series

        """

        # - Prepare time base
        vtTimeBase, __, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Set spikes for spike generator
        if ts_input is not None:
            vtEventTimes, vnEventChannels, _ = ts_input(
                t_start=vtTimeBase[0], t_stop=vtTimeBase[-1] + self.dt
            )
            self._sggInput.set_spikes(
                vnEventChannels, vtEventTimes * second, sorted=False
            )
        else:
            self._sggInput.set_spikes([], [] * second)

        # - Generate a noise trace
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.size)
            * self.noise_std
            * np.sqrt(2 * self.tTauSyn / self.dt)
        )
        # mfNoiseStep = np.zeros((np.size(vtTimeBase), self.size))
        # mfNoiseStep[0,:] = self.noise_std

        # - Specifiy noise input currents, construct TimedArray
        taI_noise = TAShift(
            np.asarray(mfNoiseStep) * amp,
            self.dt * second,
            tOffset=self.t * second,
            name="noise_input",
        )

        # - Perform simulation
        self._net.run(
            num_timesteps * self.dt * second, namespace={"I_inp": taI_noise}, level=0
        )
        self._timestep += num_timesteps

        # - Build response TimeSeries
        vtTimeBaseOutput = self._stmReceiver.t_
        vbUseTime = self._stmReceiver.t_ >= vtTimeBase[0]
        vtTimeBaseOutput = vtTimeBaseOutput[vbUseTime]
        mfA = self._stmReceiver.I_syn_.T
        mfA = mfA[vbUseTime, :]

        # - Return the current state as final time point
        if vtTimeBaseOutput[-1] != self.t:
            vtTimeBaseOutput = np.concatenate((vtTimeBaseOutput, [self.t]))
            mfA = np.concatenate((mfA, np.reshape(self.state, (1, self.size))))

        return TSContinuous(vtTimeBaseOutput, mfA, name="Receiver current")

    ### --- Properties

    @property
    def input_type(self):
        return TSEvent

    @property
    def weights(self):
        if hasattr(self, "_sgReceiver"):
            return np.reshape(self._sgReceiver.w, (self.size, -1))
        else:
            return self._mfW

    @weights.setter
    def weights(self, mfNewW):
        assert np.size(mfNewW) == self.size * self.size_in, (
            "`mfNewW` must have [" + str(self.size * self.size_in) + "] elements."
        )

        self._mfW = mfNewW

        if hasattr(self, "_sgReceiver"):
            # - Assign recurrent weights
            mfNewW = np.asarray(mfNewW).reshape(self.size, -1)
            self._sgReceiver.w = mfNewW.flatten()

    @property
    def state(self):
        return self._ngReceiver.I_syn_

    @state.setter
    def state(self, vNewState):
        self._ngReceiver.I_syn = (
            np.asarray(self._expand_to_net_size(vNewState, "vNewState")) * amp
        )

    @property
    def tTauSyn(self):
        return self._ngReceiver.tau_s_[0]

    @tTauSyn.setter
    def tTauSyn(self, tNewTau):
        self._ngReceiver.tau_s = np.asarray(tNewTau) * second

    @property
    def t(self):
        return self._net.t_

    @Layer.dt.setter
    def dt(self, _):
        raise ValueError("The `dt` property cannot be set for this layer")
