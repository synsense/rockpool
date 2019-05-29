###
# dynapse_brian.py - Class implementing a recurrent layer in Brian using Dynap equations from teili lib
###

# - Imports
import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

import sys

strNetworkPath = sys.path[0] + "../../.."
sys.path.insert(1, strNetworkPath)

from NetworksPython.timeseries import TSContinuous, TSEvent

from NetworksPython.layers import Layer

from NetworksPython.layers import TimedArray as TAShift

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
        vfWIn: np.ndarray,
        dt: float = 0.1 * ms,
        noise_std: float = 0 * mV,
        tRefractoryTime=0 * ms,
        dParamNeuron=None,
        dParamSynapse=None,
        strIntegrator: str = "rk4",
        name: str = "unnamed",
    ):
        """
        RecIAFBrian - Construct a spiking recurrent layer with IAF neurons, with a Brian2 back-end

        :param weights:             np.array NxN weight matrix
        :param vfWIn:             np.array 1xN input weight matrix.

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param dParamNeuron:    dict Parameters to over overwriting neuron defaulst

        :param dParamSynapse:    dict Parameters to over overwriting synapse defaulst

        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param name:         str Name for the layer. Default: 'unnamed'
        """

        # - Call super constructor
        super().__init__(
            weights=weights,
            dt=np.asarray(dt),
            noise_std=np.asarray(noise_std),
            name=name,
        )

        # - Input weights must be provided
        assert vfWIn is not None, "vfWIn must be provided."

        # - Warn that nosie is not implemented
        if noise_std != 0:
            print("WARNING: Noise is currently not implemented in this layer.")

        # - Set up spike source to receive spiking input
        self._sggInput = b2.SpikeGeneratorGroup(
            self.size, [0], [0 * second], dt=np.asarray(dt) * second
        )

        # - Handle unit of dt: if no unit provided, assume it is in seconds
        dt = np.asscalar(np.array(dt)) * second

        ### --- Neurons

        # - Set up reservoir neurons
        self._ngLayer = teiliNG(
            N=self.size,
            equation_builder=teiliDPIEqts(num_inputs=2),
            name="reservoir_neurons",
            refractory=tRefractoryTime,
            method=strIntegrator,
            dt=dt,
        )

        # - Overwrite default neuron parameters
        if dParamNeuron is not None:
            self._ngLayer.set_params(dict(dTeiliNeuronParam, **dParamNeuron))
        else:
            self._ngLayer.set_params(dTeiliNeuronParam)

        ### --- Synapses

        # - Add recurrent synapses (all-to-all)
        self._sgRecurrentSynapses = teiliSyn(
            self._ngLayer,
            self._ngLayer,
            equation_builder=teiliDPISynEqts,
            method=strIntegrator,
            dt=dt,
            name="reservoir_recurrent_synapses",
        )
        self._sgRecurrentSynapses.connect()

        # - Add source -> reservoir synapses (one-to-one)
        self._sgReceiver = teiliSyn(
            self._sggInput,
            self._ngLayer,
            equation_builder=teiliDPISynEqts,
            method=strIntegrator,
            dt=np.asarray(dt) * second,
            name="receiver_synapses",
        )
        # Each spike generator neuron corresponds to one reservoir neuron
        self._sgReceiver.connect("i==j")

        # - Overwrite default synapse parameters
        if dParamSynapse is not None:
            self._sgRecurrentSynapses.set_params(dParamNeuron)
            self._sgReceiver.set_params(dParamNeuron)

        # - Add spike monitor to record layer outputs
        self._spmReservoir = b2.SpikeMonitor(
            self._ngLayer, record=True, name="layer_spikes"
        )

        # - Call Network constructor
        self._net = b2.Network(
            self._ngLayer,
            self._sgRecurrentSynapses,
            self._sggInput,
            self._sgReceiver,
            self._spmReservoir,
            name="recurrent_spiking_layer",
        )

        # - Record neuron / synapse parameters
        # automatically sets weights  via setters
        self.weights = weights
        self.vfWIn = vfWIn

        # - Store "reset" state
        self._net.store("reset")

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._ngLayer.Imem = 0 * amp
        self._ngLayer.Iahp = 0.5 * pamp
        self._sgRecurrentSynapses.Ie_syn = 0.5 * pamp
        self._sgRecurrentSynapses.Ii_syn = 0.5 * pamp
        self._sgReceiver.Ie_syn = 0.5 * pamp
        self._sgReceiver.Ii_syn = 0.5 * pamp

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        # - Save state variables
        Imem = np.copy(self._ngLayer.Imem) * amp
        Iahp = np.copy(self._ngLayer.Iahp) * amp
        Ie_Recur = np.copy(self._sgRecurrentSynapses.Ie_syn) * amp
        Ii_Recur = np.copy(self._sgRecurrentSynapses.Ii_syn) * amp
        Ie_Recei = np.copy(self._sgReceiver.Ie_syn) * amp
        Ii_Recei = np.copy(self._sgReceiver.Ii_syn) * amp

        # - Save parameters
        weights = np.copy(self.weights)
        vfWIn = np.copy(self.vfWIn)

        # - Reset Network
        self._net.restore("reset")
        self._timestep = 0

        # - Restore state variables
        self._ngLayer.Imem = Imem
        self._ngLayer.Iahp = Iahp
        self._sgRecurrentSynapses.Ie_syn = Ie_Recur
        self._sgRecurrentSynapses.Ii_syn = Ii_Recur
        self._sgReceiver.Ie_syn = Ie_Recei
        self._sgReceiver.Ii_syn = Ii_Recei

        # - Restore parameters
        self.weights = weights
        self.vfWIn = vfWIn

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
        vtTimeBase, mfInputStep, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Set spikes for spike generator
        if ts_input is not None:
            vtEventTimes, vnEventChannels, _ = ts_input.find(
                [vtTimeBase[0], vtTimeBase[-1] + self.dt]
            )
            self._sggInput.set_spikes(
                vnEventChannels, vtEventTimes * second, sorted=False
            )
        else:
            self._sggInput.set_spikes([], [] * second)

        # - Perform simulation
        self._net.run(num_timesteps * self.dt * second, level=0)
        self._timestep += num_timesteps

        # - Build response TimeSeries
        vbUseEvent = self._spmReservoir.t_ >= vtTimeBase[0]
        vtEventTimeOutput = self._spmReservoir.t[vbUseEvent]
        vnEventChannelOutput = self._spmReservoir.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, name="Layer spikes")

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @property
    def input_type(self):
        return TSEvent

    @property
    def weights(self):
        if hasattr(self, "_sgRecurrentSynapses"):
            return np.reshape(self._sgRecurrentSynapses.weight, (self.size, -1))
        else:
            return self._mfW

    @weights.setter
    def weights(self, mfNewW):
        assert np.size(mfNewW) == self.size ** 2, (
            "`mfNewW` must have [" + str(self.size ** 2) + "] elements."
        )

        self._mfW = mfNewW

        if hasattr(self, "_sgRecurrentSynapses"):
            # - Assign recurrent weights
            mfNewW = np.asarray(mfNewW).reshape(self.size, -1)
            self._sgRecurrentSynapses.weight = mfNewW.flatten()

    @property
    def vfWIn(self):
        if hasattr(self, "_sgReceiver"):
            return np.reshape(self._sgReceiver.weight, (self.size, -1))
        else:
            return self._mfW

    @vfWIn.setter
    def vfWIn(self, vfNewW):
        assert np.size(vfNewW) == self.size, (
            "`mfNewW` must have [" + str(self.size) + "] elements."
        )

        self._mfW = vfNewW

        if hasattr(self, "_sgReceiver"):
            # - Assign input weights
            self._sgReceiver.weight = vfNewW.flatten()

    @property
    def state(self):
        return self._ngLayer.Imem_

    @state.setter
    def state(self, vNewState):
        self._ngLayer.Imem = (
            np.asarray(self._expand_to_net_size(vNewState, "vNewState")) * volt
        )

    @property
    def t(self):
        return self._net.t_

    @Layer.dt.setter
    def dt(self, _):
        raise ValueError("The `dt` property cannot be set for this layer")
