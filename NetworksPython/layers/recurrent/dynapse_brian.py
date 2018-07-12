###
# dynapse_brian.py - Class implementing a recurrent layer in Brian using Dynap equations from teili lib
###

# - Imports
import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

import sys
strNetworkPath = sys.path[0] + '../../..'
sys.path.insert(1, strNetworkPath)

from NetworksPython.timeseries import TSContinuous, TSEvent

from NetworksPython.layers.layer import Layer

from NetworksPython.layers.recurrent.timedarray_shift import TimedArray as TAShift

# - Teili
from teili import Neurons as teiliNG, Connections as teiliSyn, teiliNetwork
from teili.models.neuron_models import DPI as teiliDPIEqts
from teili.models.synapse_models import DPISyn as teiliDPISynEqts
from teili.models.parameters.dpi_neuron_param import parameters as dTeiliNeuronParam

# - Configure exports
__all__ = ['RecDynapseBrian']


## - RecIAFBrian - Class: define a spiking recurrent layer based on Dynap equations
class RecDynapseBrian(Layer):
    """ RecIAFBrian - Class: define a spiking recurrent layer based on Dynap equations
    """

    ## - Constructor
    def __init__(self,
                 mfW: np.ndarray = None,
                 vfWIn: np.ndarray = None,

                 tDt: float = 0.1*ms,
                 fNoiseStd: float = 1*mV,

                 tRefractoryTime = 0*ms,

                 dParamNeuron = None,

                 dParamsSynapse = None,

                 strIntegrator: str = 'rk4',

                 strName: str = 'unnamed'
                 ):
        """
        RecIAFBrian - Construct a spiking recurrent layer with IAF neurons, with a Brian2 back-end

        :param mfW:             np.array NxN weight matrix
        :param mfW:             np.array 1xN input weight matrix.

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param dParamNeuron:    dict Parameters to over overwriting neuron defaulst

        :param dParamSynapse:    dict Parameters to over overwriting synapse defaulst

        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param strName:         str Name for the layer. Default: 'unnamed'
        """

        # - Call super constructor
        super().__init__(mfW = mfW,
                         tDt = np.asarray(tDt),
                         fNoiseStd = np.asarray(fNoiseStd),
                         strName = strName)

        # - Input weights must be provided
        assert vfWIn is not None, 'vfWIn must be provided.'

        # - Set up spike source to receive spiking input
        self._sggInput = b2.SpikeGeneratorGroup(self.nSize, [0], [0*second],
                                                dt = np.asarray(tDt) * second)

        # - Set up reservoir neurons
        self._ngLayer = teiliNG(
            N = self.nSize,
            equation_builder = teiliDPIEqts(num_inputs=2),
            name = 'reservoir_neurons',
            refractory = tRefractoryTime,
            method = strIntegrator,
            dt = tDt,
        )
        
        # - Overwrite default neuron parameters
        if dParamNeuron is not None:
            self._ngLayer.set_params(dict(dTeiliNeuronParam, **dParamNeuron))
        else:
            self._ngLayer.set_params(dTeiliNeuronParam)

        # - Add recurrent weights (all-to-all)
        self._sgRecurrentSynapses = teiliSyn(
            self._ngLayer, self._ngLayer,
            equation_builder=teiliDPISynEqts,
            method = strIntegrator,
            dt = tDt,
            name = 'reservoir_recurrent_synapses'
        )
        self._sgRecurrentSynapses.connect()

        # - Add source -> reservoir synapses
        self._sgReceiver = teiliSyn(
            self._sggInput, self._ngLayer,
            equation_builder=teiliDPISynEqts,
            method = strIntegrator,
            dt = np.asarray(tDt) * second,
            name = 'receiver_synapses')
        # Each spike generator neuron corresponds to one reservoir neuron
        self._sgReceiver.connect('i==j')

        # - Add current monitors to record layer outputs
        self._spmReservoir = b2.SpikeMonitor(self._ngLayer, record = True, name = 'layer_spikes')

        # - Call Network constructor
        self._net = teiliNetwork(self._ngLayer, self._sgRecurrentSynapses,
                                 self._spmReservoir,
                                 name = 'recurrent_spiking_layer')

        # - Record neuron / synapse parameters
        # automatically sets weights  via setters
        self.mfW = mfW
        self.vfWIn = vfWIn

        # - Store "reset" state
        self._net.store('reset')


    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._ngLayer.Imem = 0 * amp
        self._ngLayer.Iahp = 0.5 * pamp

    
    def randomize_state(self):
        """ .randomize_state() - Method: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        fRangeV = abs(self.vfVThresh - self.vfVReset)
        self._ngLayer.v = (np.random.rand(self.nSize) * fRangeV + self.vfVReset) * volt
        self._ngLayer.I_syn = np.random.rand(self.nSize) * amp

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """
        self._net.restore('reset')


    ### --- State evolution

    def evolve(self,
               tsInput: TSContinuous = None,
               tDuration: float = None):
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TimeSeries TxM or Tx1 input to this layer
        :param tDuration:   float Duration of evolution, in seconds

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Discretise input, prepare time base
        vtTimeBase, mfInputStep, tDuration = self._prepare_input(tsInput, tDuration)
        nNumSteps = np.size(vtTimeBase)

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(nNumSteps, self.nSize) * self.fNoiseStd

        # - Specifiy network input currents, construct TimedArray
        taI_inp = TAShift(np.asarray(mfInputStep + mfNoiseStep) * amp,
                          self.tDt * second, tOffset = self.t * second,
                          name  = 'external_input')

        # - Perform simulation
        self._net.run(tDuration * second, namespace = {'I_inp': taI_inp}, level = 0)

        # - Build response TimeSeries
        vbUseEvent = self._spmReservoir.t_ >= vtTimeBase[0]
        vtEventTimeOutput = self._spmReservoir.t[vbUseEvent]
        vnEventChannelOutput = self._spmReservoir.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, strName = 'Layer spikes')

    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

    @property
    def mfW(self):
        if hasattr(self, '_sgRecurrentSynapses'):
            return np.reshape(self._sgRecurrentSynapses.weight, (self.nSize, -1))
        else:
            return self._mfW

    @mfW.setter
    def mfW(self, mfNewW):
        assert np.size(mfNewW) == self.nSize ** 2, \
            '`mfNewW` must have [' + str(self.nSize ** 2) + '] elements.'

        self._mfW = mfNewW

        if hasattr(self, '_sgRecurrentSynapses'):
            # - Assign recurrent weights (need to transpose)
            mfNewW = np.asarray(mfNewW).reshape(self.nSize, -1).T
            self._sgRecurrentSynapses.weight = mfNewW.flatten()

    @property
    def vfWIn(self):
        if hasattr(self, '_sgReceiver'):
            return np.reshape(self._sgReceiver.weight, (self.nSize, -1))
        else:
            return self._mfW

    @mfW.setter
    def mfW(self, mfNewW):
        assert np.size(mfNewW) == self.nSize ** 2, \
            '`mfNewW` must have [' + str(self.nSize ** 2) + '] elements.'

        self._mfW = mfNewW

        if hasattr(self, '_sgRecurrentSynapses'):
            # - Assign recurrent weights (need to transpose)
            mfNewW = np.asarray(mfNewW).reshape(self.nSize, -1).T
            self._sgRecurrentSynapses.weight = mfNewW.flatten()

    @property
    def vState(self):
        return self._ngLayer.v_

    @vState.setter
    def vState(self, vNewState):
        self._ngLayer.v = np.asarray(self._expand_to_net_size(vNewState, 'vNewState')) * volt

    @property
    def vtTauN(self):
        return self._ngLayer.tau_m_

    @vtTauN.setter
    def vtTauN(self, vtNewTauN):
        self._ngLayer.tau_m = np.asarray(self._expand_to_net_size(vtNewTauN, 'vtNewTauN')) * second

    @property
    def vtTauSynR(self):
        return self._ngLayer.tau_s_

    @vtTauSynR.setter
    def vtTauSynR(self, vtNewTauSynR):
        self._ngLayer.tau_s = np.asarray(self._expand_to_net_size(vtNewTauSynR, 'vtNewTauSynR')) * second

    @property
    def vfBias(self):
        return self._ngLayer.I_bias_

    @vfBias.setter
    def vfBias(self, vfNewBias):
        self._ngLayer.I_bias = np.asarray(self._expand_to_net_size(vfNewBias, 'vfNewBias')) * amp

    @property
    def vfVThresh(self):
        return self._ngLayer.v_thresh_

    @vfVThresh.setter
    def vfVThresh(self, vfNewVThresh):
        self._ngLayer.v_thresh = np.asarray(self._expand_to_net_size(vfNewVThresh, 'vfNewVThresh')) * volt

    @property
    def vfVRest(self):
        return self._ngLayer.v_rest_

    @vfVRest.setter
    def vfVRest(self, vfNewVRest):
        self._ngLayer.v_rest = np.asarray(self._expand_to_net_size(vfNewVRest, 'vfNewVRest')) * volt

    @property
    def vfVReset(self):
        return self._ngLayer.v_reset_

    @vfVReset.setter
    def vfVReset(self, vfNewVReset):
        self._ngLayer.v_reset = np.asarray(self._expand_to_net_size(vfNewVReset, 'vfNewVReset')) * volt

    @property
    def t(self):
        return self._net.t_

    @Layer.tDt.setter
    def tDt(self, _):
        raise ValueError('The `tDt` property cannot be set for this layer')
