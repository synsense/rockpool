###
# exp_synapses_brian.py - Class implementing a spike-to-current layer with exponential synapses
###


# - Imports
import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

from TimeSeries import TSContinuous, TSEvent
from typing import Union

from ..layer import Layer
from ..recurrent.timedarray_shift import TimedArray as TAShift

# - Configure exports
__all__ = ['FFExpSynBrian', 'eqSynapseExp']

# - Equations for an exponential synapse
eqSynapseExp = b2.Equations('''
    dI_syn/dt = (-I_syn + I_inp(t, i)) / tau_s  : amp                       # Synaptic current
    tau_s                                       : second                    # Synapse time constant
''')


## - FFExpSynBrian - Class: define an exponential synapse layer (spiking input)
class FFExpSynBrian(Layer):
    """ FFExpSynBrian - Class: define an exponential synapse layer (spiking input)
    """

    ## - Constructor
    def __init__(self,
                 mfW: Union[np.ndarray, int] = None,

                 tDt: float = 0.1*ms,
                 fNoiseStd: float = 0*mV,

                 tTauSyn: float = 5 * ms,
                 eqSynapses = eqSynapseExp,
                 strIntegrator: str = 'rk4',

                 strName: str = 'unnamed'
                 ):
        """
        FFExpSynBrian - Construct an exponential synapse layer (spiking input)

        :param mfW:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param tDt:             float Time step for state evolution
        :param fNoiseStd:       float Std. dev. of noise added to this layer. Default: 0

        :param tTauSyn:         float Output synaptic time constants. Default: 5ms
        :param eqSynapses:      Brian2.Equations set of synapse equations for receiver. Default: exponential
        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param strName:         str Name for the layer. Default: 'unnamed'
        """

        # - Provide default weight matrix for one-to-one conversion
        if isinstance(mfW, int):
            mfW = np.identity(mfW, 'float')

        # - Call super constructor
        super().__init__(mfW = mfW,
                         tDt = np.asarray(tDt),
                         fNoiseStd = np.asarray(fNoiseStd),
                         strName = strName)

        # - Set up spike source to receive spiking input
        self._sggInput = b2.SpikeGeneratorGroup(self.nDimIn, [0], [0*second],
                                                dt = np.asarray(tDt) * second)

        # - Set up layer receiver nodes
        self._ngReceiver = b2.NeuronGroup(self.nSize, eqSynapses,
                                          refractory = False,
                                          method = strIntegrator,
                                          dt = np.asarray(tDt) * second,
                                          name = 'receiver_neurons')

        # - Add source -> receiver synapses
        self._sgReceiver = b2.Synapses(self._sggInput, self._ngReceiver,
                                       model = 'w : 1',
                                       on_pre = 'I_syn_post += w*amp',
                                       method = strIntegrator,
                                       dt = np.asarray(tDt) * second,
                                       name = 'receiver_synapses')
        self._sgReceiver.connect()

        # - Add current monitors to record reservoir outputs
        self._stmReceiver = b2.StateMonitor(self._ngReceiver, 'I_syn', True, name = 'receiver_synaptic_currents')

        # - Call Network constructor
        self._net = b2.Network(self._sggInput, self._ngReceiver, self._sgReceiver,
                               self._stmReceiver,
                               name = 'ff_spiking_to_exp_layer')

        # - Record layer parameters, set weights
        self.mfW = mfW
        self.tTauSyn = tTauSyn

        # - Store "reset" state
        self._net.store('reset')


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
        self._net.restore('reset')


    ### --- State evolution

    def evolve(self,
               tsInput: TSEvent = None,
               tDuration: float = None):
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TSEvent spikes as input to this layer
        :param tDuration:   float Duration of evolution, in seconds

        :return: TimeSeries Output of this layer during evolution period
        """
        
        # - Prepare time base
        vtTimeBase, _, tDuration = self._prepare_input(tsInput, tDuration)

        # - Set spikes for spike generator
        if tsInput is not None:
            vtEventTimes, vnEventChannels, _ = tsInput.find([vtTimeBase[0], vtTimeBase[-1]+self.tDt])
            self._sggInput.set_spikes(vnEventChannels, vtEventTimes * second, sorted = False)
        else:
            self._sggInput.set_spikes([], [] * second)

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(np.size(vtTimeBase), self.nSize) * self.fNoiseStd / np.sqrt(self.tDt)
        #mfNoiseStep = np.zeros((np.size(vtTimeBase), self.nSize))
        #mfNoiseStep[0,:] = self.fNoiseStd

        # - Specifiy noise input currents, construct TimedArray
        taI_noise = TAShift(np.asarray(mfNoiseStep) * amp,
                          self.tDt * second, tOffset = self.t * second,
                          name  = 'noise_input')

        # - Perform simulation
        self._net.run(tDuration * second, namespace = {'I_inp': taI_noise}, level = 0)

        # - Build response TimeSeries
        vtTimeBaseOutput = self._stmReceiver.t_
        vbUseTime = self._stmReceiver.t_ >= vtTimeBase[0]
        vtTimeBaseOutput = vtTimeBaseOutput[vbUseTime]
        mfA = self._stmReceiver.I_syn_.T
        mfA = mfA[vbUseTime, :]

        # - Return the current state as final time point
        if vtTimeBaseOutput[-1] != self.t:
            vtTimeBaseOutput = np.concatenate((vtTimeBaseOutput, [self.t]))
            mfA = np.concatenate((mfA, np.reshape(self.vState, (1, self.nSize))))

        return TSContinuous(vtTimeBaseOutput, mfA, strName = 'Receiver current')


    ### --- Properties

    @property
    def cInput(self):
        return TSEvent

    @property
    def mfW(self):
        if hasattr(self, '_sgReceiver'):
            return np.reshape(self._sgReceiver.w, (self.nSize, -1))
        else:
            return self._mfW

    @mfW.setter
    def mfW(self, mfNewW):
        assert np.size(mfNewW) == self.nSize * self.nDimIn, \
            '`mfNewW` must have [' + str(self.nSize * self.nDimIn) + '] elements.'

        self._mfW = mfNewW

        if hasattr(self, '_sgReceiver'):
            # - Assign recurrent weights
            mfNewW = np.asarray(mfNewW).reshape(self.nSize, -1)
            self._sgReceiver.w = mfNewW.flatten()

    @property
    def vState(self):
        return self._ngReceiver.I_syn_

    @vState.setter
    def vState(self, vNewState):
        self._ngReceiver.I_syn = np.asarray(self._expand_to_net_size(vNewState, 'vNewState')) * amp

    @property
    def tTauSyn(self):
        return self._ngReceiver.tau_s_[0]

    @tTauSyn.setter
    def tTauSyn(self, tNewTau):
        self._ngReceiver.tau_s = np.asarray(tNewTau) * second

    @property
    def t(self):
        return self._net.t_

    @Layer.tDt.setter
    def tDt(self, _):
        raise ValueError('The `tDt` property cannot be set for this layer')