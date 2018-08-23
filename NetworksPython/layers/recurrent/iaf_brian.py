###
# iaf_brian.py - Class implementing an IAF simple recurrent layer in Brian
###


# - Imports
import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

from typing import Union

from ...timeseries import TSContinuous, TSEvent

from ..layer import Layer

from .timedarray_shift import TimedArray as TAShift

# - Configure exports
__all__ = ['RecIAFBrian', 'eqNeuronIAF', 'eqSynapseExp']

# - Equations for an integrate-and-fire neuron
eqNeuronIAF = b2.Equations('''
    dv/dt = (v_rest - v + r_m * I_total) / tau_m    : volt (unless refractory)  # Neuron membrane voltage
    I_total = I_inp(t, i) + I_syn + I_bias          : amp                       # Total input current
    I_bias                                          : amp                       # Per-neuron bias current
    v_rest                                          : volt                      # Rest potential
    tau_m                                           : second                    # Membrane time constant
    r_m                                             : ohm                       # Membrane resistance
    v_thresh                                        : volt                      # Firing threshold potential
    v_reset                                         : volt                      # Reset potential
''')

# - Equations for an integrate-and-fire neuron, recurrent and external input spikes
eqNeuronIAF2 = b2.Equations('''
    dv/dt = (v_rest - v + r_m * I_total) / tau_m    : volt (unless refractory)  # Neuron membrane voltage
    I_total = I_inp(t, i) + I_syn + I_bias          : amp                       # Total input current
    I_syn = I_syn_inp + I_syn_rec                   : amp                       # Synaptic currents
    I_bias                                          : amp                       # Per-neuron bias current
    v_rest                                          : volt                      # Rest potential
    tau_m                                           : second                    # Membrane time constant
    r_m                                             : ohm                       # Membrane resistance
    v_thresh                                        : volt                      # Firing threshold potential
    v_reset                                         : volt                      # Reset potential
''')

# - Equations for an exponential synapse
eqSynapseExp = b2.Equations('''
    dI_syn/dt = -I_syn / tau_s                      : amp                       # Synaptic current
    tau_s                                           : second                    # Synapse time constant
''')

# - Equations for two exponential synapses (external input and recurrent)
eqSynapseExp2 = b2.Equations('''
    dI_syn_inp/dt = -I_syn_inp / tau_syn_inp        : amp                       # Synaptic current, input synapses
    dI_syn_rec/dt = -I_syn_rec / tau_syn_rec        : amp                       # Synaptic current, recurrent synapses
    tau_syn_inp                                     : second                    # Synapse time constant, input
    tau_syn_rec                                     : second                    # Synapse time constant, recurrent
''')


## - RecIAFBrian - Class: define a spiking recurrent layer with exponential synaptic outputs
class RecIAFBrian(Layer):
    """ RecIAFBrian - Class: define a spiking recurrent layer with exponential synaptic outputs
    """

    ## - Constructor
    def __init__(self,
                 mfW: np.ndarray = None,
                 vfBias: Union[float, np.ndarray] = 10.5*mA,

                 tDt: float = 0.1*ms,
                 fNoiseStd: float = 1*mV,

                 vtTauN: Union[float, np.ndarray] = 20*ms,
                 vtTauSynR: Union[float, np.ndarray] = 5 * ms,

                 vfVThresh: Union[float, np.ndarray] = -55*mV,
                 vfVReset: Union[float, np.ndarray] = -65*mV,
                 vfVRest: Union[float, np.ndarray] = -65*mV,

                 tRefractoryTime = 0*ms,

                 eqNeurons = eqNeuronIAF,
                 eqSynRecurrent = eqSynapseExp,

                 strIntegrator: str = 'rk4',

                 strName: str = 'unnamed'
                 ):
        """
        RecIAFBrian - Construct a spiking recurrent layer with IAF neurons, with a Brian2 back-end

        :param mfW:             np.array NxN weight matrix. Default: [100x100] unit-lambda matrix
        :param vfBias:          np.array Nx1 bias vector. Default: 10.5mA

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 5ms
        :param vtTauSynR:       np.array NxN vector of recurrent synaptic time constants. Default: 5ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param eqNeurons:       Brian2.Equations set of neuron equations. Default: IAF equation set
        :param eqSynRecurrent:  Brian2.Equations set of synapse equations for recurrent connects. Default: exponential

        :param strIntegrator:   str Integrator to use for simulation. Default: 'exact'

        :param strName:         str Name for the layer. Default: 'unnamed'
        """

        # - Call super constructor
        super().__init__(mfW = mfW,
                         tDt = np.asarray(tDt),
                         fNoiseStd = np.asarray(fNoiseStd),
                         strName = strName)

        # - Set up reservoir neurons
        self._ngLayer = b2.NeuronGroup(self.nSize, eqNeurons + eqSynRecurrent,
                                       threshold = 'v > v_thresh',
                                       reset = 'v = v_reset',
                                       refractory = np.asarray(tRefractoryTime) * second,
                                       method = strIntegrator,
                                       dt = np.asarray(tDt) * second,
                                       name = 'reservoir_neurons')
        self._ngLayer.v = vfVRest
        self._ngLayer.r_m = 1 * ohm

        # - Add recurrent weights (all-to-all)
        self._sgRecurrentSynapses = b2.Synapses(self._ngLayer, self._ngLayer,
                                                model = 'w : 1',
                                                on_pre = 'I_syn_post += w*amp',
                                                method = strIntegrator,
                                                dt = tDt,
                                                name = 'reservoir_recurrent_synapses')
        self._sgRecurrentSynapses.connect()

        # - Add current monitors to record layer outputs
        self._spmReservoir = b2.SpikeMonitor(self._ngLayer, record = True, name = 'layer_spikes')

        # - Call Network constructor
        self._net = b2.Network(self._ngLayer, self._sgRecurrentSynapses,
                               self._spmReservoir,
                               name = 'recurrent_spiking_layer')

        # - Record neuron / synapse parameters
        self.mfW = mfW
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vtTauSynR = vtTauSynR
        self.tRefractoryTime = tRefractoryTime
        self.vfBias = vfBias

        # - Store "reset" state
        self._net.store('reset')


    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._ngLayer.v = self.vfVRest * volt
        self._ngLayer.I_syn = 0 * amp

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
        
        # - Store state variables
        vfV = np.copy(self._ngLayer.v) * volt
        vfIsyn = np.copy(self._ngLayer.I_syn) * amp

        # - Store parameters
        vfVThresh = np.copy(self.vfVThresh)
        vfVReset = np.copy(self.vfVReset)
        vfVRest = np.copy(self.vfVRest)
        vtTauN = np.copy(self.vtTauN)
        vtTauSynR = np.copy(self.vtTauSynR)
        tRefractoryTime = np.copy(self.tRefractoryTime)
        vfBias = np.copy(self.vfBias)
        mfW = np.copy(self.mfW)
        
        # - Reset network
        self._net.restore('reset')
        
        # - Restork parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vtTauSynR = vtTauSynR
        self.tRefractoryTime = tRefractoryTime
        self.vfBias = vfBias
        self.mfW = mfW  

        # - Restore state variables
        self._ngLayer.v = vfV
        self._ngLayer.I_syn = vfIsyn

    ### --- State evolution

    def evolve(self,
               tsInput: TSContinuous = None,
               tDuration: float = None,
               bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TimeSeries TxM or Tx1 input to this layer
        :param tDuration:   float Duration of evolution, in seconds
        :param bVerbose:    bool Currently no effect, just for conformity

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Discretise input, prepare time base
        vtTimeBase, mfInputStep, tDuration = self._prepare_input(tsInput, tDuration)
        nNumSteps = np.size(vtTimeBase)

        # - Generate a noise trace
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.nSize)
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.fNoiseStd * np.sqrt(2.*self.vtTauN/self.tDt) * 1.63
        )
        
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
            return np.reshape(self._sgRecurrentSynapses.w, (self.nSize, -1))
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
            self._sgRecurrentSynapses.w = mfNewW.flatten()

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


# - RecIAFSpkInBrian - Class: Spiking recurrent layer with spiking in- and outputs
class RecIAFSpkInBrian(RecIAFBrian):
    """ RecIAFSpkInBrian - Class: Spiking recurrent layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(self,
                 mfWIn: np.ndarray,
                 mfWRec: np.ndarray,
                 vfBias: np.ndarray = 10*mA,

                 tDt: float = 0.1*ms,
                 fNoiseStd: float = 0*mV,

                 vtTauN: np.ndarray = 20*ms,
                 vtTauSInp: np.ndarray = 20*ms,
                 vtTauSRec: np.ndarray = 20*ms,

                 vfVThresh: np.ndarray = -55*mV,
                 vfVReset: np.ndarray = -65*mV,
                 vfVRest: np.ndarray = -65*mV,

                 tRefractoryTime = 0*ms,

                 eqNeurons = eqNeuronIAF2,
                 eqSynapses = eqSynapseExp2,

                 strIntegrator: str = 'rk4',

                 strName: str = 'unnamed',

                 bRecord: bool = False,
                 ):
        """
        RecIAFSpkInBrian - Construct a spiking recurrent layer with IAF neurons, with a Brian2 back-end
                           in- and outputs are spiking events

        :param mfWIn:           np.array MxN input weight matrix.
        :param mfWRec:          np.array NxN recurrent weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param tDt:             float Time-step. Default: 0.1 ms
        :param fNoiseStd:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param vtTauSInp:       np.array Nx1 vector of synapse time constants. Default: 20ms
        :param vtTauSRec:       np.array Nx1 vector of synapse time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param eqNeurons:       Brian2.Equations set of neuron equations. Default: IAF equation set
        :param eqSynapses:      Brian2.Equations set of synapse equations for recurrent connects. Default: exponential

        :param strIntegrator:   str Integrator to use for simulation. Default: 'rk4'

        :param strName:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions
        """

        # - Call Layer constructor
        Layer.__init__(
            self,
            mfW = mfWIn,
            tDt = np.asarray(tDt),
            fNoiseStd = np.asarray(fNoiseStd),
            strName = strName
        )

        # - Set up spike source to receive spiking input
        self._sggInput = b2.SpikeGeneratorGroup(self.nDimIn, [0], [0*second],
                                                dt = np.asarray(tDt) * second)
        # - Set up layer neurons
        self._ngLayer = b2.NeuronGroup(
            self.nSize,
            eqNeurons + eqSynapses,
            threshold = 'v > v_thresh',
            reset = 'v = v_reset',
            refractory = np.asarray(tRefractoryTime) * second,
            method = strIntegrator,
            dt = np.asarray(tDt) * second,
            name = 'spiking_ff_neurons'
        )
        self._ngLayer.v = vfVRest
        self._ngLayer.r_m = 1 * ohm

        # - Add source -> receiver synapses
        self._sgReceiver = b2.Synapses(
            self._sggInput, self._ngLayer,
            model = 'w : 1',
            on_pre = 'I_syn_post += w*amp',
            method = strIntegrator,
            dt = np.asarray(tDt) * second,
            name = 'receiver_synapses'
        )
        self._sgReceiver.connect()

        # - Add recurrent synapses
        self._sgRecurrentSynapses = b2.Synapses(
            self._ngLayer,
            self._ngLayer,
            model = 'w : 1',
            on_pre = 'I_syn_rec_post += w*amp',
            method = strIntegrator,
            dt = np.asarray(tDt) * second,
            name = 'recurrent_synapses',
        )
        self._sgRecurrentSynapses.connect()

        # - Add monitors to record layer outputs
        self._spmReservoir = b2.SpikeMonitor(self._ngLayer, record = True, name = 'layer_spikes')

        # - Call Network constructor
        self._net = b2.Network(
            self._sggInput,
            self._sgReceiver,
            self._sgRecurrentSynapses,
            self._ngLayer,
            self._spmReservoir,
            name = 'rec_spiking_layer'
        )

        if bRecord:
            # - Monitor for recording network potential
            self._stmVmem = b2.StateMonitor(self._ngLayer, ['v'], record=True, name = "layer_potential")
            self._net.add(self._stmVmem)

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vtTauSInp = vtTauSInp
        self.vtTauSRec = vtTauSRec
        self.tRefractoryTime = tRefractoryTime
        self.vfBias = vfBias
        self.mfWIn = mfWIn
        self.mfWRec = mfWRec

        # - Store "reset" state
        self._net.store('reset')

    def evolve(
        self,
        tsInput: TSContinuous = None,
        tDuration: float = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TimeSeries TxM or Tx1 input to this layer
        :param tDuration:   float Duration of evolution, in seconds
        :param bVerbose:    bool Currently no effect, just for conformity

        :return: TimeSeries Output of this layer during evolution period
        """

        # - Prepare time base
        vtTimeBase, _, tDuration = self._prepare_input(tsInput, tDuration)

        # - Generate a noise trace
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.nSize)
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.fNoiseStd * np.sqrt(2.*self.vtTauN/self.tDt) * 1.63
        )

        # - Specifiy network input currents, construct TimedArray
        taI_inp = TAShift(np.asarray(mfNoiseStep) * amp,
                          self.tDt * second, tOffset = self.t * second,
                          name  = 'external_input')

        # - Perform simulation
        self._net.run(tDuration * second, namespace = {'I_inp': taI_inp}, level = 0)

        # - Build response TimeSeries
        vbUseEvent = self._spmReservoir.t_ >= vtTimeBase[0]
        vtEventTimeOutput = self._spmReservoir.t[vbUseEvent]
        vnEventChannelOutput = self._spmReservoir.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, strName = 'Layer spikes')


        
        # - Set spikes for spike generator
        if tsInput is not None:
            vtEventTimes, vnEventChannels, _ = tsInput.find([vtTimeBase[0], vtTimeBase[-1]+self.tDt])
            self._sggInput.set_spikes(vnEventChannels, vtEventTimes * second, sorted = False)
        else:
            self._sggInput.set_spikes([], [] * second)

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(np.size(vtTimeBase), self.nSize) * self.fNoiseStd / np.sqrt(self.tDt)
        
        # - Specifiy noise input currents, construct TimedArray
        taI_noise = TAShift(np.asarray(mfNoiseStep) * amp,
                          self.tDt * second, tOffset = self.t * second,
                          name  = 'noise_input')

        # - Perform simulation
        self._net.run(tDuration * second, namespace = {'I_inp': taI_noise}, level = 0)

        # - Build response TimeSeries
        vbUseEvent = self._spmReservoir.t_ >= vtTimeBase[0]
        vtEventTimeOutput = self._spmReservoir.t_[vbUseEvent]
        vnEventChannelOutput = self._spmReservoir.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, strName = 'Layer spikes')

    def reset_time(self):

        # - Store state variables
        vfV = np.copy(self._ngLayer.v) * volt
        vfIsynRec = np.copy(self._ngLayer.I_syn_rec) * amp
        vfIsynInp = np.copy(self._ngLayer.I_syn_inp) * amp

        # - Store parameters
        vfVThresh = np.copy(self.vfVThresh)
        vfVReset = np.copy(self.vfVReset)
        vfVRest = np.copy(self.vfVRest)
        vtTauN = np.copy(self.vtTauN)
        vtTauSInp = np.copy(self.vtTauSInp)
        vtTauSRec = np.copy(self.vtTauSRec)
        tRefractoryTime = np.copy(self.tRefractoryTime)
        vfBias = np.copy(self.vfBias)
        mfWIn = np.copy(self.mfWIn)
        mfWRec = np.copy(self.mfWRec)

        self._net.restore('reset')
        
        # - Restork parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vtTauSInp = vtTauSInp
        self.vtTauSRec = vtTauSRec
        self.tRefractoryTime = tRefractoryTime
        self.vfBias = vfBias
        self.mfWIn = mfWIn
        self.mfWRec = mfWRec

        # - Restore state variables
        self._ngLayer.v = vfV
        self._ngLayer.I_syn_inp = vfIsynInp
        self._ngLayer.I_syn_rec = vfIsynRec

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._ngLayer.v = self.vfVRest * volt
        self._ngLayer.I_syn_inp = 0 * amp
        self._ngLayer.I_syn_rec = 0 * amp

    def reset_all(self, bKeepParams=True):
        if bKeepParams:
            # - Store parameters
            vfVThresh = np.copy(self.vfVThresh)
            vfVReset = np.copy(self.vfVReset)
            vfVRest = np.copy(self.vfVRest)
            vtTauN = np.copy(self.vtTauN)
            vtTauS = np.copy(self.vtTauS)
            vtTauSInp = np.copy(self.vtTauSInp)
            tRefractoryTime = np.copy(self.tRefractoryTime)
            vfBias = np.copy(self.vfBias)
            mfWIn = np.copy(self.mfWIn)
            mfWRec = np.copy(self.mfWRec)

        self.reset_state()
        self._net.restore('reset')
        
        if bKeepParams:
            # - Restork parameters
            self.vfVThresh = vfVThresh
            self.vfVReset = vfVReset
            self.vfVRest = vfVRest
            self.vtTauN = vtTauN
            self.vtTauSInp = vtTauSInp
            self.vtTauSRec = vtTauSRec
            self.tRefractoryTime = tRefractoryTime
            self.vfBias = vfBias
            self.mfWIn = mfWIn
            self.mfWRec = mfWRec
    
    def randomize_state(self):
        """ .randomize_state() - Method: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        fRangeV = abs(self.vfVThresh - self.vfVReset)
        self._ngLayer.v = (np.random.rand(self.nSize) * fRangeV + self.vfVReset) * volt
        self._ngLayer.I_syn_inp = np.random.randn(self.nSize) * np.mean(np.abs(self.mfWIn)) * amp
        self._ngLayer.I_syn_rec = np.random.randn(self.nSize) * np.mean(np.abs(self.mfWRec)) * amp

    @property
    def cInput(self):
        return TSEvent

    @property
    def mfW(self):
        return self.mfWRec

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWRec = mfNewW

    @property
    def mfWIn(self):
        return np.array(self._sgReceiver.w).reshape(self.nDimIn, self.nSize)

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        assert (
            mfNewW.shape == (self.nDimIn, self.nSize)
            or mfNewW.shape == self._sgReceiver.w.shape
        ), "mfW must be of dimensions ({}, {}) or flat with size {}.".format(
            self.nDimIn, self.nSize, self.nDimIn*self.nSize
        )
        
        self._sgReceiver.w = np.array(mfNewW).flatten()

    @property
    def mfWRec(self):
        return np.array(self._sgRecurrentSynapses.w).reshape(self.nDimIn, self.nSize)

    @mfWRec.setter
    def mfWRec(self, mfNewW):
        assert (
            mfNewW.shape == (self.nSize, self.nSize)
            or mfNewW.shape == self._sgReceiver.w.shape
        ), "mfW must be of dimensions ({}, {}) or flat with size {}.".format(
            self.nSize, self.nSize, self.nsize*self.nSize
        )
        
        self._sgRecurrentSynapses.w = np.array(mfNewW).flatten()

    @property
    def vtTauSInp(self):
        return self._ngLayer.tau_syn_inp

    @vtTauSInp.setter
    def vtTauSInp(self, vtNewTauS):
        self._ngLayer.tau_syn_inp = np.asarray(self._expand_to_net_size(vtNewTauS, 'vtTauSInp')) * second

    @property
    def vtTauSRec(self):
        return self._ngLayer.tau_syn_rec

    @vtTauSRec.setter
    def vtTauSRec(self, vtNewTauS):
        self._ngLayer.tau_syn_rec = np.asarray(self._expand_to_net_size(vtNewTauS, 'vtTauSRec')) * second

