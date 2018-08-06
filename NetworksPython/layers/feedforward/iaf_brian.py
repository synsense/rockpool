###
# iaf_brian.py - Class implementing an IAF simple feed-forward layer in Brian
###


# - Imports
import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *

from ...timeseries import TSContinuous, TSEvent

from ..layer import Layer
from ..recurrent.timedarray_shift import TimedArray as TAShift

# - Configure exports
__all__ = ['FFIAFBrian', 'eqNeuronIAF']

# - Equations for an integrate-and-fire neuron
eqNeuronIAF = b2.Equations('''
    dv/dt = (v_rest - v + r_m * I_total) / tau_m    : volt (unless refractory)  # Neuron membrane voltage
    I_total = I_inp(t, i) + I_bias                  : amp                       # Total input current
    I_bias                                          : amp                       # Per-neuron bias current
    v_rest                                          : volt                      # Rest potential
    tau_m                                           : second                    # Membrane time constant
    r_m                                             : ohm                       # Membrane resistance
    v_thresh                                        : volt                      # Firing threshold potential
    v_reset                                         : volt                      # Reset potential
''')

eqNeuronIAFSpkIn = b2.Equations('''
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
''')


## - FFIAFBrian - Class: define a spiking feedforward layer with spiking outputs
class FFIAFBrian(Layer):
    """ FFIAFBrian - Class: define a spiking feedforward layer with spiking outputs
    """

    ## - Constructor
    def __init__(self,
                 mfW: np.ndarray,
                 vfBias: np.ndarray = 10*mA,

                 tDt: float = 0.1*ms,
                 fNoiseStd: float = 0*mV,

                 vtTauN: np.ndarray = 20*ms,

                 vfVThresh: np.ndarray = -55*mV,
                 vfVReset: np.ndarray = -65*mV,
                 vfVRest: np.ndarray = -65*mV,

                 tRefractoryTime = 0*ms,

                 eqNeurons = eqNeuronIAF,

                 strIntegrator: str = 'rk4',

                 strName: str = 'unnamed'
                 ):
        """
        FFIAFBrian - Construct a spiking feedforward layer with IAF neurons, with a Brian2 back-end
                     Inputs are continuous currents; outputs are spiking events

        :param mfW:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param tDt:             float Time-step. Default: 0.1 ms
        :param fNoiseStd:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param eqNeurons:       Brian2.Equations set of neuron equations. Default: IAF equation set

        :param strIntegrator:   str Integrator to use for simulation. Default: 'rk4'

        :param strName:         str Name for the layer. Default: 'unnamed'
        """

        # - Call super constructor
        super().__init__(mfW = mfW,
                         tDt = np.asarray(tDt),
                         fNoiseStd = np.asarray(fNoiseStd),
                         strName = strName)

        # - Set up layer neurons
        self._ngLayer = b2.NeuronGroup(self.nSize, eqNeurons,
                                       threshold = 'v > v_thresh',
                                       reset = 'v = v_reset',
                                       refractory = np.asarray(tRefractoryTime) * second,
                                       method = strIntegrator,
                                       dt = np.asarray(tDt) * second,
                                       name = 'spiking_ff_neurons')
        self._ngLayer.v = vfVRest
        self._ngLayer.r_m = 1 * ohm

        # - Add monitors to record layer outputs
        self._spmLayer = b2.SpikeMonitor(self._ngLayer, record = True, name = 'layer_spikes')

        # - Call Network constructor
        self._net = b2.Network(self._ngLayer, self._spmLayer,
                               name = 'ff_spiking_layer')

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.tRefractoryTime = tRefractoryTime
        self.vfBias = vfBias
        self.mfW = mfW

        # - Store "reset" state
        self._net.store('reset')


    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._ngLayer.v = self.vfVRest * volt

    def randomize_state(self):
        """ .randomize_state() - Method: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        fRangeV = abs(self.vfVThresh - self.vfVReset)
        self._ngLayer.v = (np.random.rand(self.nSize) * fRangeV + self.vfVReset) * volt

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

        # - Weight inputs
        mfNeuronInputStep = mfInputStep @ self.mfW

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(np.size(vtTimeBase), self.nSize) * self.fNoiseStd

        # - Specifiy network input currents, construct TimedArray
        taI_inp = TAShift(np.asarray(mfNeuronInputStep + mfNoiseStep) * amp,
                          self.tDt * second, tOffset = self.t * second,
                          name  = 'external_input')

        # - Perform simulation
        self._net.run(tDuration * second, namespace = {'I_inp': taI_inp}, level = 0)

        # - Build response TimeSeries
        vtEventTimeOutput = self._spmLayer.t_
        vbUseEvent = self._spmLayer.t_ >= vtTimeBase[0]
        vnEventChannelOutput = self._spmLayer.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, strName = 'Layer spikes')


    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

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


# - FFIAFSpkInBrian - Class: Spiking feedforward layer with spiking in- and outputs
class FFIAFSpkInBrian(FFIAFBrian):
    """ FFIAFSpkInBrian - Class: Spiking feedforward layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(self,
                 mfW: np.ndarray,
                 vfBias: np.ndarray = 10*mA,

                 tDt: float = 0.1*ms,
                 fNoiseStd: float = 0*mV,

                 vtTauN: np.ndarray = 20*ms,
                 vtTauS: np.ndarray = 20*ms,

                 vfVThresh: np.ndarray = -55*mV,
                 vfVReset: np.ndarray = -65*mV,
                 vfVRest: np.ndarray = -65*mV,

                 tRefractoryTime = 0*ms,

                 eqNeurons = eqNeuronIAFSpkIn,

                 strIntegrator: str = 'rk4',

                 strName: str = 'unnamed'
                 ):
        """
        FFIAFSpkInBrian - Construct a spiking feedforward layer with IAF neurons, with a Brian2 back-end
                          in- and outputs are spiking events

        :param mfW:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param tDt:             float Time-step. Default: 0.1 ms
        :param fNoiseStd:       float Noise std. dev. per second. Default: 0

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param vtTauS:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron thresholds. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param eqNeurons:       Brian2.Equations set of neuron equations. Default: IAF equation set

        :param strIntegrator:   str Integrator to use for simulation. Default: 'rk4'

        :param strName:         str Name for the layer. Default: 'unnamed'
        """

        # - Call Layer constructor
        Layer.__init__(
            self,
            mfW = mfW,
            tDt = np.asarray(tDt),
            fNoiseStd = np.asarray(fNoiseStd),
            strName = strName
        )

        # - Set up spike source to receive spiking input
        self._sggInput = b2.SpikeGeneratorGroup(self.nDimIn, [0], [0*second],
                                                dt = np.asarray(tDt) * second)
        # - Set up layer neurons
        self._ngLayer = b2.NeuronGroup(self.nSize, eqNeurons,
                                       threshold = 'v > v_thresh',
                                       reset = 'v = v_reset',
                                       refractory = np.asarray(tRefractoryTime) * second,
                                       method = strIntegrator,
                                       dt = np.asarray(tDt) * second,
                                       name = 'spiking_ff_neurons')
        self._ngLayer.v = vfVRest
        self._ngLayer.r_m = 1 * ohm

        # - Add source -> receiver synapses
        self._sgReceiver = b2.Synapses(self._sggInput, self._ngLayer,
                                       model = 'w : 1',
                                       on_pre = 'I_syn_post += w*amp',
                                       method = strIntegrator,
                                       dt = np.asarray(tDt) * second,
                                       name = 'receiver_synapses')
        self._sgReceiver.connect()

        # - Add monitors to record layer outputs
        self._spmLayer = b2.SpikeMonitor(self._ngLayer, record = True, name = 'layer_spikes')

        # - Call Network constructor
        self._net = b2.Network(
            self._sggInput,
            self._sgReceiver,
            self._ngLayer,
            self._spmLayer,
            name = 'ff_spiking_layer'
        )

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vtTauS = vtTauS
        self.tRefractoryTime = tRefractoryTime
        self.vfBias = vfBias
        self.mfW = mfW

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

    def evolve(
        self,
        tsInput: TSContinuous = None,
        tDuration: float = None
    ):
        """
        evolve - Evolve the state of this layer

        :param tsInput:     TimeSeries TxM or Tx1 input to this layer
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
        
        # - Specifiy noise input currents, construct TimedArray
        taI_noise = TAShift(np.asarray(mfNoiseStep) * amp,
                          self.tDt * second, tOffset = self.t * second,
                          name  = 'noise_input')

        # - Perform simulation
        self._net.run(tDuration * second, namespace = {'I_inp': taI_noise}, level = 0)

        # - Build response TimeSeries
        vtEventTimeOutput = self._spmLayer.t_
        vbUseEvent = self._spmLayer.t_ >= vtTimeBase[0]
        vnEventChannelOutput = self._spmLayer.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, strName = 'Layer spikes')

    def reset_all(self, bResetParams=False):
        if not bResetParams:
            # - Store parameters
            vfVThresh = self.vfVThresh
            vfVReset = self.vfVReset
            vfVRest = self.vfVRest
            vtTauN = self.vtTauN
            vtTauS = self.vtTauS
            tRefractoryTime = self.tRefractoryTime
            vfBias = self.vfBias
            mfW = self.mfW

        super().reset_all()
        
        if not bResetParams:
            # - Restork parameters
            self.vfVThresh = vfVThresh
            self.vfVReset = vfVReset
            self.vfVRest = vfVRest
            self.vtTauN = vtTauN
            self.vtTauS = vtTauS
            self.tRefractoryTime = tRefractoryTime
            self.vfBias = vfBias
            self.mfW = mfW


    @property
    def cInput(self):
        return TSEvent

    @property
    def mfW(self):
        return self._sgReceiver.w

    @mfW.setter
    def mfW(self, mfNewW):
        assert (
            mfNewW.shape == (self.nDimIn, self.nSize)
            or mfNewW.shape == self._sgReceiver.w.shape
        ), "mfW must be of dimensions ({}, {}).".format(self.nDimIn, self.nSize)
        
        self._sgReceiver.w = np.array(mfNewW).flatten()

    @property
    def vtTauS(self):
        return self._ngLayer.tau_s_

    @vtTauS.setter
    def vtTauS(self, vtNewTauS):
        self._ngLayer.tau_s = np.asarray(self._expand_to_net_size(vtNewTauS, 'vtNewTauS')) * second

