###
# iaf_brian.py - Class implementing an IAF simple feed-forward layer in Brian
###


# - Imports
import brian2 as b2
import brian2.numpy_ as np
from brian2.units.stdunits import *
from brian2.units.allunits import *
from typing import List, Tuple, Union

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


## - FFIAFBrian - Class: define a spiking feedforward layer with spiking outputs
class FFIAFBrian(Layer):
    """ FFIAFBrian - Class: define a spiking feedforward layer with spiking outputs
    """

    ## - Constructor
    def __init__(self,
                 mfW: np.ndarray,
                 vfBias: Union[float, np.ndarray] = 15*mA,

                 tDt: float = 0.1*ms,
                 fNoiseStd: float = 0*mV,

                 vtTauN: Union[float, np.ndarray] = 20*ms,

                 vfVThresh: Union[float, np.ndarray] = -55*mV,
                 vfVReset: Union[float, np.ndarray] = -65*mV,
                 vfVRest: Union[float, np.ndarray] = -65*mV,

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
                                       refractory = tRefractoryTime,
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
                          name = 'external_input')

        # - Perform simulation
        self._net.run(tDuration * second, namespace = {'I_inp': taI_inp}, level = 0)

        # - Build response TimeSeries
        vtEventTimeOutput = self._spmLayer.t_
        vbUseEvent = self._spmLayer.t_ >= vtTimeBase[0]
        vnEventChannelOutput = self._spmLayer.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, strName = 'Layer spikes')

    def stream(self,
               tDuration: float,
               tDt: float,
               bVerbose: bool = False,
               ) -> Tuple[float, List[float]]:
        """
        stream - Stream data through this layer
        :param tDuration:   float Total duration for which to handle streaming
        :param tDt:         float Streaming time step
        :param bVerbose:    bool Display feedback

        :yield: (vtEventTimes, vnEventChannels)

        :return: Final (vtEventTimes, vnEventChannels)
        """

        # - Initialise simulation, determine how many tDt to evolve for
        if bVerbose: print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, tDuration+tDt, tDt)
        nNumSteps = np.size(vtTimeTrace)-1

        # - Generate a noise trace
        mfNoiseStep = np.random.randn(nNumSteps, self.nSize) * self.fNoiseStd * np.sqrt(self.tDt)

        # - Generate a TimedArray to use for step-constant input currents
        taI_inp = TAShift(np.zeros((1, self._nDimIn)) * amp,
                          self.tDt * second,
                          name = 'external_input',
                          )

        if bVerbose: print("Layer: Prepared")

        # - Loop over tDt steps
        for nStep in range(nNumSteps):
            if bVerbose: print('Layer: Yielding from internal state.')
            if bVerbose: print('Layer: step', nStep)
            if bVerbose: print('Layer: Waiting for input...')

            # - Yield current output spikes, receive input for next time step
            vbUseEvents = self._spmLayer.t_ >= vtTimeTrace[nStep]
            if bVerbose: print('Layer: Yielding {} spikes'.format(np.sum(vbUseEvents)))
            tupInput = yield self._spmLayer.t_[vbUseEvents], self._spmLayer.i_[vbUseEvents]

            # - Specify network input currents for this streaming step
            if tupInput is None:
                taI_inp.values = mfNoiseStep[nStep, :]
            else:
                taI_inp.values = np.reshape(tupInput[1][0, :], (1, -1)) + mfNoiseStep[nStep, :]

            # - Reinitialise TimedArray
            taI_inp._init_2d()

            if bVerbose: print('Layer: Input was: ', tupInput)

            # - Evolve layer (increments time implicitly)
            self._net.run(tDt * second, namespace = {'I_inp': taI_inp}, level = 0)

        # - Return final spikes, if any
        vbUseEvents = self._spmLayer.t_ >= vtTimeTrace[-2]  # Should be tDuration - tDt
        return self._spmLayer.t_[vbUseEvents], self._spmLayer.i_[vbUseEvents]


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
