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
b2.set_device('genn')

from ...timeseries import TSContinuous, TSEvent

from ..layer import Layer
from .timedarray_shift import TimedArray as TAShift

from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Configure exports
__all__ = [
    "FFIAFBrian",
    "FFIAFSpkInBrian",
    "eqNeuronIAFSpkInFF",
    "eqNeuronIAFSpkInRec",
    "eqSynapseExp",
    "eqSynapseExpSpkInRec"
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
        mfW: np.ndarray,
        vfBias: Union[float, np.ndarray] = 15 * mA,
        tDt: float = 0.1 * ms,
        fNoiseStd: float = 0 * mV,
        vtTauN: Union[float, np.ndarray] = 20 * ms,
        vfVThresh: Union[float, np.ndarray] = -55 * mV,
        vfVReset: Union[float, np.ndarray] = -65 * mV,
        vfVRest: Union[float, np.ndarray] = -65 * mV,
        tRefractoryTime=0 * ms,
        eqNeurons=eqNeuronIAFFF,
        strIntegrator: str = "rk4",
        strName: str = "unnamed",
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

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            mfW=np.asarray(mfW),
            tDt=np.asarray(tDt),
            fNoiseStd=np.asarray(fNoiseStd),
            strName=strName,
        )

        # - Set up layer neurons
        self._ngLayer = b2.NeuronGroup(
            self.nSize,
            eqNeurons,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=np.asarray(tRefractoryTime) * second,
            method=strIntegrator,
            dt=np.asarray(tDt) * second,
            name="spiking_ff_neurons",
        )
        self._ngLayer.v = vfVRest
        self._ngLayer.r_m = 1 * ohm

        # - Add monitors to record layer outputs
        self._spmLayer = b2.SpikeMonitor(
            self._ngLayer, record=True, name="layer_spikes"
        )

        # - Call Network constructor
        self._net = b2.Network(self._ngLayer, self._spmLayer, name="ff_spiking_layer")

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vfBias = vfBias
        self.mfW = mfW

        # - Store "reset" state
        self._net.store("reset")

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

        # - Sotre state variables
        vfV = np.copy(self._ngLayer.v) * volt

        # - Store parameters
        vfVThresh = np.copy(self.vfVThresh)
        vfVReset = np.copy(self.vfVReset)
        vfVRest = np.copy(self.vfVRest)
        vtTauN = np.copy(self.vtTauN)
        vfBias = np.copy(self.vfBias)
        mfW = np.copy(self.mfW)

        # - Reset network
        self._net.restore("reset")
        self._nTimeStep = 0

        # - Restork parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vfBias = vfBias
        self.mfW = mfW

        # - Restore state variables
        self._ngLayer.v = vfV

    ### --- State evolution

    def evolve(
        self,
        tsInput: Optional[TSContinuous] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        vtTimeBase, mfInputStep, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        # - Weight inputs
        mfNeuronInputStep = mfInputStep @ self.mfW

        # - Generate a noise trace
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.nSize)
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.fNoiseStd
            * np.sqrt(2. * self.vtTauN / self.tDt)
            * 1.63
        )

        # - Specifiy network input currents, construct TimedArray
        taI_inp = TAShift(
            np.asarray(mfNeuronInputStep + mfNoiseStep) * amp,
            self.tDt * second,
            tOffset=self.t * second,
            name="external_input",
        )

        # - Perform simulation
        self._net.run(
            nNumTimeSteps * self.tDt * second, namespace={"I_inp": taI_inp}, level=0
        )
        self._nTimeStep += nNumTimeSteps

        # - Build response TimeSeries
        vbUseEvent = self._spmLayer.t_ >= vtTimeBase[0]
        vtEventTimeOutput = self._spmLayer.t_[vbUseEvent]
        vnEventChannelOutput = self._spmLayer.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, strName="Layer spikes")

    def stream(
        self, tDuration: float, tDt: float, bVerbose: bool = False
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
        if bVerbose:
            print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, tDuration + tDt, tDt)
        nNumSteps = np.size(vtTimeTrace) - 1

        # - Generate a noise trace
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeTrace), self.nSize)
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.fNoiseStd
            * np.sqrt(2. * self.vtTauN / self.tDt)
            * 1.63
        )

        # - Generate a TimedArray to use for step-constant input currents
        taI_inp = TAShift(
            np.zeros((1, self._nSizeIn)) * amp, self.tDt * second, name="external_input"
        )

        if bVerbose:
            print("Layer: Prepared")

        # - Loop over tDt steps
        for nStep in range(nNumSteps):
            if bVerbose:
                print("Layer: Yielding from internal state.")
            if bVerbose:
                print("Layer: step", nStep)
            if bVerbose:
                print("Layer: Waiting for input...")

            # - Yield current output spikes, receive input for next time step
            vbUseEvents = self._spmLayer.t_ >= vtTimeTrace[nStep]
            if bVerbose:
                print("Layer: Yielding {} spikes".format(np.sum(vbUseEvents)))
            tupInput = (
                yield self._spmLayer.t_[vbUseEvents],
                self._spmLayer.i_[vbUseEvents],
            )

            # - Specify network input currents for this streaming step
            if tupInput is None:
                taI_inp.values = mfNoiseStep[nStep, :]
            else:
                taI_inp.values = (
                    np.reshape(tupInput[1][0, :], (1, -1)) + mfNoiseStep[nStep, :]
                )

            # - Reinitialise TimedArray
            taI_inp._init_2d()

            if bVerbose:
                print("Layer: Input was: ", tupInput)

            # - Evolve layer (increments time implicitly)
            self._net.run(tDt * second, namespace={"I_inp": taI_inp}, level=0)

        # - Return final spikes, if any
        vbUseEvents = self._spmLayer.t_ >= vtTimeTrace[-2]  # Should be tDuration - tDt
        return self._spmLayer.t_[vbUseEvents], self._spmLayer.i_[vbUseEvents]

    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

    @property
    def tRefractoryTime(self):
        return self._ngLayer._refractory

    @property
    def vState(self):
        return self._ngLayer.v_

    @vState.setter
    def vState(self, vNewState):
        self._ngLayer.v = (
            np.asarray(self._expand_to_net_size(vNewState, "vNewState")) * volt
        )

    @property
    def vtTauN(self):
        return self._ngLayer.tau_m_

    @vtTauN.setter
    def vtTauN(self, vtNewTauN):
        self._ngLayer.tau_m = (
            np.asarray(self._expand_to_net_size(vtNewTauN, "vtNewTauN")) * second
        )

    @property
    def vfBias(self):
        return self._ngLayer.I_bias_

    @vfBias.setter
    def vfBias(self, vfNewBias):
        self._ngLayer.I_bias = (
            np.asarray(self._expand_to_net_size(vfNewBias, "vfNewBias")) * amp
        )

    @property
    def vfVThresh(self):
        return self._ngLayer.v_thresh_

    @vfVThresh.setter
    def vfVThresh(self, vfNewVThresh):
        self._ngLayer.v_thresh = (
            np.asarray(self._expand_to_net_size(vfNewVThresh, "vfNewVThresh")) * volt
        )

    @property
    def vfVRest(self):
        return self._ngLayer.v_rest_

    @vfVRest.setter
    def vfVRest(self, vfNewVRest):
        self._ngLayer.v_rest = (
            np.asarray(self._expand_to_net_size(vfNewVRest, "vfNewVRest")) * volt
        )

    @property
    def vfVReset(self):
        return self._ngLayer.v_reset_

    @vfVReset.setter
    def vfVReset(self, vfNewVReset):
        self._ngLayer.v_reset = (
            np.asarray(self._expand_to_net_size(vfNewVReset, "vfNewVReset")) * volt
        )

    @property
    def t(self):
        return self._net.t_

    @Layer.tDt.setter
    def tDt(self, _):
        raise ValueError("The `tDt` property cannot be set for this layer")


# - FFIAFSpkInBrian - Class: Spiking feedforward layer with spiking in- and outputs
class FFIAFSpkInBrian(FFIAFBrian):
    """ FFIAFSpkInBrian - Class: Spiking feedforward layer with spiking in- and outputs
    """

    ## - Constructor
    def __init__(
        self,
        mfW: np.ndarray,
        vfBias: np.ndarray = 10 * mA,
        tDt: float = 0.1 * ms,
        fNoiseStd: float = 0 * mV,
        vtTauN: np.ndarray = 20 * ms,
        vtTauS: np.ndarray = 20 * ms,
        vfVThresh: np.ndarray = -55 * mV,
        vfVReset: np.ndarray = -65 * mV,
        vfVRest: np.ndarray = -65 * mV,
        tRefractoryTime=0 * ms,
        eqNeurons=eqNeuronIAFSpkInFF,
        strIntegrator: str = "rk4",
        strName: str = "unnamed",
        bRecord: bool = False,
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

        :param bRecord:         bool Record membrane potential during evolutions
        """

        b2.defaultclock.dt = tDt

        # - Call Layer constructor
        Layer.__init__(
            self,
            mfW=mfW,
            tDt=np.asarray(tDt),
            fNoiseStd=np.asarray(fNoiseStd),
            strName=strName,
        )

        # - Set up spike source to receive spiking input
        self._sggInput = b2.SpikeGeneratorGroup(
            self.nSizeIn, [0], [0 * second]
        )
        # - Set up layer neurons
        self._ngLayer = b2.NeuronGroup(
            self.nSize,
            eqNeurons,
            threshold="v > v_thresh",
            reset="v = v_reset",
            refractory=np.asarray(tRefractoryTime) * second,
            method=strIntegrator,
            name="spiking_ff_neurons",
        )
        self._ngLayer.v = vfVRest
        self._ngLayer.r_m = 1 * ohm

        # - Add source -> receiver synapses
        self._sgReceiver = b2.Synapses(
            self._sggInput,
            self._ngLayer,
            model="w : 1",
            on_pre="I_syn_post += w*amp",
            method=strIntegrator,
            name="receiver_synapses",
        )
        self._sgReceiver.connect()

        # - Add monitors to record layer outputs
        self._spmLayer = b2.SpikeMonitor(
            self._ngLayer, record=True, name="layer_spikes"
        )

        # - Call Network constructor
        self._net = b2.Network(
            self._sggInput,
            self._sgReceiver,
            self._ngLayer,
            self._spmLayer,
            name="ff_spiking_layer",
        )

        if bRecord:
            # - Monitor for recording network potential
            self._stmVmem = b2.StateMonitor(
                self._ngLayer, ["v"], record=True, name="layer_potential"
            )
            self._net.add(self._stmVmem)

        # - Record neuron parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vtTauS = vtTauS
        self.vfBias = vfBias
        self.mfW = mfW

        # - Store "reset" state
        # self._net.store("reset")

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """

        # - Prepare time base
        vtTimeBase, __, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        # - Set spikes for spike generator
        if tsInput is not None:
            vtEventTimes, vnEventChannels, _ = tsInput.find(
                [vtTimeBase[0], vtTimeBase[-1] + self.tDt]
            )
            self._sggInput.set_spikes(
                vnEventChannels, vtEventTimes * second, sorted=False
            )
        else:
            self._sggInput.set_spikes([], [] * second)

        # - Generate a noise trace
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.nSize)
            # - Standard deviation slightly smaller than expected (due to brian??),
            #   therefore correct with empirically found factor 1.63
            * self.fNoiseStd
            * np.sqrt(2. * self.vtTauN / self.tDt)
            * 1.63
        )

        # - Specifiy noise input currents, construct TimedArray
        # taI_noise = TAShift(
        #     np.asarray(mfNoiseStep) * amp,
        #     self.tDt * second,
        #     tOffset=self.t * second,
        #     name="noise_input",
        # )

        # - Perform simulation
        self._net.run(
            nNumTimeSteps * self.tDt * second, level=0
        )
        self._nTimeStep += nNumTimeSteps

        # - Build response TimeSeries
        vbUseEvent = self._spmLayer.t_ >= vtTimeBase[0]
        vtEventTimeOutput = self._spmLayer.t_[vbUseEvent]
        vnEventChannelOutput = self._spmLayer.i[vbUseEvent]

        return TSEvent(vtEventTimeOutput, vnEventChannelOutput, strName="Layer spikes")

    def reset_time(self):

        # - Store state variables
        vfV = np.copy(self._ngLayer.v) * volt
        vfIsyn = np.copy(self._ngLayer.I_syn) * amp

        # - Store parameters
        vfVThresh = np.copy(self.vfVThresh)
        vfVReset = np.copy(self.vfVReset)
        vfVRest = np.copy(self.vfVRest)
        vtTauN = np.copy(self.vtTauN)
        vtTauS = np.copy(self.vtTauS)
        vfBias = np.copy(self.vfBias)
        mfW = np.copy(self.mfW)

        self._net.restore("reset")
        self._nTimeStep = 0

        # - Restork parameters
        self.vfVThresh = vfVThresh
        self.vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vtTauS = vtTauS
        self.vfBias = vfBias
        self.mfW = mfW

        # - Restore state variables
        self._ngLayer.v = vfV
        self._ngLayer.I_syn = vfIsyn

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """
        self._ngLayer.v = self.vfVRest * volt
        self._ngLayer.I_syn = 0 * amp

    def reset_all(self, bKeepParams=True):
        if bKeepParams:
            # - Store parameters
            vfVThresh = np.copy(self.vfVThresh)
            vfVReset = np.copy(self.vfVReset)
            vfVRest = np.copy(self.vfVRest)
            vtTauN = np.copy(self.vtTauN)
            vtTauS = np.copy(self.vtTauS)
            vfBias = np.copy(self.vfBias)
            mfW = np.copy(self.mfW)

        self.reset_state()
        self._net.restore("reset")
        self._nTimeStep = 0

        if bKeepParams:
            # - Restork parameters
            self.vfVThresh = vfVThresh
            self.vfVReset = vfVReset
            self.vfVRest = vfVRest
            self.vtTauN = vtTauN
            self.vtTauS = vtTauS
            self.vfBias = vfBias
            self.mfW = mfW

    def randomize_state(self):
        """ .randomize_state() - Method: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        fRangeV = abs(self.vfVThresh - self.vfVReset)
        self._ngLayer.v = (np.random.rand(self.nSize) * fRangeV + self.vfVReset) * volt
        self._ngLayer.I_syn = np.random.rand(self.nSize) * amp

    def pot_kernel(self, t):
        """ pot_kernel - response of the membrane potential to an
                         incoming spike at a single synapse with
                         weight 1*amp (not considering vfVRest)
        """
        t = t.reshape(-1, 1)
        fConst = self.vtTauS / (self.vtTauS - self.vtTauN) * self._ngLayer.r_m * amp
        return fConst * (np.exp(-t / self.vtTauS) - np.exp(-t / self.vtTauN))

    def train_mst_simple(
        self,
        tDuration: float,
        tStart: float,
        tsInput: TSEvent,
        vnTargetCounts: np.ndarray = None,
        fLambda: float = 1e-5,
        fEligibilityRatio: float = 0.1,
        fMomentum: float = 0,
        bFirst: bool = True,
        bFinal: bool = False,
        bVerbose: bool = False,
    ):
        """
        train_mst_simple - Use the multi-spike tempotron learning rule
                           from Guetig2017, in its simplified version,
                           where no gradients are calculated
        """

        assert hasattr(self, "_stmVmem"), (
            "Layer needs to be instantiated with bRecord=True for "
            + "this learning rule."
        )

        # - End time of current batch
        tStop = tStart + tDuration

        if tsInput is not None:
            vtEventTimes, vnEventChannels, _ = tsInput.find([tStart, tStop])
        else:
            print("No tsInput defined, assuming input to be 0.")
            vtEventTimes, vnEventChannels = [], []

        # - Prepare target
        if vnTargetCounts is None:
            vnTargetCounts = np.zeros(self.nSize)
        else:
            assert (
                np.size(vnTargetCounts) == self.nSize
            ), "Target array size must match layer size ({}).".format(
                self.nSize
            )

        ## -- Determine eligibility for each neuron and synapse
        mfEligibiity = np.zeros((self.nSizeIn, self.nSize))

        # - Iterate over source neurons
        for iSource in range(self.nSizeIn):
            if bVerbose:
                print(
                    "\rProcessing input {} of {}".format(iSource + 1, self.nSizeIn),
                    end="",
                )
            # - Find spike timings
            vtEventTimesSource = vtEventTimes[vnEventChannels == iSource]
            # - Sum individual correlations over input spikes, for all synapses
            for tSpkIn in vtEventTimesSource:
                # - Membrane potential between input spike time and now (transform to vfVRest at 0)
                vfVmem = (
                    self._stmVmem.v.T[self._stmVmem.t_ >= tSpkIn] - self.vfVRest * volt
                )
                # - Kernel between input spike time and now
                vfKernel = self.pot_kernel(
                    self._stmVmem.t_[self._stmVmem.t_ >= tSpkIn] - tSpkIn
                )
                # - Add correlations to eligibility matrix
                mfEligibiity[iSource, :] += np.sum(vfKernel * vfVmem)

        ## -- For each neuron sort eligibilities and choose synapses with largest eligibility
        nEligible = int(fEligibilityRatio * self.nSizeIn)
        # - Mark eligible neurons
        miEligible = np.argsort(mfEligibiity, axis=0)[:nEligible:-1]

        ##  -- Compare target number of events with spikes and perform weight updates for chosen synapses
        # - Numbers of (output) spike times for each neuron
        vbUseEventOut = (self._spmLayer.t_ >= tStart) & (self._spmLayer.t_ <= tStop)
        viSpkNeuronOut = self._spmLayer.i[vbUseEventOut]
        vnSpikeCount = np.array(
            [np.sum(viSpkNeuronOut == iNeuron) for iNeuron in range(self.nSize)]
        )

        # - Updates to eligible synapses of each neuron
        vfUpdates = np.zeros(self.nSize)
        # - Negative update if spike count too high
        vfUpdates[vnSpikeCount > vnTargetCounts] = -fLambda
        # - Positive update if spike count too low
        vfUpdates[vnSpikeCount < vnTargetCounts] = fLambda

        # - Reset previous weight changes that are used for momentum heuristic
        if bFirst:
            self._mfDW_previous = np.zeros_like(self.mfW)

        # - Accumulate updates to me made to weights
        mfDW_current = np.zeros_like(self.mfW)

        # - Update only eligible synapses
        for iTarget in range(self.nSize):
            mfDW_current[miEligible[:, iTarget], iTarget] += vfUpdates[iTarget]

        # - Include previous weight changes for momentum heuristic
        mfDW_current += fMomentum * self._mfDW_previous

        # - Perform weight update
        self.mfW += mfDW_current
        # - Store weight changes for next iteration
        self._mfDW_previous = mfDW_current

    @property
    def cInput(self):
        return TSEvent

    @property
    def tRefractoryTime(self):
        return self._ngLayer._refractory

    @property
    def mfW(self):
        return np.array(self._sgReceiver.w).reshape(self.nSizeIn, self.nSize)

    @mfW.setter
    def mfW(self, mfNewW):
        assert (
            mfNewW.shape == (self.nSizeIn, self.nSize)
            or mfNewW.shape == self._sgReceiver.w.shape
        ), "mfW must be of dimensions ({}, {}) or flat with size {}.".format(
            self.nSizeIn, self.nSize, self.nSizeIn * self.nSize
        )

        self._sgReceiver.w = np.array(mfNewW).flatten()

    @property
    def vtTauS(self):
        return self._ngLayer.tau_s_

    @vtTauS.setter
    def vtTauS(self, vtNewTauS):
        self._ngLayer.tau_s = (
            np.asarray(self._expand_to_net_size(vtNewTauS, "vtNewTauS")) * second
        )