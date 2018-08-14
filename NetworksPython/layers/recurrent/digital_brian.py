###
# digital_brian.py - Class implementing a recurrent layer consisting of
#                    digital neurons with constant leak in Brian
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


# - Configure exports
__all__ = ['RecDIAFBrian', 'eqNeuronDIAF', 'eqSynapseExp']


# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9


# - Equations for an integrate-and-fire neuron
eqNeuronDIAF = b2.Equations('''
    dv/dt = - c_leak                                : volt (unless refractory)  # Neuron membrane voltage
    v_thresh                                        : volt                      # Firing threshold potential
    v_reset                                         : volt                      # Reset potential
''')

## - RecDIAFBrian - Class: define a spiking recurrent layer based on digital IAF neurons
class RecDIAF(Layer):
    """ RecDIAFBrian - Class: define a spiking recurrent layer based on digital IAF neurons
    """

    ## - Constructor
    def __init__(self,

                 mfW: np.ndarray = None,
                 vfWIn: np.ndarray = None,

                 tDt: float = 0.1*ms,
                 fNoiseStd: float = 1*mV,

                 tRefractoryTime = 0*ms,

                 vfVThresh: np.ndarray = -55*mV,
                 vfVReset: np.ndarray = -65*mV,
                 vfCleak: np.ndarray = 1 * volt / second,

                 strDtypeState: str = "int8",

                 strIntegrator: str = 'rk4',

                 strName: str = 'unnamed'
                 ):
        """
        RecDIAFBrian - Construct a spiking recurrent layer with digital IAF neurons, with a Brian2 back-end

        :param mfW:             np.array NxN weight matrix
        :param vfWIn:           np.array 1xN input weight matrix.

        :param tDt:             float Length of single time step
        :param fNoiseStd:       float Standard deviation of noise

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron thresholds. Default: -65mV
        
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

        # - Handle unit of tDt: if no unit provided, assume it is in seconds
        tDt = np.asscalar(np.array(tDt)) * second

        ### --- Neurons

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


        ### --- Synapses

        # - Add recurrent synapses (all-to-all)
        self._sgRecurrentSynapses = teiliSyn(
            self._ngLayer, self._ngLayer,
            equation_builder=teiliDPISynEqts,
            method = strIntegrator,
            dt = tDt,
            name = 'reservoir_recurrent_synapses'
        )
        self._sgRecurrentSynapses.connect()

        # - Add source -> reservoir synapses (one-to-one)
        self._sgReceiver = teiliSyn(
            self._sggInput, self._ngLayer,
            equation_builder=teiliDPISynEqts,
            method = strIntegrator,
            dt = np.asarray(tDt) * second,
            name = 'receiver_synapses')
        # Each spike generator neuron corresponds to one reservoir neuron
        self._sgReceiver.connect('i==j')

        # - Overwrite default synapse parameters
        if dParamSynapse is not None:
            self._sgRecurrentSynapses.set_params(dParamNeuron)
            self._sgReceiver.set_params(dParamNeuron)


        # - Add spike monitor to record layer outputs
        self._spmReservoir = b2.SpikeMonitor(self._ngLayer, record = True, name = 'layer_spikes')


        # - Call Network constructor
        self._net = b2.Network(self._ngLayer, self._sgRecurrentSynapses,
                                 self._sggInput, self._sgReceiver,
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
        mfW = np.copy(self.mfW)
        vfWIn = np.copy(self.vfWIn)

        # - Reset Network
        self._net.restore('reset')

        # - Restore state variables
        self._ngLayer.Imem = Imem
        self._ngLayer.Iahp = Iahp
        self._sgRecurrentSynapses.Ie_syn = Ie_Recur
        self._sgRecurrentSynapses.Ii_syn = Ii_Recur
        self._sgReceiver.Ie_syn = Ie_Recei
        self._sgReceiver.Ii_syn = Ii_Recei

        # - Restore parameters 
        self.mfW = mfW
        self.vfWIn = vfWIn


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

        # - Infer real duration of evolution
        *__, tDuration = self._prepare_input(tsInput, tDuration)
        tFinal = self.t + tDuration

        # - Extract spike timings and channels
        if tsInput is not None:
            vtEventTimes, vnEventChannels, __ = tsInput.find([self.t, tFinal+self.tDt])
            if np.size(vnEventChannels) > 0:
                # - Make sure channels are within range
                assert np.amax(vnEventChannels) < self.nDimIn, /
                "Only channels between 0 and {} are allowed".format(self.nDimIn-1)
        else:
            vtEventTimes, vnEventChannels = [], []

        ## -- Consider leak as periodic input spike with fixed weight
        
        # - Leak timings
        # First leak is at multiple of self.tTauLeak
        tFirstLeak = b2.ceil(self.t / self.tTauLeak) * self.tTauLeak
        # Maximum possible number of leak steps in evolution period
        nMaxNumLeaks = b2.ceil(tDuration / self.tTauLeak)
        vtLeak = np.arange(nMaxNumLeaks) * self.tTauLeak + tFirstLeak
        vtLeak = vtLeak[vtLeak <= tFinal + fTolAbs*second]

        # - Include leaks in event trace
        # - Assign channel -1 to leak, make sure it is not in vnEventChannels
        assert not (-1 in vnEventChannels), "tsInput must not contain events on channel -1."
        vnEventChannels = np.r_[vnEventChannels, -np.ones_like(vtLeak)]
        vtEventTimes = np.r_[vtEventTimes, vtLeak]
        viSorted = np.sort(np.r_[vtEventTimes, vtLeak])
        vnEventChannels = vnEventChannels[viSorted]
        vtEventTimes = vtEventTimes[viSorted]

        # - Store states in matrix (starting with currrent state at t=self.t)
        mnState = np.zeros((vtEventTimes.size, self.nSize), dtype=self.dtypeState)
        vfStateOld = self.vfState
        for iTimeIndex, (tTime, nChannel) in enumerate(, vtEventTimes, vnEventChannels):
            vfStateNew = np.clip(
                mfState[iTimeIndex, : ] + self._mfW[nChannel],
                self._nStateMin,
                self._nStateMax
            )
            # - Neurons above threshold
            vbAboveThresh = (vStateNew >= self.vfVThresh)
            # - Set state to reset potential
            vStateNew[vbAboveThresh] = self.vfVReset[vbAboveThresh]
            # # - Subtract value from state
            # vStateNew[vbAboveThresh] = np.clip(
            #     vStateNew[vbAboveThresh]-self.vfVReset[vbAboveThresh],
            #     self._nStateMin,
            #     None
            # )
            mnState[iTimeIndex + 1, :]
            # Times not necessary?
        # - Copy latest state entry to save state explicitly for tFinal
        self.vfState = mnState[-1, : ] = mnState[-2, :]

        # - Update time
        self.t += tDuration

        # - Output time series
        return TSContinuous(vtEventTimes, mnState)

    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

    @property
    def cInput(self):
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
            # - Assign recurrent weights
            mfNewW = np.asarray(mfNewW).reshape(self.nSize, -1)
            self._sgRecurrentSynapses.weight = mfNewW.flatten()

    @property
    def vfWIn(self):
        if hasattr(self, '_sgReceiver'):
            return np.reshape(self._sgReceiver.weight, (self.nSize, -1))
        else:
            return self._mfW

    @vfWIn.setter
    def vfWIn(self, vfNewW):
        assert np.size(vfNewW) == self.nSize, \
            '`mfNewW` must have [' + str(self.nSize) + '] elements.'

        self._mfW = vfNewW

        if hasattr(self, '_sgReceiver'):
            # - Assign input weights
            self._sgReceiver.weight = vfNewW.flatten()

    @property
    def vState(self):
        return self._ngLayer.I_mem_

    @vState.setter
    def vState(self, vNewState):
        self._ngLayer.I_mem = np.asarray(self._expand_to_net_size(vNewState, 'vNewState')) * volt

    @property
    def t(self):
        return self._net.t_

    @Layer.tDt.setter
    def tDt(self, _):
        raise ValueError('The `tDt` property cannot be set for this layer')
