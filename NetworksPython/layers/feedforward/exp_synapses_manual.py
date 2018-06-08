###
# exp_synapses_manual.py - Class implementing a spike-to-current layer with exponential synapses
###


# - Imports
from typing import Union
import numpy as np
from scipy.signal import fftconvolve

from TimeSeries import TSContinuous, TSEvent
from ..layer import Layer


# - Configure exports
__all__ = ['FFExpSyn']



## - FFExpSyn - Class: define an exponential synapse layer (spiking input)
class FFExpSyn(Layer):
    """ FFExpSyn - Class: define an exponential synapse layer (spiking input)
    """

    ## - Constructor
    def __init__(self,
                 mfW: Union[np.ndarray, int] = None,

                 tDt: float = 0.0001,
                 fNoiseStd: float = 0.001,

                 tTauSyn: float = 0.005,

                 strName: str = 'unnamed'
                 ):
        """
        FFExpSyn - Construct an exponential synapse layer (spiking input)

        :param mfW:             np.array MxN weight matrix
                                int Size of layer -> creates one-to-one conversion layer
        :param tDt:             float Time step for state evolution
        :param fNoiseStd:       float Std. dev. of noise added to this layer

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

        # - Parameter for exponentials
        self.tTauSyn = tTauSyn

        # - set time and state to 0
        self.reset_all()

    

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

        # - Generate spike trains from tsInput
        mSpikeTrains = np.zeros((vtTimeBase.size, self.nSize))
        if tsInput is not None:
            vtEventTimes, vnEventChannels, _ = tsInput.find([vtTimeBase[0], tDuration])
            mSpikeTrains = np.zeros((vtTimeBase.size, self.nSize))
            # - Iterate over channel indices and create their spike trains
            for channel in range(self.nSize):
                # Times with event in current channel
                vtEventTimesChannel = vtEventTimes[np.where(vnEventChannels == channel)]
                # Indices of vtTimeBase corresponding to these times
                viEventIndicesChannel = ((vtEventTimesChannel-vtTimeBase[0]) / self.tDt).astype(int)
                # Set spike trains
                mSpikeTrains[viEventIndicesChannel, channel] = 1

        # Add current state
        mSpikeTrains[0, :] += self.vState

        # - Add a noise trace
        mSpikeTrains += np.random.randn(*mSpikeTrains.shape) * self.fNoiseStd

        # - Define exponential kernel
        vfKernel = np.exp(-np.arange(0, tDuration, self.tDt)/self.tTauSyn)

        # - Apply kernel to spike trains
        mfFiltered = np.zeros_like(mSpikeTrains)
        for channel, vEvents in enumerate(mSpikeTrains.T):
            mfFiltered[:, channel] = fftconvolve(vEvents, vfKernel, 'full')[:vtTimeBase.size]

        # - Update time and state
        self._tDt = vtTimeBase[-1]
        self.vState = mfFiltered[-1]

        return TSContinuous(vtTimeBase, mfFiltered, strName = 'Receiver current')

    ### --- Properties

    @property
    def cInput(self):
        return TSEvent