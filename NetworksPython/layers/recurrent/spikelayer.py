###
# spikelayer.py - Class implementing a recurrent layer consisting of
#                 I&F-neurons with constant leak. Clock based.
###


import numpy as np
from tqdm import tqdm
from ...timeseries import TSEvent
from .. import Layer


# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9

class SpikingLayer(Layer):
    '''
    SpikingLayer: 
    '''
    def __init__(self,
                 mfWRec: np.ndarray = None,
                 mfWIn: np.ndarray = None, 
                 vfVth: np.ndarray = 8,
                 vfBias: np.ndarray = 0,
                 tDt: float = 1,
                 strName: str = 'unnamed'):
        """
        EventCNLayer - Implements a 2D convolutional layer of spiking neurons

        :param nfWRec:  np.ndarray  Weight matrix
        :param mfWIn:   np.array  nDimInxN input weight matrix.
        :param vfVth:   np.array  Spiking threshold
        :param vfBias:  np.array  Constant bias to be added to state at each time step
        :param tDt:     float  Time step
        :param strName: str  Name of this layer.
        """
        # Call parent constructor
        super().__init__(mfW = mfWIn,
                         tDt = tDt,
                         strName = strName)

        # - Input weights must be provided
        assert mfWRec is not None, 'Recurrent weights mfWRec must be provided.'

        # - One large weight matrix to process input and recurrent connections
        self._mfWTotal = np.zeros((self._nDimIn + self._nSize, self.nSize))

        # - Set neuron parameters
        self.mfW = mfWRec
        self.mfWIn = mfWIn
        self.vfVth = vfVth
        self.vfBias = vfBias

        self.reset_state()
        
        self.__nIdMonitor__ = 0  # Default monitorin of neuron 0

    def evolve(self,
               tsInput: TSEvent = None,
               tDuration: float = None,
               bVerbose: bool = False,
    ) -> (TSEvent, np.ndarray):
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:  TSEvent  Input spike trian
        :param tDuration:   float    Simulation/Evolution time
        :param bVerbose:    bool Currently no effect, just for conformity
        :return:          TSEvent  output spike series

        """

        # - Generate input in rasterized form, get actual evolution duration
        mfInptSpikeRaster, tDuration = self._prepare_input(tsInput, tDuration)

        # Hold the sate of network at any time step when updated
        aStateTimeSeries = []
        ltSpikeTimes = []
        liSpikeIDs = []

        # Record initial state of the network
        self.addToRecord(aStateTimeSeries, 0)

        # Local variables
        vState = self.vState
        vfVth = self.vfVth
        mfWRec = self.mfWRec
        mfWIn = self.mfWIn
        vfBias = self.vfBias
        tDt = self.tDt
        nDimIn = self.nDimIn
        # Number of potential spike sources (input neurons and layer neurons)
        nSpikeSources = self.nDimIn + self.nSize

        # Initialize spike raster with recurrent spikes from last time step
        vbSpikeRaster = self._vbSpikeRaster

        # Iterate over all time steps
        for iCurrentTimeStep in tqdm(mfInptSpikeRaster.shape[0]):
            
            # - Spikes from input synapses
            vbSpikeRaster = mfInptSpikeRaster[iCurrentTimeStep]
                      
            # Update neuron states
            vfUpdate = np.sum(mfWIn[vbSpikeRaster], axis=0) + np.sum(mfWRec[vbRecurrentSpikeRaster])

            # State update
            vState[:] += vfUpdate + vfBias  # Membrane update with synaptic input

            # - Record state before reset
            tCurrentTime = tCurrentTimeStep*tDt
            self.addToRecord(aStateTimeSeries, tCurrentTime, nIdOut=self.__nIdMonitor__)

            # - Check threshold crossings for spikes
            vbSpikeRaster = (vState >= vfVth)

            # - Reset membrane state
            vState[vbSpikeRaster] -= vfVth[vbSpikeRaster]

            # - Record spikes
            ltSpikeTimes += [tCurrentTime] * np.sum(vbSpikeRaster)
            liSpikeIDs += list(np.where(vbSpikeRaster)[0])

            # - Record state after reset
            self.addToRecord(aStateTimeSeries, tCurrentTime, nIdOut=self.__nIdMonitor__)

        # - Update state
        self._vState = vState

        # - Store IDs of neurons that would spike in next time step
        self._vbSpikeRaster = vbSpikeRaster

        # Update time
        self._t += tDuration
        
        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            vtTimeTrace = ltSpikeTimes,
            vnEventChannels = liSpikeIDs,
            nNumChannels = self.nSize)

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        return tseOut

    def addToRecord(self,
                    aStateTimeSeries: list,
                    tCurrentTime: float,
                    nIdOut: int = None,
                    debug: bool = False):
        """
        addToRecord: Convenience function to record current state of the layer
                     or individual neuron

        :param aStateTimeSeries: list  A simple python list object to which the
                                       state needs to be appended
        :param tCurrentTime:     float Current simulation time
        :param nIdOut:           int   Neuron id to record the state of,
                                       if None all the neuron's states
                                       will be added to the record.
                                       Default = None
        """
        # Local variable
        mfState = self.vState

        if nIdOut is None:
            # Update record of state changes
            for nIdOutIter in range(self.nSize):
                aStateTimeSeries.append([tCurrentTime,
                                         nIdOutIter,
                                         mfState[nIdOutIter]])
                if debug:
                    print([tCurrentTime,
                           nIdOutIter,
                           mfState[nIdOutIter, 0]])
        else:
            aStateTimeSeries.append([tCurrentTime,
                                     nIdOut,
                                     mfState[nIdOut]])
            if debug:
                print([tCurrentTime,
                       nIdOutIter,
                       mfState[nIdOutIter, 0]])

        return

    def _prepare_input(
        self,
        tsInput: TSEvent = None,
        tDuration: float = None
    ) -> (np.ndarray, float, float):
        """
        _prepare_input - Sample input, set up time base

        :param tsInput:     TimeSeries TxM or Tx1 Input signals for this layer
        :param tDuration:   float Duration of the desired evolution, in seconds

        :return: (vtEventTimes, vnEventChannels, tDuration, tFinal)
            mfSpikeRaster:    ndarray Boolean raster containing spike info
            tDuration:        float Actual duration for evolution
        """

        # - Determine default duration
        if tDuration is None:
            assert tsInput is not None, \
                'One of `tsInput` or `tDuration` must be supplied'

            if tsInput.bPeriodic:
                # - Use duration of periodic TimeSeries, if possible
                tDuration = tsInput.tDuration

            else:
                # - Evolve until the end of the input TImeSeries
                tDuration = tsInput.tStop - self.t
                assert tDuration > 0, \
                    'Cannot determine an appropriate evolution duration. `tsInput` finishes before the current ' \
                    'evolution time.'

        # - Discretize tDuration wrt self.tDt
        nSamples = tDuration // self.tDt
        tDuration = nSamples * self.tDt

        # - Extract spike timings and channels
        if tsInput is not None:
            # Extract spike data from the input variable
            __, __, mfSpikeRaster, __ = tsInput.raster(
                tDt = self.tDt,
                tStart = self.t,
                tStop = self.t + tDuration,
                vnSelectChannels = np.arange(self.nDimIn),
            )

        else:
            mfSpikeRaster = np.zeros((nSamples, nDimIn), bool)

        return mfSpikeRaster, tDuration