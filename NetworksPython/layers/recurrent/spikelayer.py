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

class RecCLIAF(Layer):
    '''
    RecCLIAF - Recurrent layer of integrate and fire neurons with constant leak
    '''
    def __init__(self,
                 mfWRec: np.ndarray = None,
                 mfWIn: np.ndarray = None, 
                 vfVth: np.ndarray = 8,
                 vfBias: np.ndarray = 0,
                 tDt: float = 1,
                 vnIdMonitor: np.ndarray = [],
                 strName: str = 'unnamed'):
        """
        RecCLIAF - Recurrent layer of integrate and fire neurons with constant leak

        :param nfWRec:  array-like  Weight matrix
        :param mfWIn:   array-like  nDimInxN input weight matrix.
        :param vfVth:   array-like  Spiking threshold
        :param vfBias:  array-like  Constant bias to be added to state at each time step
        :param tDt:     float  Time step
        :vnIdMonitor:   array-like  IDs of neurons to be recorded
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
        self.mfWRec = mfWRec
        self.mfWIn = mfWIn
        self.vfVth = vfVth
        self.vfBias = vfBias

        # - IDs of neurons to be recorded
        self.vnIdMonitor = vnIdMonitor

        self.reset_state()       
        

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

        # Local variables
        vState = self.vState
        vfVth = self.vfVth
        mfWRec = self.mfWRec
        mfWIn = self.mfWIn
        vfBias = self.vfBias
        tDt = self.tDt
        nDimIn = self.nDimIn
        
        nSpikeSources = self.nDimIn + self.nSize  # Number of spike sources (input neurons and layer neurons)
        vbRecSpikeRaster = self._vbRecSpikeRaster  # recurrent spikes from previous time step
        vnIdMonitor = None if self.vnIdMonitor.size == 0 else self.vnIdMonitor

        if vnIdMonitor is not None:
            # Record initial state of the network
            self.addToRecord(aStateTimeSeries, self.t)


        # Iterate over all time steps
        for iCurrentTimeStep in tqdm(range(mfInptSpikeRaster.shape[0])):
            
            # - Spikes from input synapses
            vbInptSpikeRaster = mfInptSpikeRaster[iCurrentTimeStep]
                      
            # Update neuron states
            vfUpdate = np.sum(mfWIn[vbInptSpikeRaster], axis=0) + np.sum(mfWRec[vbRecSpikeRaster])

            # State update
            vState[:] += vfUpdate + vfBias  # Membrane update with synaptic input

            tCurrentTime = iCurrentTimeStep*tDt

            if vnIdMonitor is not None:
                # - Record state before reset
                self.addToRecord(aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor, vState=vState)

            # - Check threshold crossings for spikes
            vbRecSpikeRaster = (vState >= vfVth)

            # - Reset membrane state
            vState[vbRecSpikeRaster] -= vfVth[vbRecSpikeRaster]

            # - Record spikes
            ltSpikeTimes += [tCurrentTime] * np.sum(vbRecSpikeRaster)
            liSpikeIDs += list(np.where(vbRecSpikeRaster)[0])

            if vnIdMonitor is not None:
                # - Record state after reset
                self.addToRecord(aStateTimeSeries, tCurrentTime, vnIdOut=vnIdMonitor)

        # - Update state
        self._vState = vState

        # - Store IDs of neurons that would spike in next time step
        self._vbRecSpikeRaster = vbRecSpikeRaster

        # Update time
        self._t += tDuration
        
        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            vtTimeTrace = ltSpikeTimes,
            vnChannels = liSpikeIDs,
            nNumChannels = self.nSize)

        # TODO: Is there a time series object for this too?
        mfStateTimeSeries = np.array(aStateTimeSeries)

        # This is only for debugging purposes. Should ideally not be saved
        self._mfStateTimeSeries = mfStateTimeSeries

        return tseOut


    def addToRecord(self,
                    aStateTimeSeries: list,
                    tCurrentTime: float,
                    vnIdOut: np.ndarray = True,
                    vState: np.ndarray = None,
                    bDebug: bool = False):
        """
        addToRecord: Convenience function to record current state of the layer
                     or individual neuron

        :param aStateTimeSeries: list  A simple python list object to which the
                                       state needs to be appended
        :param tCurrentTime:     float Current simulation time
        :param vnIdOut:          np.ndarray   Neuron IDs to record the state of,
                                              if True all the neuron's states
                                              will be added to the record.
                                              Default = True
        :param vState:           np.ndarray If not None, record this as state,
                                            otherwise self.vState
        :param bDebug:           bool Print debug info
        """
        
        vState = self.vState if vState is None else vState

        if vnIdOut is True:
            vnIdOut = np.arange(self.nSize)
    
        # Update record of state changes
        for nIdOutIter in np.asarray(vnIdOut):
            aStateTimeSeries.append([tCurrentTime,
                                     nIdOutIter,
                                     vState[nIdOutIter]])
            if bDebug:
                print([tCurrentTime,
                       nIdOutIter,
                       vState[nIdOutIter, 0]])
        

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

    def reset_time(self):
        # - Set internal clock to 0
        self._t = 0
        
    def reset_state(self):
        # - Delete spikes that would arrive in recurrent synapses during next time step
        self._vbRecSpikeRaster = np.zeros(self.nSize, bool)
        # - Reset neuron state to 0
        self._vState = np.zeros(self.nSize)


    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

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
    def mfWRec(self):
        return self._mfWRec

    @mfWRec.setter
    def mfWRec(self, mfNewW):

        self._mfWRec = self._expand_to_weight_size(mfNewW, "mfWRec")

    @property
    def mfWIn(self):
        return self._mfWIn

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        assert np.size(mfNewW) == self.nDimIn * self.nSize, \
            '`mfWIn` must have [{}] elements.'.format(self.nDimIn * self.nSize)

        self._mfWIn = np.array(mfNewW).reshape(self.nDimIn, self.nSize)

    @property
    def vState(self):
        return self._vState

    @vState.setter
    def vState(self, vNewState):
        self._vState = self._expand_to_net_size(vNewState, "vState")

    @property
    def vfVth(self):
        return self._vfVth

    @vfVth.setter
    def vfVth(self, vfNewThresh):
        self._vfVth = self._expand_to_net_size(vfNewThresh, "vfVth")

    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias):

        self._vfBias = self._expand_to_net_size(vfNewBias, 'vfBias')

    @Layer.tDt.setter
    def tDt(self, tNewDt):
        assert tNewDt > 0, "tDt must be greater than 0."
        self._tDt = tNewDt

    @property
    def vnIdMonitor(self):
        return self._vnIdMonitor

    @vnIdMonitor.setter
    def vnIdMonitor(self, vnNewIDs):
        if vnNewIDs is True:
            self._vnIdMonitor = np.arange(self.nSize)
        elif (
            vnNewIDs is None
            or vnNewIDs is False
            or np.size(vnNewIDs) == 0
        ):
            self._vnIdMonitor = np.array([])
        else:
            self._vnIdMonitor = self._expand_to_net_size(vnNewIDs, "vnIdMonitor")

    
    