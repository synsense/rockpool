###
# averagepooling.py - Class implementing a pooling layer.
###

import numpy as np

# from typing import Optional, Union, List, Tuple
from ...timeseries import TSEvent
from ..layer import Layer
from ..cnnweights import CNNWeight


class AveragePooling(Layer):
    """
    AveragePooling: Implements average pooling by simply merging inputs. So this is more of sum than average pooling.
    """

    def __init__(
        self,
        inShape: tuple,
        pool_size: tuple = (1, 1),
        tDt: float = 1,
        strName: str = "unnamed",
    ):
        """
        :param inShape:     tuple Input shape
        :param pool_size:   tuple Pooling width along each dimension
        :param tDt:         float  Time step for simulations
        :param strName:     str  Name of this layer.
        """

        # Call parent constructor
        mfW = CNNWeight(
            inShape=inShape, nKernels=1, kernel_size=pool_size, strides=pool_size
        )  # Simple hack
        super().__init__(mfW=mfW, tDt=np.asarray(tDt), strName=strName)

        self.reset_state()

    def evolve(
        self, tsInput: TSEvent = None, tDuration: float = None, bVerbose: bool = False
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

        # - Count number of spikes for each neuron in each time step
        vnNumSpikes = np.zeros(nSize, int)
        # - Time before first time step
        tCurrentTime = self.t

        if vnIdMonitor is not None:
            # Record initial state of the network
            self._add_to_record(aStateTimeSeries, tCurrentTime)

        # Do average pooling here

        # Update time
        self._t += tDuration

        # Convert arrays to TimeSeries objects
        tseOut = TSEvent(
            vtTimeTrace=ltSpikeTimes, vnChannels=liSpikeIDs, nNumChannels=self.nSize
        )

        return tseOut
