###
# network.py - Code for encapsulating networks
###

### --- Imports
import numpy as np

from copy import deepcopy

from typing import Callable

from .timeseries import TimeSeries, TSContinuous, TSEvent
from .layers.layer import Layer


# - Configure exports
__all__ = ["Network"]

# - Relative tolerance for float comparions
fTolRel = 1e-5
fTolAbs = 1e-10

### --- Helper functions


def isMultiple(a: float, b: float, fTolRel: float = fTolRel) -> bool:
    """
    isMultiple - Check whether a%b is 0 within some tolerance.
    :param a: float The number that may be multiple of b
    :param b: float The number a may be a multiple of
    :param fTolRel: float Relative tolerance
    :return bool: True if a is a multiple of b within some tolerance
    """
    fMinRemainder = min(a % b, b - a % b)
    return fMinRemainder < fTolRel * b + fTolAbs


### --- Network class


class Network:
    def __init__(self, *layers: Layer):
        """
        Network - Super class to encapsulate several Layers, manage signal routing

        :param layers:   Layers to be added to the network. They will
                         be connected in series. The Order in which
                         they are received determines the order in
                         which they are connected. First layer will
                         receive external input
        """

        # - Network time
        self._t = 0

        # Maintain set of all layers
        self.setLayers = set()

        if layers:
            # - First layer receives external input
            self.lyrInput = self.add_layer(layers[0], bExternalInput=True)

            # - Keep track of most recently added layer
            lyrLastLayer = layers[0]

            # - Add and connect subsequent layers
            for lyr in layers[1:]:
                self.add_layer(lyr, lyrInput=lyrLastLayer)
                lyrLastLayer = lyr

            # - Handle to last layer
            self.lyrOutput = lyrLastLayer

        # - Set evolution order if no layers have been connected
        if not hasattr(self, "lEvolOrder"):
            self.lEvolOrder = self._evolution_order()

    def add_layer(
        self,
        lyr: Layer,
        lyrInput: Layer = None,
        lyrOutput: Layer = None,
        bExternalInput: bool = False,
        bVerbose: bool = False,
    ) -> Layer:
        """
        add_layer - Add a new layer to the network

        Add lyr to self and to self.setLayers. Its attribute name
        is 'lyr'+lyr.strName. Check whether layer with this name
        already exists (replace anyway).
        Connect lyr to lyrInput and lyrOutput.

        :param lyr:             Layer layer to be added to self
        :param lyrInput:        Layer input layer to lyr
        :param lyrOutput:       Layer layer to which lyr is input layer
        :param bExternalInput:  bool This layer receives external input (Default: False)
        :param bVerbose:        bool Print feedback about layer addition (Default: False)

        :return:                Layer lyr
        """
        # - Check whether layer time matches network time
        assert lyr.t == self.t, (
            "Layer time must match network time "
            + "(network: t={}, layer: t={})".format(self.t, lyr.t)
        )

        # - Check whether self already contains a layer with the same name as lyr.
        if hasattr(self, lyr.strName):
            # - Check if layers are the same object.
            if getattr(self, lyr.strName) is lyr:
                print("Layer `{}` is already part of the network".format(lyr.strName))
                return lyr
            else:
                sNewName = lyr.strName
                # - Find a new name for lyr.
                while hasattr(self, sNewName):
                    sNewName = self._new_name(sNewName)
                if bVerbose:
                    print(
                        "A layer with name `{}` already exists.".format(lyr.strName)
                        + "The new layer will be renamed to  `{}`.".format(sNewName)
                    )
                lyr.strName = sNewName

        # - Add set of input layers and flag to determine if lyr receives external input
        lyr.lyrIn = None
        lyr.bExternalInput = bExternalInput

        # - Add lyr to the network
        setattr(self, lyr.strName, lyr)
        if bVerbose:
            print("Added layer `{}` to network\n".format(lyr.strName))

        # - Update inventory of layers
        self.setLayers.add(lyr)

        # - Connect in- and outputs
        if lyrInput is not None:
            self.connect(lyrInput, lyr)
        if lyrOutput is not None:
            self.connect(lyr, lyrOutput)

        # - Make sure evolution order is updated if it hasn't been before
        if lyrInput is None and lyrOutput is None:
            self.lEvolOrder = self._evolution_order()

        return lyr

    @staticmethod
    def _new_name(strName: str) -> str:
        """
        _new_name: Generate a new name by first checking whether
                  the old name ends with '_i', with i an integer.
                  If so, replace i by i+1, otherwise append '_0'
        :param strName:   str - Name to be modified
        :return:        str - Modified name
        """

        # - Check wheter strName already ends with '_...'
        lsSplitted = strName.split("_")
        if len(lsSplitted) > 1:
            try:
                # - If the part after the last '_' is an integer, raise it by 1
                i = int(lsSplitted[-1])
                lsSplitted[-1] = str(i + 1)
                sNewName = "_".join(lsSplitted)
            except ValueError:
                sNewName = strName + "_0"
        else:
            sNewName = strName + "_0"

        return sNewName

    def remove_layer(self, lyrDel: Layer):
        """
        remove_layer: Remove a layer from the network by removing it
                      from the layer inventory and make sure that no
                      other layer receives input from it.
        :param lyrDel: Layer to be removed from network
        """

        # - Remove connections from lyrDel to others
        for lyr in self.setLayers:
            if lyrDel is lyr.lyrIn:
                lyr.lyrIn = None

        # - Remove lyrDel from the inventory and delete it
        self.setLayers.remove(lyrDel)

        # - Reevaluate the layer evolution order
        self.lEvolOrder = self._evolution_order()

    def connect(self, lyrSource: Layer, lyrTarget: Layer, bVerbose: bool = False):
        """
        connect: Connect two layers by defining one as the input layer
                 of the other.
        :param lyrSource:   The source layer
        :param lyrTarget:   The target layer
        :param bVerbose:    bool Flag indicating whether to print feedback
        """

        # - Make sure that layer dimensions match
        if lyrSource.nSize != lyrTarget.nSizeIn:
            raise NetworkError(
                "Dimensions of layers `{}` (nSize={}) and `{}`".format(
                    lyrSource.strName, lyrSource.nSize, lyrTarget.strName
                )
                + " (nSizeIn={}) do not match".format(lyrTarget.nSizeIn)
            )

        # - Check for compatible input / output
        if lyrSource.cOutput != lyrTarget.cInput:
            raise NetworkError(
                "Input / output classes of layer `{}` (cOutput = {})".format(
                    lyrSource.strName, lyrSource.cOutput.__name__
                )
                + " and `{}` (cInput = {}) do not match".format(
                    lyrTarget.strName, lyrTarget.cInput.__name__
                )
            )

        # - Add source layer to target's set of inputs
        lyrTarget.lyrIn = lyrSource

        # - Make sure that the network remains a directed acyclic graph
        #   and reevaluate evolution order
        try:
            self.lEvolOrder = self._evolution_order()
            if bVerbose:
                print(
                    "Layer `{}` now receives input from layer `{}` \n".format(
                        lyrTarget.strName, lyrSource.strName
                    )
                )
        except NetworkError as e:
            lyrTarget.lyrIn = None
            raise e

    def disconnect(self, lyrSource: Layer, lyrTarget: Layer):
        """
        disconnect: Remove the connection between two layers by setting
                    the input of the target layer to None.
        :param lyrSource:   The source layer
        :param lyrTarget:   The target layer
        """

        # - Check whether layers are connected at all
        if lyrTarget.lyrIn is lyrSource:
            # - Remove the connection
            lyrTarget.lyrIn = None
            print(
                "Layer {} no longer receives input from layer `{}`".format(
                    lyrTarget.strName, lyrSource.strName
                )
            )

            # - Reevaluate evolution order
            try:
                self.lEvolOrder = self._evolution_order()
            except NetworkError as e:
                raise e

        else:
            print(
                "There is no connection from layer `{}` to layer `{}`".format(
                    lyrSource.strName, lyrTarget.strName
                )
            )

    def _evolution_order(self) -> list:
        """
        _evolution_order() - Determine the order in which layers are evolved. Requires Network
        to be a directed acyclic graph, otherwise evolution has to happen
        timestep-wise instead of layer-wise.
        """

        # - Function to find next evolution layer
        def next_layer(setCandidates: set) -> Layer:
            while True:
                try:
                    lyrCandidate = setCandidates.pop()
                # If no candidate is left, raise an exception
                except KeyError:
                    raise NetworkError("Cannot resolve evolution order of layers")
                    # Could implement timestep-wise evolution...
                else:
                    # - If there is a candidate and none of the remaining layers
                    #   is its input layer, this will be the next
                    if not (lyrCandidate.lyrIn in setlyrRemaining):
                        return lyrCandidate

        # - Set of layers that are not in evolution order yet
        setlyrRemaining = self.setLayers.copy()

        # # - Begin with input layer
        # lOrder = [self.lyrInput]
        # setlyrRemaining.remove(self.lyrInput)
        lOrder = []

        # - Loop through layers
        while bool(setlyrRemaining):
            # - Find the next layer to be evolved
            lyrNext = next_layer(setlyrRemaining.copy())
            lOrder.append(lyrNext)
            setlyrRemaining.remove(lyrNext)

        # - Return a list with the layers in their evolution order
        return lOrder

    def _fix_duration(self, t: float) -> float:
        """
        _fix_duration - Due to rounding errors it can happen that a
                        duration or end time t is slightly below its intened
                        value, causing the layers to not evolve sufficiently.
                        This method fixes the problem by increasing
                        t if it is slightly below a multiple of
                        tDt of any of the layers in the network.
            :param t:   float - time to be fixed
            :return:    float - Fixed duration
        """
        
        # - All tDt
        vtDt = np.array([lyr.tDt for lyr in self.lEvolOrder])

        if (
            (np.abs(t % vtDt) > fTolAbs)
            & (np.abs(t % vtDt) - vtDt < fTolAbs)
        ).any():
            return t + fTolAbs
        else: 
            return t


    def evolve(
        self,
        tsExternalInput: TimeSeries = None,
        tDuration: float = None,
        bVerbose: bool = True,
    ) -> dict:
        """
        evolve - Evolve each layer in the network according to self.lEvolOrder.
                 For layers with bExternalInput==True their input is
                 tsExternalInput. If not but an input layer is defined, it
                 will be the output of that, otherwise None.
                 Return a dict with each layer's output.
        :param tsExternalInput:  TimeSeries with external input data.
        :param tDuration:        float - duration over which netŵork should
                                         be evolved. If None, evolution is
                                         over the duration of tsExternalInput
        :param bVerbose:         bool - Print info about evolution state
        :return:                 Dict with each layer's output time Series
        """

        # - Determine default duration
        if tDuration is None:
            assert (
                tsExternalInput is not None
            ), "One of `tsExternalInput` or `tDuration` must be supplied"

            if tsExternalInput.bPeriodic:
                # - Use duration of periodic TimeSeries, if possible
                tDuration = tsExternalInput.tDuration

            else:
                # - Evolve until the end of the input TimeSeries
                tDuration = tsExternalInput.tStop - self.t
                assert tDuration > 0, (
                    "Cannot determine an appropriate evolution duration. "
                    + "`tsExternalInput` finishes before the current evolution time."
                )

        # - Correct tDuration and last point of time series in case of rounding errors
        tDuration = self._fix_duration(tDuration)
        if tsExternalInput is not None:
            tsExternalInput.vtTimeTrace[-1] = self._fix_duration(tsExternalInput.vtTimeTrace[-1])

        # - List of layers where tDuration is not a multiple of tDt
        llyrDtMismatch = list(
            filter(lambda lyr: not isMultiple(tDuration, lyr.tDt), self.lEvolOrder)
        )

        # - Throw an exception if llyrDtMismatch is not empty, showing for
        #   which layers there is a mismatch
        if llyrDtMismatch:
            strLayers = ", ".join(
                ("{}: tDt={}".format(lyr.strName, lyr.tDt) for lyr in llyrDtMismatch)
            )
            raise ValueError(
                "`tDuration` ({}) is not a multiple of `tDt`".format(tDuration)
                + " for the following layer(s):\n"
                + strLayers
            )
        ## (this can actually introduce more errors) ##
        # # - Correct tDuration in case of rounding errors
        # tDuration = int(np.round(tDuration / self.lEvolOrder[0].tDt)) * self.lEvolOrder[0].tDt


        # - Set external input name if not set already
        if tsExternalInput.strName is None:
            tsExternalInput.strName = 'External input'

        # - Dict to store external input and each layer's output time series
        dtsSignal = {"external": tsExternalInput}

        # - Make sure layers are in sync with netowrk
        self._check_sync(bVerbose=False)

        # - Iterate over evolution order and evolve layers
        for lyr in self.lEvolOrder:

            # - Determine input for current layer
            if lyr.bExternalInput:
                # External input
                tsCurrentInput = tsExternalInput
                strIn = "external input"
            elif lyr.lyrIn is not None:
                # Output of current layer's input layer
                tsCurrentInput = dtsSignal[lyr.lyrIn.strName]
                strIn = lyr.lyrIn.strName + "'s output"
            else:
                # No input
                tsCurrentInput = None
                strIn = "nothing"

            if bVerbose:
                print("Evolving layer `{}` with {} as input".format(lyr.strName, strIn))

            # - Evolve layer and store output in dtsSignal
            dtsSignal[lyr.strName] = lyr.evolve(tsCurrentInput, tDuration, bVerbose)

            # - Set name for time series, if not already set
            if dtsSignal[lyr.strName].strName is None:
                dtsSignal[lyr.strName].strName = lyr.strName

        # - Update network time
        self._t += tDuration

        # - Make sure layers are still in sync with netowrk
        self._check_sync(bVerbose=False)

        # - Return dict with layer outputs
        return dtsSignal

    def train(
        self,
        fhTraining: Callable,
        tsExternalInput: TimeSeries = None,
        tDuration: float = None,
        tDurBatch: float = None,
        bVerbose=True,
        bHighVerbosity=False,
    ):
        """
        train - Train the network batch-wise by evolving the layers and
                calling fhTraining.

        :param fhTraining:      Function that is called after each evolution
                fhTraining(netObj, dtsSignals, bFirst, bFinal)
                :param netObj:      Network the network object to be trained
                :param dtsSignals:  Dictionary containing all signals in the current evolution batch
                :param bFirst:      bool Is this the first batch?
                :param bFinal:      bool Is this the final batch?

        :param tsExternalInput: TimeSeries with external input to network
        :param tDuration:       float - Duration over which netŵork should
                                        be evolved. If None, evolution is
                                        over the duration of tsExternalInput
        :param tDurBatch:       float - Duration of one batch (can also pass array with several values)
        :param bVerbose:        bool  - Print info about training progress
        :param bHighVerbosity:  bool  - Print info about layer evolution
                                        (only has effect if bVerbose is True)
        """

        # - Determine duration of training
        if tDuration is None:
            assert (
                tsExternalInput is not None
            ), "One of `tsExternalInput` or `tDuration` must be supplied"

            if tsExternalInput.bPeriodic:
                # - Use duration of periodic TimeSeries, if possible
                tFinal = self._fix_duration(self.t + tsExternalInput.tDuration)

            else:
                # - Evolve until the end of the input TimeSeries
                tFinal = self._fix_duration(tsExternalInput.tStop)
                assert tFinal > self.t, (
                    "Cannot determine an appropriate evolution duration. "
                    + "`tsExternalInput` finishes before the current evolution time."
                )
        else:
            tFinal = self._fix_duration(self.t + tDuration)

        tRemaining = self._fix_duration(tFinal - self.t)

        # - Determine batch duration and number
        if tDurBatch is None:
            vtDurBatch = np.array([tRemaining])
        else:
            # - Batch duration the same for all batches
            if np.size(tDurBatch) == 1:
                nNumBatches = (np.ceil(np.asarray(tRemaining) / tDurBatch)).astype(int)
                vtDurBatch = np.repeat(tDurBatch, nNumBatches)
            else:
                # - Generate iterable with possibly different durations for each batch
                # - Time value after each batch
                vtPassed = np.cumsum(tDurBatch)
                # - If sum of batch durations larger than tRemaining, ignore last entries
                if vtPassed[-1] > tRemaining:
                    # Index of first element with t past tRemaining - include only until here
                    # It is not dramatic if the last(!) batch goes beyond tRemaining
                    iFirstPast = np.where(vtPassed > tRemaining)[0][0]
                    vtDurBatch = tDurBatch[: iFirstPast+1]
                elif vtPassed[-1] < tRemaining:
                    # - Add batch with missing duration
                    vtDurBatch = np.r_[tDurBatch, tRemaining-vtPassed[-1]]
        nNumBatches = np.size(vtDurBatch)
        
        ## -- Actual training starts here:

        # - Iterate over batches
        bFirst = True
        bFinal = False
        for nBatch, tDurBatch in enumerate(vtDurBatch):
            
            # - Duration of next batch
            tRemaining = tFinal - self.t
            tCurrentDur = self._fix_duration(min(tDurBatch, tRemaining))
            
            if bVerbose:
                print(
                    "\rTraining batch {} of {} from t={:.3f} to {:.3f}.              ".format(
                        nBatch+1, nNumBatches, self.t, self.t+tCurrentDur, end=""
                    )
                )
            # - Evolve network
            dtsSignal = self.evolve(
                tsExternalInput.resample_within(self.t, self.t+tCurrentDur),
                tCurrentDur,
                bVerbose=(bHighVerbosity and bVerbose),
            )
            # - Remaining simulation time
            tRemaining -= tDurBatch
            # - Determine if this batch was the first or the last of training
            if nBatch == nNumBatches-1:
                bFinal = True
            # - Call the callback
            fhTraining(self, dtsSignal, bFirst, bFinal)
            bFirst = False

        if bVerbose:
            print("\nTraining successful\n")

    def stream(self,
               tsInput: TimeSeries,
               tDuration: float,
               bVerbose: bool = False,
               fhStepCallback: Callable = None,
               ) -> dict:
        """
        stream - Stream data through layers, evolving by single time steps

        :param tsInput:         TimeSeries External input to the network
        :param tDuration:       float Total duration to stream for
        :param bVerbose:        bool Display feedback
        :param fhStepCallback:  Callable[Network]

        :return: dtsSignals dict Collected output signals from each layer
        """

        # - Check that all layers implement the streaming interface
        assert all([hasattr(lyr, 'stream') for lyr in self.setLayers]), \
            'Not all layers implement the `stream` interface.'

        # - Check that external input has the correct class
        assert isinstance(tsInput, self.lyrInput.cInput), \
            'External input must be of class {} for this network.'.format(self.lyrInput.cInput.__name__)

        # - Check that external input has the correct size
        assert tsInput.nNumTraces == self.lyrInput.nSizeIn, \
            'External input must have {} traces for this network.'.format(self.lyrInput.nSizeIn)

        # - Find the largest common tDt
        self.ltDts = [lyr.tDt for lyr in self.setLayers]
        self.tCommonDt = max(self.ltDts)
        if bVerbose: print('Net: Common time base: ', self.tCommonDt)

        # - Prepare time base
        vtTimeBase = np.arange(0, tDuration + self.tCommonDt, self.tCommonDt) + self._t
        vtTimeBase = vtTimeBase[vtTimeBase <= self._t + tDuration]
        tDuration = vtTimeBase[-1] - vtTimeBase[0]
        nNumSteps = np.size(vtTimeBase)-1

        # - Prepare all layers
        self.lStreamers = [lyr.stream(tDuration, self.tCommonDt, bVerbose = bVerbose)
                           for lyr in self.lEvolOrder]
        nNumLayers = np.size(self.lEvolOrder)

        # - Prepare external input
        if tsInput is not None:
            lInput = [tsInput.find((t, t + self.tCommonDt))
                      for t in vtTimeBase]
        else:
            lInput = [None] * nNumSteps

        # - Get initial state of all layers
        if bVerbose: print('Net: getting initial state')

        # - Determine input state size, obtain initial layer state
        tupInputState = lInput[0]
        lLastState = [tupInputState] + [deepcopy(lyr.send(None)) for lyr in self.lStreamers]

        # - Initialise layer output variables with initial state, convert to lists
        lLayerOutputs = [tuple([np.reshape(x, (1, -1))] for x in state) for state in lLastState[1:]]

        # - Display some feedback
        if bVerbose:
            print('Net: got initial state:')
            print(lLayerOutputs)

        # - Streaming loop
        lState = deepcopy(lLastState)
        for nStep in range(nNumSteps):
            if bVerbose: print('Net: Start of step', nStep)

            # - Set up external input
            lLastState[0] = (lInput[nStep])

            # - Loop over layers, stream data in and out
            for nLayerInd in range(nNumLayers):
                # - Display some feedback
                if bVerbose: print('Net: Evolving layer {}'.format(nLayerInd))

                # - Try / Catch to handle end of streaming iteration
                try:
                    # - `send` input data for current layer
                    # - wait for the output state for the current layer
                    lState[nLayerInd + 1] = deepcopy(self.lStreamers[nLayerInd].send(lLastState[nLayerInd]))

                except StopIteration as e:
                    # - StopIteration returns the final state
                    lState[nLayerInd + 1] = e.args[0]

            # - Collate layer outputs
            for nLayer in range(nNumLayers):
                for nTupleIndex in range(len(lLayerOutputs[nLayer])):
                    lLayerOutputs[nLayer][nTupleIndex].append(np.reshape(lState[nLayer + 1][nTupleIndex], (1, -1)))

            # - Save last state to use as input for next step
            lLastState = deepcopy(lState)

            # - Call callback function
            if fhStepCallback is not None:
                fhStepCallback(self)

        # - Build return dictionary
        dtsSignal = {'external': tsInput.copy()}
        for nLayer in range(nNumLayers):
            # - Concatenate time series
            lvData = [np.stack(np.array(data, 'float')) for data in lLayerOutputs[nLayer]]

            # - Filter out nans in time trace (always first data element)
            vbUseSamples = ~np.isnan(lvData[0]).flatten()
            tupData = tuple(data[vbUseSamples, :] for data in lvData)

            if bVerbose: print(tupData[0])

            # - Build output dictionary (using appropriate output class)
            dtsSignal[self.lEvolOrder[nLayer].strName] = self.lEvolOrder[nLayer].cOutput(*tupData)

            # - Set name for time series, if not already set
            if dtsSignal[self.lEvolOrder[nLayer].strName].strName is None:
                dtsSignal[self.lEvolOrder[nLayer].strName].strName = self.lEvolOrder[nLayer].strName

        # - Increment time
        self._t += tDuration

        # - Return collated signals
        return dtsSignal

    def _check_sync(self, bVerbose: bool = True) -> bool:
        """
        _check_sync - Check whether the time t of all layers matches self.t
                     If not, throw an exception.
        """
        bSync = True
        if bVerbose:
            print("Network time is {}. \n\t Layer times:".format(self.t))
            print(
                "\n".join(
                    (
                        "\t\t {}: {}".format(lyr.strName, lyr.t)
                        for lyr in self.lEvolOrder
                    )
                )
            )
        for lyr in self.lEvolOrder:
            if abs(lyr.t - self.t) >= fTolRel * self.t + fTolAbs:
                bSync = False
                print(
                    "\t WARNING: Layer `{}` is not in sync (t={})".format(
                        lyr.strName, lyr.t
                    )
                )
        if bSync:
            if bVerbose:
                print("\t All layers are in sync with network.")
        else:
            raise NetworkError("Not all layers are in sync with the network.")
        return bSync

    def reset_time(self):
        """
        reset_time() - Reset the time of the network to zero. Does not reset state.
        """
        # - Reset time for each layer
        for lyr in self.setLayers:
            lyr.reset_time()

        # - Reset global network time
        self._t = 0

    def reset_state(self):
        """
        reset_state() - Reset the state of the network. Does not reset time.
        """
        # - Reset state for each layer
        for lyr in self.setLayers:
            lyr.reset_state()

    def reset_all(self):
        """
        reset_all() - Reset state and time of the network.
        """
        for lyr in self.setLayers:
            lyr.reset_all()

        # - Reset global network time
        self._t = 0

    def __repr__(self):
        return (
            "{} object with {} layers\n".format(
                self.__class__.__name__, len(self.setLayers)
            )
            + "    "
            + "\n    ".join([str(lyr) for lyr in self.lEvolOrder])
        )

    @property
    def t(self):
        return self._t

    # @fDt.setter
    # def fDt(self, fNewDt):
    #     self.__fDt = fNewDt
    #     for lyr in self.setLayers:
    #         lyr.fDt = self.__fDt


### --- NetworkError exception class
class NetworkError(Exception):
    pass
