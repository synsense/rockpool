###
# network.py - Code for encapsulating networks
###

### --- Imports
import json

import numpy as np
from decimal import Decimal
from copy import deepcopy
from NetworksPython import layers

try:
    from tqdm.autonotebook import tqdm

    bUseTqdm = True
except ImportError:
    bUseTqdm = False


from typing import Callable, Union

from ..timeseries import TimeSeries
from ..layers import Layer

RealValue = Union[float, Decimal, str]

# - Configure exports
__all__ = ["Network"]

# - Relative tolerance for float comparisons
fTolRel = 1e-7
fTolAbs = 1e-10

### --- Helper functions


def is_multiple(
    a: RealValue,
    b: RealValue,
    fTolRel: RealValue = fTolRel,
    fTolAbs: RealValue = fTolAbs,
) -> bool:
    """
    is_multiple - Check whether a%b is 0 within some tolerance.
    :param a: float The number that may be multiple of b
    :param b: float The number a may be a multiple of
    :param fTolRel: float Relative tolerance
    :return bool: True if a is a multiple of b within some tolerance
    """
    # - Convert to decimals
    a = Decimal(str(a))
    b = Decimal(str(b))
    fTolRel = Decimal(str(fTolRel))
    fTolAbs = Decimal(str(fTolAbs))
    fMinRemainder = min(a % b, b - a % b)
    return fMinRemainder < fTolRel * b + fTolAbs


def gcd(a: RealValue, b: RealValue) -> Decimal:
    """ gcd - Return the greatest common divisor of two values a and b"""
    a = Decimal(str(a))
    b = Decimal(str(b))
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def lcm(a: RealValue, b: RealValue) -> Decimal:
    """ lcm - Return the least common multiple of two values a and b"""
    a = Decimal(str(np.round(float(a) / fTolRel)))
    b = Decimal(str(np.round(float(b) / fTolRel)))
    return a / gcd(a, b) * b * Decimal(str(fTolRel))


### --- Network class


class Network:
    def __init__(self, *layers: Layer, tDt=None, inp2out=False):
        """
        Network - Super class to encapsulate several Layers, manage signal routing

        :param layers:   Layers to be added to the network. They will
                         be connected in series (or not). The Order in which
                         they are received determines the order in
                         which they are connected. First layer will
                         receive external input
        :param tDt:      float If not none, network time step is forced to
                               this values. Layers that are added must have
                               time step that is multiple of tDt.
                               If None, network will try to determine
                               suitable tDt each time a layer is added.
        :param inp2out:  bool if true input connects to output else not
        """

        self.inp2out = inp2out

        # - Network time
        self._nTimeStep = 0

        # Maintain set of all layers
        self.setLayers = set()

        if tDt is not None:
            assert tDt > 0, "Network: tDt must be positive."
            # - Force tDt
            self._tDt = tDt
            self._bForceDt = True
        else:
            self._bForceDt = False

        if layers:
            for lyr in layers:
                if lyr.strName == "input":
                    self.inputLayer = lyr
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

        # - Set evolution order and time step if no layers have been connected
        if not hasattr(self, "lEvolOrder"):
            self.lEvolOrder = self._evolution_order()
        if not hasattr(self, "_tDt"):
            self._tDt = None

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
        assert np.isclose(lyr.t, self.t), (
            "Network: Layer time must match network time "
            + "(network: t={}, layer: t={})".format(self.t, lyr.t)
        )

        # - Check whether self already contains a layer with the same name as lyr.
        if hasattr(self, lyr.strName):
            # - Check if layers are the same object.
            if getattr(self, lyr.strName) is lyr:
                print(
                    "Network: Layer `{}` is already part of the network".format(
                        lyr.strName
                    )
                )
                return lyr
            else:
                sNewName = lyr.strName
                # - Find a new name for lyr.
                while hasattr(self, sNewName):
                    sNewName = self._new_name(sNewName)
                if bVerbose:
                    print(
                        "Network: A layer with name `{}` already exists.".format(
                            lyr.strName
                        )
                        + "The new layer will be renamed to  `{}`.".format(sNewName)
                    )
                lyr.strName = sNewName

        # - Add set of input layers and flag to determine if lyr receives external input
        lyr.lyrIn = None
        lyr.bExternalInput = bExternalInput

        # - Add lyr to the network
        setattr(self, lyr.strName, lyr)
        if bVerbose:
            print("Network: Added layer `{}` to network\n".format(lyr.strName))

        # - Update inventory of layers
        self.setLayers.add(lyr)

        # - Update global tDt
        self._set_tDt()

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

        # - Update global tDt
        self._tDt = self._set_tDt()

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

        if lyrSource.nSize != lyrTarget.nSizeIn and lyrTarget.strName != "output":
            raise NetworkError(
                "Network: Dimensions of layers `{}` (nSize={}) and `{}`".format(
                    lyrSource.strName, lyrSource.nSize, lyrTarget.strName
                )
                + " (nSizeIn={}) do not match".format(lyrTarget.nSizeIn)
            )
        elif lyrSource.nSize + int(self.inp2out) * self.inputLayer.nSize != lyrTarget.nSizeIn and lyrTarget.strName == "output":
            raise NetworkError(
                "Network: Dimensions of layers `{}` (nSize={}) and `{}`".format(
                    lyrSource.strName, lyrSource.nSize, lyrTarget.strName
                )
                + " (nSizeIn={}) do not match".format(lyrTarget.nSizeIn)
            )

        # - Check for compatible input / output
        if lyrSource.cOutput != lyrTarget.cInput:
            raise NetworkError(
                "Network: Input / output classes of layer `{}` (cOutput = {})".format(
                    lyrSource.strName, lyrSource.cOutput.__name__
                )
                + " and `{}` (cInput = {}) do not match".format(
                    lyrTarget.strName, lyrTarget.cInput.__name__
                )
            )

        # - Add source layer to target's set of inputs
        if lyrTarget.strName == 'output':
            pass
        lyrTarget.lyrIn = lyrSource

        # - Make sure that the network remains a directed acyclic graph
        #   and reevaluate evolution order
        try:
            self.lEvolOrder = self._evolution_order()
            if bVerbose:
                print(
                    "Network: Layer `{}` now receives input from layer `{}` \n".format(
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
                "Network: Layer {} no longer receives input from layer `{}`".format(
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
                "Network: There is no connection from layer `{}` to layer `{}`".format(
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
                    raise NetworkError(
                        "Network: Cannot resolve evolution order of layers"
                    )
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

    def _set_tDt(self, fMaxFactor: float = 100):
        """
        _set_tDt - Set a time step size for the network
                   which is the lcm of all layers' tDt's.
            :param fMaxFactor   float - By which factor can the network tDt
                                        exceed the largest layer tDt before
                                        an error is assumed
        """
        if self._bForceDt:
            # - Just make sure layer tDt are multiples of self.tDt
            for lyr in self.setLayers:
                assert is_multiple(
                    self.tDt, lyr.tDt
                ), "Network: tDt is set to {}, which is not a multiple of layer `{}`'s time step ({}).".format(
                    self.tDt, lyr.strName, lyr.tDt
                )
        else:
            ## -- Try to determine self._tDt from layer time steps
            # - Collect layer time steps, convert to Decimals for numerical stability
            ltDt = [Decimal(str(lyr.tDt)) for lyr in self.setLayers]

            # - If list is empty, there are no layers in the network
            if not ltDt:
                return None

            # - Determine lcm
            tLCM = ltDt[0]
            for tDt in ltDt[1:]:
                tLCM = lcm(tLCM, tDt)

            #   Also
            assert (
                # - If result is way larger than largest tDt, assume it hasn't worked
                tLCM < fMaxFactor * np.amax(ltDt)
                # - Also make sure that tLCM is indeed a multiple of all tDt's
                and not list(filter(lambda tDt: not is_multiple(tLCM, tDt), ltDt))
            ), "Network: Couldn't find a reasonable common time step (layer tDt's: {}, found: {}".format(
                ltDt, tLCM
            )

            # - Store global time step, for now as float for compatibility
            self._tDt = float(tLCM)

        # - Store number of layer time steps per global time step for each layer
        for lyr in self.setLayers:
            lyr._nNumTimeStepsPerGlobal = int(np.round(self._tDt / lyr.tDt))

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

        if ((np.abs(t % vtDt) > fTolAbs) & (np.abs(t % vtDt) - vtDt < fTolAbs)).any():
            return t + fTolAbs
        else:
            return t

    def evolve(
        self,
        tsInput: TimeSeries = None,
        tDuration: float = None,
        nNumTimeSteps: int = None,
        bVerbose: bool = True,
    ) -> dict:
        """
        evolve - Evolve each layer in the network according to self.lEvolOrder.
                 For layers with bExternalInput==True their input is
                 tsInput. If not but an input layer is defined, it
                 will be the output of that, otherwise None.
                 Return a dict with each layer's output.
        :param tsInput:  TimeSeries with external input data.
        :param tDuration:        float - duration over which netŵork should
                                         be evolved. If None, evolution is
                                         over the duration of tsInput
        :param nNumTimeSteps:    int - Number of evolution time steps
        :param bVerbose:         bool - Print info about evolution state
        :return:                 Dict with each layer's output time Series
        """

        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Network: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied"

                if tsInput.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.duration

                else:
                    # - Evolve until the end of the input TimeSeries
                    tDuration = tsInput.t_stop - self.t
                    assert tDuration > 0, (
                        "Network: Cannot determine an appropriate evolution duration. "
                        + "`tsInput` finishes before the current evolution time."
                    )
            nNumTimeSteps = int(np.floor(tDuration / self.tDt))

        if tsInput is not None:
            # - Set external input name if not set already
            if tsInput.name is None:
                tsInput.name = "External input"
            # - Check if input contains information about trial timings
            try:
                vtTrialStarts = tsInput.vtTrialStarts
            except AttributeError:
                vtTrialStarts = None
        else:
            vtTrialStarts = None

        # - Dict to store external input and each layer's output time series
        dtsSignal = {"external": tsInput}

        # - Make sure layers are in sync with netowrk
        self._check_sync(bVerbose=False)

        # - Iterate over evolution order and evolve layers
        for lyr in self.lEvolOrder:

            # - Determine input for current layer
            if lyr.bExternalInput:
                # - External input
                tsCurrentInput = tsInput
                strIn = "external input"

            elif lyr.lyrIn is not None:
                if lyr.strName == "output" and self.inp2out:
                    strIn = ''

                    tsCurrentInput = dtsSignal[lyr.lyrIn.strName]
                    tsCurrentInput.append_c(dtsSignal['input'], inplace=True)

                    strIn += "input" + "'s output"
                    strIn += lyr.lyrIn.strName + "'s output"
                else:
                    # - Output of current layer's input layer
                    tsCurrentInput = dtsSignal[lyr.lyrIn.strName]
                    strIn = lyr.lyrIn.strName + "'s output"

            else:
                # - No input
                tsCurrentInput = None
                strIn = "nothing"

            if bVerbose:
                print(
                    "Network: Evolving layer `{}` with {} as input".format(
                        lyr.strName, strIn
                    )
                )
            # - Evolve layer and store output in dtsSignal
            dtsSignal[lyr.strName] = lyr.evolve(
                tsInput=tsCurrentInput,
                nNumTimeSteps=int(nNumTimeSteps * lyr._nNumTimeStepsPerGlobal),
                bVerbose=bVerbose,
            )

            # - Add information about trial timings if present
            if vtTrialStarts is not None:
                dtsSignal[lyr.strName].vtTrialStarts = vtTrialStarts.copy()

            # - Set name for response time series, if not already set
            if dtsSignal[lyr.strName].name is None:
                dtsSignal[lyr.strName].name = lyr.strName

        # - Update network time
        self._nTimeStep += nNumTimeSteps

        # - Make sure layers are still in sync with netowrk
        self._check_sync(bVerbose=False)

        # - Return dict with layer outputs
        return dtsSignal

    def train(
        self,
        fhTraining: Callable,
        tsInput: TimeSeries = None,
        tDuration: float = None,
        vtDurBatch: float = None,
        nNumTimeSteps: int = None,
        vnNumTSBatch: int = None,
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

        :param tsInput:         TimeSeries with external input to network
        :param tDuration:       float - Duration over which netŵork should
                                        be evolved. If None, evolution is
                                        over the duration of tsInput
        :param vtDurBatch:      Array-like or float - Duration of one batch (can also pass array with several values)
        :param nNumTimeSteps:   int   - Total number of training time steps
        :param vnNumTSBatch:    Array-like or int - Number of time steps per batch
        :param bVerbose:        bool  - Print info about training progress
        :param bHighVerbosity:  bool  - Print info about layer evolution
                                        (only has effect if bVerbose is True)
        """

        if nNumTimeSteps is None:
            # - Try to determine nNumTimeSteps from tDuration
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Network: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied"

                if tsInput.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.duration

                else:
                    # - Evolve until the end of the input TimeSeries
                    tDuration = tsInput.t_stop - self.t
                    assert tDuration > 0, (
                        "Network: Cannot determine an appropriate evolution duration. "
                        + "`tsInput` finishes before the current evolution time."
                    )
            nNumTimeSteps = int(np.floor(tDuration / self.tDt))

        # - Number of time steps per batch
        if vnNumTSBatch is None:
            if vtDurBatch is None:
                vnTSBatch = np.array([nNumTimeSteps], dtype=int)
            elif np.size(vtDurBatch) == 1:
                # - Same value for all batches
                nNumTSSingleBatch = int(
                    np.floor(np.asscalar(np.asarray(vtDurBatch)) / self.tDt)
                )
                nNumBatches = int(np.ceil(nNumTimeSteps / nNumTSSingleBatch))
                vnTSBatch = np.repeat(nNumTSSingleBatch, nNumBatches)
                vnTSBatch[-1] = nNumTimeSteps - np.sum(vnTSBatch[:-1])
            else:
                # - Individual batch durations
                # - Convert batch durations to time step numbers - Rounding down should
                #   not be too problematic as total training will always be nNumTimeSteps
                vnTSBatch = np.floor(np.array(vtDurBatch) / self.tDt).astype(int)
        else:
            if np.size(vnNumTSBatch) == 1:
                # - Same value for all batches
                nNumTSSingleBatch = np.asscalar(np.asarray(vnNumTSBatch, dtype=int))
                nNumBatches = int(np.ceil(nNumTimeSteps / nNumTSSingleBatch))
                vnTSBatch = np.repeat(nNumTSSingleBatch, nNumBatches)
                vnTSBatch[-1] = nNumTimeSteps - np.sum(vnTSBatch[:-1])
            else:
                # - Individual batch durations
                vnTSBatch = np.asarray(vnNumTSBatch)

        # - Make sure time steps add up to nNumTimeSteps
        nTSDiff = nNumTimeSteps - np.sum(vnTSBatch)
        if nTSDiff > 0:
            vnTSBatch = np.r_[vnTSBatch, nTSDiff]
        elif nTSDiff < 0:
            # - Index of first element where cumulated number of time steps > nNumTimeSteps
            iFirstPast = np.where(np.cumsum(vnTSBatch) > nNumTimeSteps)[0][0]
            vnTSBatch = vnTSBatch[: iFirstPast + 1]
            # - Correct last value

        ## -- Actual training starts here:

        # - Iterate over batches
        nNumBatches = np.size(vnTSBatch)

        def batch(nBatch, nTSCurrent, nNumBatches):
            if bHighVerbosity or (bVerbose and not bUseTqdm):
                print(
                    "Network: Training batch {} of {} from t = {:.3f} to {:.3f}.".format(
                        nBatch + 1,
                        nNumBatches,
                        self.t,
                        self.t + nTSCurrent * self.tDt,
                        end="",
                    ),
                    end="\r",
                )

            # - Evolve network
            dtsSignal = self.evolve(
                tsInput=tsInput.clip(
                    self.t, self.t + nTSCurrent * self.tDt, include_stop=True
                ),
                nNumTimeSteps=nTSCurrent,
                bVerbose=bHighVerbosity,
            )

            # - Call the callback
            fhTraining(self, dtsSignal, nBatch == 0, nBatch == nNumBatches - 1)

        if bVerbose and bUseTqdm:
            with tqdm(total=nNumBatches, desc="Network training") as pbar:
                for nBatch, nTSCurrent in enumerate(vnTSBatch):
                    batch(nBatch, nTSCurrent, nNumBatches)
                    pbar.update(1)
        else:
            for nBatch, nTSCurrent in enumerate(vnTSBatch):
                batch(nBatch, nTSCurrent, nNumBatches)

        if bVerbose:
            print(
                "Network: Training successful                                        \n"
            )

    def stream(
        self,
        tsInput: TimeSeries,
        tDuration: float = None,
        nNumTimeSteps: int = None,
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
        assert all(
            [hasattr(lyr, "stream") for lyr in self.setLayers]
        ), "Network: Not all layers implement the `stream` interface."

        # - Check that external input has the correct class
        assert isinstance(
            tsInput, self.lyrInput.cInput
        ), "Network: External input must be of class {} for this network.".format(
            self.lyrInput.cInput.__name__
        )

        # - Check that external input has the correct size
        assert (
            tsInput.num_channels == self.lyrInput.nSizeIn
        ), "Network: External input must have {} traces for this network.".format(
            self.lyrInput.nSizeIn
        )

        if nNumTimeSteps is None:
            # - Try to determine time step number from tDuration
            assert (
                tDuration is not None
            ), "Network: Either `nNumTimeSteps` or `tDuration` must be provided."
            nNumTimeSteps = int(np.floor(tDuration / self.tDt))

        # - Prepare time base
        vtTimeBase = np.arange(nNumTimeSteps + 1) * self._tDt + self.t
        tDuration = vtTimeBase[-1] - vtTimeBase[0]

        # - Prepare all layers
        self.lStreamers = [
            lyr.stream(tDuration, self.tDt, bVerbose=bVerbose)
            for lyr in self.lEvolOrder
        ]
        nNumLayers = np.size(self.lEvolOrder)

        # - Prepare external input
        if tsInput is not None:
            lInput = [tsInput(t, t + self.tDt) for t in vtTimeBase]
        else:
            lInput = [None] * nNumTimeSteps

        # - Get initial state of all layers
        if bVerbose:
            print("Network: getting initial state")

        # - Determine input state size, obtain initial layer state
        tupInputState = lInput[0]
        lLastState = [tupInputState] + [
            deepcopy(lyr.send(None)) for lyr in self.lStreamers
        ]

        # - Initialise layer output variables with initial state, convert to lists
        lLayerOutputs = [
            tuple([np.reshape(x, (1, -1))] for x in state) for state in lLastState[1:]
        ]

        # - Display some feedback
        if bVerbose:
            print("Network: got initial state:")
            print(lLayerOutputs)

        # - Streaming loop
        lState = deepcopy(lLastState)
        for nStep in range(nNumTimeSteps):
            if bVerbose:
                print("Network: Start of step", nStep)

            # - Set up external input
            lLastState[0] = lInput[nStep]

            # - Loop over layers, stream data in and out
            for nLayerInd in range(nNumLayers):
                # - Display some feedback
                if bVerbose:
                    print("Network: Evolving layer {}".format(nLayerInd))

                # - Try / Catch to handle end of streaming iteration
                try:
                    # - `send` input data for current layer
                    # - wait for the output state for the current layer
                    lState[nLayerInd + 1] = deepcopy(
                        self.lStreamers[nLayerInd].send(lLastState[nLayerInd])
                    )

                except StopIteration as e:
                    # - StopIteration returns the final state
                    lState[nLayerInd + 1] = e.args[0]

            # - Collate layer outputs
            for nLayer in range(nNumLayers):
                for nTupleIndex in range(len(lLayerOutputs[nLayer])):
                    lLayerOutputs[nLayer][nTupleIndex].append(
                        np.reshape(lState[nLayer + 1][nTupleIndex], (1, -1))
                    )

            # - Save last state to use as input for next step
            lLastState = deepcopy(lState)

            # - Call callback function
            if fhStepCallback is not None:
                fhStepCallback(self)

        # - Build return dictionary
        dtsSignal = {"external": tsInput.copy()}
        for nLayer in range(nNumLayers):
            # - Concatenate time series
            lvData = [
                np.stack(np.array(data, "float")) for data in lLayerOutputs[nLayer]
            ]

            # - Filter out nans in time trace (always first data element)
            vbUseSamples = ~np.isnan(lvData[0]).flatten()
            tupData = tuple(data[vbUseSamples, :] for data in lvData)

            if bVerbose:
                print(tupData[0])

            # - Build output dictionary (using appropriate output class)
            dtsSignal[self.lEvolOrder[nLayer].strName] = self.lEvolOrder[
                nLayer
            ].cOutput(*tupData)

            # - Set name for time series, if not already set
            if dtsSignal[self.lEvolOrder[nLayer].strName].name is None:
                dtsSignal[self.lEvolOrder[nLayer].strName].name = self.lEvolOrder[
                    nLayer
                ].strName

        # - Increment time
        self._nTimeStep += nNumTimeSteps

        # - Return collated signals
        return dtsSignal

    def _check_sync(self, bVerbose: bool = True) -> bool:
        """
        _check_sync - Check whether the time t of all layers matches self.t
                     If not, throw an exception.
        """
        bSync = True
        if bVerbose:
            print("Network: Network time is {}. \n\t Layer times:".format(self.t))
            print(
                "\n".join(
                    (
                        "\t\t {}: {}".format(lyr.strName, lyr.t)
                        for lyr in self.lEvolOrder
                    )
                )
            )
        for lyr in self.lEvolOrder:
            if lyr._nTimeStep != self._nTimeStep * lyr._nNumTimeStepsPerGlobal:
                bSync = False
                print(
                    "\t Network: WARNING: Layer `{}` is not in sync (t={})".format(
                        lyr.strName, lyr.t
                    )
                )
        if bSync:
            if bVerbose:
                print("\t Network: All layers are in sync with network.")
        else:
            raise NetworkError("Network: Not all layers are in sync with the network.")
        return bSync

    def reset_time(self):
        """
        reset_time() - Reset the time of the network to zero. Does not reset state.
        """
        # - Reset time for each layer
        for lyr in self.setLayers:
            lyr.reset_time()

        # - Reset global network time
        self._nTimeStep = 0

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
        self._nTimeStep = 0

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
        return (
            0
            if not hasattr(self, "_tDt") or self._tDt is None
            else self._tDt * self._nTimeStep
        )

    @property
    def tDt(self):
        return self._tDt

    # @fDt.setter
    # def fDt(self, fNewDt):
    #     self.__fDt = fNewDt
    #     for lyr in self.setLayers:
    #         lyr.fDt = self.__fDt

    def save(self, filename):
        listofLayers = []
        for lyr in self.lEvolOrder:
            listofLayers.append(lyr.to_dict())
        with open(filename, "w") as f:
            json.dump(listofLayers, f)

    @staticmethod
    def load(filename):
        """
        load the network from a json file
        :param filename: filename of json that contains
        :return: returns a network object with all the layers
        """

        with open(filename, "r") as f:
            listofLayers = json.load(f)
        lEvolOrder = []
        for lyr in listofLayers:
            classLyr = getattr(layers, lyr["ClassName"])
            lEvolOrder.append(classLyr.load_from_dict(lyr))
            print(lEvolOrder[-1].strName)
        inp2out = False
        if lEvolOrder[-1].mfW.shape[0] != lEvolOrder[-2].mfW.shape[1]:
            inp2out = True
        return Network(*lEvolOrder, inp2out=inp2out)



### --- NetworkError exception class
class NetworkError(Exception):
    pass
