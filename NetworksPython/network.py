###
# network.py - Code for encapsulating networks
###

### --- Imports
import numpy as np
# from math import gcd
# from functools import reduce
from typing import Callable

# from layers import feedforward, recurrent
from TimeSeries import TimeSeries
from layers.layer import Layer


# - Configure exports
__all__ = ['Network']

# - Relative tolerance for float comparions
fTolerance = 1e-5

### --- Helper functions

def isMultiple(a: float, b: float, fTolerance: float = fTolerance) -> bool:
    """
    isMultiple - Check whether a%b is 0 within some tolerance.
    :param a: float The number that may be multiple of b
    :param b: float The number a may be a multiple of
    :param fTolerance: float Relative tolerance
    :return bool: True if a is a multiple of b within some tolerance
    """
    fMinRemainder = min(a%b, b-a%b)
    return fMinRemainder < fTolerance*b


### --- Network class

class Network:
    def __init__(self,
                 lyrInput: Layer = None,
                 lyrRes: Layer = None,
                 lyrReadout: Layer = None):
        """
        Network - Super class to encapsulate several Layers, manage signal routing

        :param lyrInput:   Layer Input layer (recieves network-external input)
        :param lyrRes:     Layer Internal layer (usually a recurrent reservoir)
        :param lyrReadout:  Layer Output layer (provides network-external output)
        """

        # - Network time
        self._t = 0

        # Maintain set of all layers
        self.setLayers = set()

        # - Add layers
        if lyrInput is not None:
            self.lyrInput = self.add_layer(lyrInput, bExternalInput=True)

        if lyrInput is not None and lyrRes is not None:
            self.lyrRes = self.add_layer(lyrRes, lyrInput =self.lyrInput)

        if lyrRes is not None and lyrReadout is not None:
            self.lyrReadout = self.add_layer(lyrReadout, lyrInput =self.lyrRes)

    def add_layer(self,
                  lyr: Layer,
                  lyrInput: Layer = None,
                  lyrOutput: Layer = None,
                  bExternalInput: bool = False) -> Layer:
        """Add lyr to self and to self.setLayers. Its attribute name
        is 'lyr'+lyr.strName. Check whether layer with this name
        already exists (replace anyway).
        Connect lyr to lyrInput and lyrOutput.
        Return lyr.
            lyr : layer to be added to self
            lyrInput : input layer to lyr
            lyrOutput : layer to which lyr is input layer
        """

        # - Check whether layer time matches network time
        assert lyr.t == self.t, ('Layer time must match network time '
            +'(network: t={}, layer: t={})'.format(self.t, lyr.t))

        # - Check whether self already contains a layer with the same name as lyr.
        if hasattr(self, lyr.strName):
            # - Check if layers are the same object.
            if getattr(self, lyr.strName) is lyr:
                print('This layer is already part of the network')
                return lyr
            else:
                sNewName = lyr.strName
                # - Find a new name for lyr.
                while hasattr(self, sNewName):
                    sNewName = self._new_name(sNewName)
                print('A layer with name `{}` already exists.'.format(lyr.strName)
                          + 'The new layer will be renamed to  `{}`.'.format(sNewName))
                lyr.strName = sNewName

        # - Add set of input layers and flag to determine if lyr receives external input
        lyr.lyrIn = None
        lyr.bExternalInput = bExternalInput

        # - Add lyr to the network
        setattr(self, lyr.strName, lyr)
        print('Added layer `{}` to network\n'.format(lyr.strName))

        # - Update inventory of layers
        self.setLayers.add(lyr)

        # - Connect in- and outputs
        if lyrInput is not None:
            self.connect(lyrInput, lyr)
        if lyrOutput is not None:
            self.connect(lyr, lyrOutput)

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
        lsSplitted = strName.split('_')
        if len(lsSplitted) > 1:
            try:
                # - If the part after the last '_' is an integer, raise it by 1
                i = int(lsSplitted[-1])
                lsSplitted[-1] = str(i+1)
                sNewName = '_'.join(lsSplitted)
            except ValueError:
                sNewName = strName + '_0'
        else:
            sNewName = strName + '_0'

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

    def connect(self, lyrSource: Layer, lyrTarget: Layer):
        """
        connect: Connect two layers by defining one as the input layer
                 of the other.
        :param lyrSource:   The source layer
        :param lyrTarget:   The target layer
        """

        # - Make sure that layer dimensions match
        if lyrSource.nSize != lyrTarget.nDimIn:
            raise NetworkError('Dimensions of layers `{}` (nSize={}) and `{}`'.format(
                               lyrSource.strName, lyrSource.nSize, lyrTarget.strName)
                              +' (nDimIn={}) do not match'.format(lyrTarget.nDimIn))

        # - Check for compatible input / output
        if lyrSource.cOutput != lyrTarget.cInput:
            raise NetworkError('Input / output classes of layer `{}` (cOutput = {})'.format(
                                    lyrSource.strName, lyrSource.cOutput.__name__) +
                               ' and `{}` (cInput = {}) do not match'.format(
                                    lyrTarget.strName, lyrTarget.cInput.__name__))

        # - Add source layer to target's set of inputs
        lyrTarget.lyrIn = lyrSource

        # - Make sure that the network remains a directed acyclic graph
        #   and reevaluate evolution order
        try:
            self.lEvolOrder = self._evolution_order()
            print('Layer `{}` now receives input from layer `{}` \n'.format(
                  lyrTarget.strName, lyrSource.strName)) #,
                  # 'and new layer evolution order has been set.')
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
        if lyrTarget.lyrIn  is lyrSource:
            # - Remove the connection
            lyrTarget.lyrIn = None
            print('Layer {} no longer receives input from layer `{}`'.format(
                  lyrTarget.strName, lyrSource.strName))

            # - Reevaluate evolution order
            try:
                self.lEvolOrder = self._evolution_order()
            except NetworkError as e:
                raise e

        else:
            print('There is no connection from layer `{}` to layer `{}`'.format(
                  lyrSource.strName, lyrTarget.strName))

    def _evolution_order(self) -> list:
        """
        Determine the order in which layers are evolved. Requires Network
        to be a directed acyclic graph, otherwise evolution has to happen
        timestep-wise instead of layer-wise
        """
        def next_layer(setCandidates: set) -> Layer:
            while True:
                try:
                    lyrCandidate = setCandidates.pop()
                # If no candidate is left, raise an exception
                except KeyError:
                    raise NetworkError('Cannot resolve evolution order of layers')
                    # Could implement timestep-wise evolution...
                else:
                    # - If there is a candidate and none of the remaining layers
                    #   is its input layer, this will be the next
                    if not (lyrCandidate.lyrIn in setlyrRemaining):
                        return lyrCandidate

        # - Set of layers that are not in evolution order yet
        setlyrRemaining = self.setLayers.copy()
        # - Begin with input layer
        lOrder = [self.lyrInput]
        setlyrRemaining.remove(self.lyrInput)
        while bool(setlyrRemaining):
            # - Find the next layer to be evolved
            lyrNext = next_layer(setlyrRemaining.copy())
            lOrder.append(lyrNext)
            setlyrRemaining.remove(lyrNext)

        # - Return a list with the layers in their evolution order
        return lOrder

    def evolve(self,
               tsExternalInput: TimeSeries = None,
               tDuration: float = None) -> dict:
        """
        evolve - Evolve each layer in the network according to self.lEvolOrder.
                 For layers with bExternalInput==True their input is
                 tsExternalInput. If not but an input layer is defined, it
                 will be the output of that, otherwise None.
                 Return a dict with each layer's output.
        :param tsExternalInput:  TimeSeries with external input data.
        :param tDuration:        Float - duration over which netŵork should
                                         be evolved. If None, evolution is
                                         over the duration of tsExternalInput
        :return:                 Dict with each layer's output time Series
        """

        # - Determine default duration
        if tDuration is None:
            assert tsExternalInput is not None, (
                'One of `tsExternalInput` or `tDuration` must be supplied')

            if tsExternalInput.bPeriodic:
                # - Use duration of periodic TimeSeries, if possible
                tDuration = tsExternalInput.tDuration

            else:
                # - Evolve until the end of the input TimeSeries
                tDuration = tsExternalInput.tStop - self.t
                assert tDuration > 0, (
                    'Cannot determine an appropriate evolution duration. '
                   +'`tsExternalInput` finishes before the current evolution time.')

        # - List of layers where tDuration is not a multiple of tDt
        llyrDtMismatch = list(filter(lambda lyr: not isMultiple(tDuration, lyr.tDt), self.lEvolOrder))

        # - Throw an exception if llyrDtMismatch is not empty, showing for
        #   which layers there is a mismatch
        if llyrDtMismatch:
            strLayers = ', '.join(('{}: tDt={}'.format(lyr.strName, lyr.tDt)
                                   for lyr in llyrDtMismatch))
            raise ValueError('`tDuration` is not a multiple of `tDt` for the following layer(s):\n'
                             + strLayers)

        # - Dict to store external input and each layer's output time series
        dtsSignal = {'external' : tsExternalInput}

        # - Make sure layers are in sync with netowrk
        self._check_sync()

        # - Iterate over evolution order and evolve layers
        for lyr in self.lEvolOrder:

            # - Determine input for current layer
            if lyr.bExternalInput:
                # External input
                tsCurrentInput = tsExternalInput
                strIn = 'external input'
            elif lyr.lyrIn is not None:
                # Output of current layer's input layer
                tsCurrentInput = dtsSignal[lyr.lyrIn.strName]
                strIn = lyr.lyrIn.strName + "'s output"
            else:
                # No input
                tsCurrentInput = None
                strIn = 'nothing'

            print('Evolving layer `{}` with {} as input'.format(lyr.strName, strIn))
            # Evolve layer and store output in dtsSignal
            dtsSignal[lyr.strName] = lyr.evolve(tsCurrentInput, tDuration)

        # - Update network time
        self._t += tDuration

        # - Make sure layers are still in sync with netowrk
        self._check_sync()

        # - Return dict with layer outputs
        return dtsSignal

    def train(self,
              fhTraining: Callable,
              tsExternalInput: TimeSeries = None,
              tDuration: float = None,
              tDurBatch: float = None):
        """
        train - Train the network batch-wise by evolving the layers and
                calling fhTraining.
        :param fhTraining:      Function that is called after each evolution
        :param tsExternalInput: TimeSeries with external input to network
        :param tDuration:       float - Duration over which netŵork should
                                        be evolved. If None, evolution is
                                        over the duration of tsExternalInput
        :param tDurBatch:       float - Duration of one batch
        """

        # - Determine duration of training
        if tDuration is None:
            assert tsExternalInput is not None, (
                'One of `tsExternalInput` or `tDuration` must be supplied')

            if tsExternalInput.bPeriodic:
                # - Use duration of periodic TimeSeries, if possible
                tRemaining = tsExternalInput.tDuration

            else:
                # - Evolve until the end of the input TimeSeries
                tRemaining = tsExternalInput.tStop - self.t
                assert tRemaining > 0, (
                    'Cannot determine an appropriate evolution duration. '
                   +'`tsExternalInput` finishes before the current evolution time.')

        # - Determine batch duration and number
        tDurBatch = tRemaining if tDurBatch is None else tDurBatch
        nNumBatches = int(np.ceil(tRemaining/tDurBatch))

        # - Iterate over batches
        bFirst = True
        bFinal = False
        for nBatch in range(1, nNumBatches+1):
            print('\nTraining batch {} of {}'.format(nBatch, nNumBatches))
            # - Evolve network
            dtsSignal = self.evolve(tsExternalInput, min(tDurBatch, tRemaining))
            # - Remaining simulation time
            tRemaining -= tDurBatch
            # - Determine if this batch was the first or the last of training
            if nBatch == nNumBatches: bFinal = True
            # - Call the callback
            fhTraining(self, dtsSignal, bFirst, bFinal)
            bFirst = False           

        print('\nTraining successful')

    def _check_sync(self) -> bool:
        """
        _check_sync - Check whether the time t of all layers matches self.t
                     If not, throw an exception.
        """
        bSync = True
        print('Network time is {}'.format(self.t))
        for lyr in self.lEvolOrder:
            if lyr.t != self.t:
                bSync = False
                print('\t WARNING: Layer `{}` is not in sync (t={})'.format(lyr.strName, lyr.t))
        if bSync:
            print('\t All layers are in sync with network.')
        else:
            raise NetworkError('Not all layers are in sync with the network.')
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
        self.reset_state()
        self.reset_time()

    def __repr__(self):
        return '{} object with {} layers'.format(self.__class__.__name__, len(self.setLayers))


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


"""Older stuff that might be useful again

# - Asserting that tDuration % self.tDt == 0
if (   min(tDuration%self.tDt, self.tDt-(tDuration%self.tDt))
     > fTolerance * self.tDt):
    raise ValueError('Creation of time trace failed. tDuration ({}) '
                    +'is not a multiple of self.tDt ({})'.format(tDuration, self.tDt))
# - or assert that last value of time series is tSTart+tDuration
# tStop = tStart + tDuration
# if np.abs(vtTimeTrace[-1] - tStop) > fTol*self._tDt:
#     raise ValueError( 'Creation of time trace failed. Make sure that '
#                      +'tDuration ({}) is a multiple of self.tDt ({}).'.format(
#                      tDuration, self.tDt) )


def lcm(*numbers: int) -> int:
        lcm - Return the least common multiple of a series of numbers
    :param numbers: iterable containing integer values
    :return: int The least common multiple of *numbers

    # - The LCM of two numbers is their product divided by their gcd
    def _lcm(x: int, y: int) -> int:
        return x*y//gcd(x,y)

    return reduce(_lcm, numbers, 1)


        # Default parameters for layers
        dParamsIn = {'nSize' : 1,
                     'sKind' : 'pass'}
        dParamsOut = {'nSize' : 1,
                      'sKind' : 'ffrate'}
        dParamsRes = {'nSize' : 256,
                      # 'nReservoirs' : 1,
                      # 'sResConn' : 'serial',
                      'sKind' : 'reservoir'}
        dParamsIn.update(kwInput)
        dParamsRes.update(kwReservoir)
        dParamsOut.update(kwOutput)
        try:
            dParamsRes['vTau_n'] = kwReservoir['vTau_n']
        except KeyError:
            dParamsRes['vTau_n'] = np.random.rand(dParamsRes['nSize'])


        Constructing reservoirs in series or parallel
        # self.nReservoirs = dParamsRes.pop('nReservoirs')
        # sResConn = dParamsRes.pop('sResConn')
        # # Broadcast single paramters to arrays that apply to all reservoirs
        # #   Time constants
        # if (len(np.atleast_2d(dParamsRes['vTau_n'])) == 1
        #       and len(dParamsRes['vTau_n']) != self.nReservoirs):
        #     dParamsRes['vTau_n'] = np.repeat(np.atleast_2d(dParamsRes['vTau_n']),
        #                                      self.nReservoirs, axis=0)
        # #  All other parameters
        # for sParam in dParamsRes.keys():
        #     if sParam != 'vTau_n':
        #         try:
        #             dParamsRes[sParam] = np.full(self.nReservoirs, dParamsRes[sParam])
        #         except ValueError:
        #             raise ValueError('The number of arguments for {} '.format(sParam)
        #                              + '({0}) does not match nReservoirs ({1})'.format(
        #                             len(dParamsRes), self.nReservoirs))
        # for i in range(self.nReservoirs):
        #     setattr(self, 'lyrRes{}'.format(i),
        #             Recurrent.RecLayer(**{s : v[i] for s, v in dParamsRes.items()}))

        # - Generate connections: Each layer except input has a set with
        # references to its input layer

        # if sResConn == 'serial':
        #     self.lyrRes0.setIn = {self.lyrInput}
        #     for i in range(1, self.nReservoirs):
        #         getattr(self, 'lyrRes{}'.format(i)).setIn = {getattr(self, 'lyrRes{}'.format(i-1))}
        #     self.lyrOut.setIn = {getattr(self, 'lyrRes{}'.format(self.nReservoirs-1))}
        # elif sResConn == 'parallel':
        #     for i in Range(self.nReservoirs):
        #         getattr(self, 'lyrRes{}'.format(i)).setIn = {self.lyrInput}
        #     self.lyrOut.setIn = {getattr(self, 'lyrRes{}'.format(i))
        #                        for i in range(self.nReservoirs)}
        # else:
        #     raise NetworkError('Connection type `{}` not understood'.format(sResConn))

        ""Add feedforward or recurrent layer to network and maintain
        set containing all layers.""
        sLyrName = 'lyr'+strName
        if hasattr(self, sLyrName):
            raise NameError('There already exists a layer with this name')

        if sKind == 'ffrate':
            setattr(self, sLyrName, FeedForward.FFRate(strName=strName, fDt=self.__fDt, **kwargs))
            print('Rate-based feedforward layer `{}` has been added to network.'.format(strName))

        elif sKind == 'pass':
            setattr(self, sLyrName, FeedForward.PassThrough(strName=strName, fDt=self.__fDt, **kwargs))
            print('PassThrough layer `{}` has been added to network.'.format(strName))

        elif sKind in ['Reservoir', 'reservoir', 'Recurrent', 'recurrent', 'res', 'rec']:
            setattr(self, sLyrName, Recurrent.RecLayer(strName=strName, fDt=self.__fDt, **kwargs))
            print('Recurrent layer `{}` has been added to network.'.format(strName))

        else:
            raise NetworkError('Layer type not understood')
"""