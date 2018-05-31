import numpy as np
from math import gcd
from functools import reduce

from layers import feedforward, recurrent



def multiple(a: float, b: float, fTolerance: float = 1e-5) -> bool:
    """
    multiple - Check whether a%b is 0 within some tolerance.
    :param a: float The number that may be multiple of b
    :param b: float The number a may be a multiple of
    :param fTolerance: float Relative tolerance
    :return bool: True if a is a multiple of b within some tolerance
    """
    fMinRemainder = min(a%b, b-a%b)
    return fMinRemainder < fTolerance*b

class Network():
    def __init__(self, lyrIn, lyrRes, lyrOut):
        """
        Create Network instance consisting of input layer, output
        layer and an arbitrary number of reservoirs. Create forward
        connections between input, reservoirs and output.
        To overwrite default parameters, pass them inside dict to
        corresponding parameter (e.g. parameters concerning reservoirs
        are passed to kwReservoir).
        """
        
        # self.__fDt = fDt

        # Maintain set of all layers
        self.setLayers = set()
        
        # - Add layers
        self.lyrIn = self.add_layer(lyrIn)
        self.lyrRes = self.add_layer(lyrRes,  tplIn=(self.lyrIn, ))
        self.lyrOut = self.add_layer(lyrOut,  tplIn=(self.lyrRes, ))
               
    def add_layer(self, lyr, tplIn=None, tplOut=None):
        """Add lyr to self and to self.setLayers. Its attribute name
        is 'lyr'+lyr.sName. Check whether layer with this name 
        already exists (replace anyway). 
        Connect lyr to those in setIn and setOut.
        Return lyr.
            lyr : layer to be added to self
            setIn : set of input layers to lyr
            setOut : set of layers to which lyr is input layer
        """
        
        # - Check whether self already contains a layer with the same
        #   name as lyr. If so, check whether they are the same objects.
        #   If they are not, rename lyr.
        if hasattr(self, lyr.sName):
            if getattr(self, lyr.sName) is lyr:
                print('This layer is already part of the network')
                return
            else:
                sNewName = lyr.sName
                while hasattr(self, sNewName):
                    sNewName = self.new_layer_name(sNewName)
                print('A layer with name {} already exists.'.format(lyr.sName)
                          + 'The new layer will be renamed to  {}.'.format(sNewName))
                lyr.sName = sNewName

        setattr(self, lyr.sName, lyr)
        print('Added layer {} to network\n'.format(lyr.sName))
        # - Update inventory of layers
        self.setLayers.add(lyr)

        # - Connect in- and outputs
        if tplIn is not None:
            for lyrIn in tplIn:
                self.connect(lyrIn, lyr)
        if tplOut is not None:
            for lyrOut in tplOut:
                self.connect(lyr, lyrOut)

        return lyr

    def new_layer_name(self, sName):
        lsSplitted = sName.split('_')
        if len(lsSplitted) > 1:
            try:
                i = int(lsSplitted[-1])
                lsSplitted[-1] = str(i+1)
                sNewName = '_'.join(lsSplitted)
            except ValueError:
                sNewName = sName + '_0'
        else:
            sNewName = sName + '_0'

        return sNewName

    def remove_layer(self, lyrDel):
        # Remove connections from lyrDel to others
        for lyr in self.setLayers:
            try:
                lyr.setIn.discard(lyrDel)
            except AttributeError:
                pass
        self.setLayers.remove(lyrDel)
        del lyrDel
        self.lEvolOrder = self.evolution_order()

    def connect(self, lyrSource, lyrTarget):
        try:
            lyrTarget.setIn.add(lyrSource)
        except AttributeError:
            lyrTarget.setIn = {lyrSource}
        try:
            self.lEvolOrder = self.evolution_order()
            print('Layer "{}" now receives input from layer "{}" \n'.format(
                  lyrTarget.sName, lyrSource.sName)) #,
                  # 'and new layer evolution order has been set.')
        except NetworkError as e:
            lyrTarget.setIn.remove(lyrSource)
            raise e 

    def disconnect(self, lyrSource, lyrTarget):
        try:
            lyrTarget.setIn.remove(lyrSource)
            print('Layer {} does no longer receive input from layer "{}"'.format(
                  lyrTarget.sName, lyrSource.sName))
        except KeyError:
            print('There is no connection from layer "{}" to layer "{}"'.format(
                  lyrSource.sName, lyrTarget.sName))

    def evolve(self, tsInput, tDuration):
        llyrProblematic = list(filter(lambda lyr: not multiple(tDuration, lyr.tDt), self.lEvolOrder))
        if llyrProblematic != []:
            strLayers = ', '.join(('{}: tDt={}'.format(lyr.sName, lyr.tDt)
                                   for lyr in llyrProblematic))
            raise ValueError('tDuration is not a multiple of tDt for the following layer(s):\n'
                             + strLayers)

        for lyr in self.lEvolOrder:
            print('Evolving layer "{}"'.format(lyr.sName))
            lyr.evolve(tsInput, tDuration)

    def evolution_order(self):
        """
        Determine the order in which layers are evolved. Requires Network
        to be a directed acyclic graph, otherwise evolution has to happen
        timestep-wise instead of layer-wise
        """
        def next_layer(setCandidates):
            while True:
                try:
                    lyrCandidate = setCandidates.pop()
                # If no candidate is left, raise an exception
                except KeyError:
                    raise NetworkError('Cannot resolve evolution order of layers')
                    # Could implement timestep-wise evolution...
                else:
                    # If none of the remaining layers is in the input of the
                    #   candidate set, this will be the next
                    if (not hasattr(lyrCandidate, 'setIn')
                        or len(lyrCandidate.setIn & setlyrRemaining) == 0):
                        return lyrCandidate

        setlyrRemaining = self.setLayers.copy()
        lOrder = [self.lyrIn]
        setlyrRemaining.remove(self.lyrIn)
        while len(setlyrRemaining) > 0:
            lyrNext = next_layer(setlyrRemaining.copy())
            lOrder.append(lyrNext)
            setlyrRemaining.remove(lyrNext)
        return lOrder

    # @property
    # def fDt(self):
    #     return self.__fDt

    # @fDt.setter
    # def fDt(self, fNewDt):
    #     self.__fDt = fNewDt
    #     for lyr in self.setLayers:
    #         lyr.fDt = self.__fDt

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
        #     self.lyrRes0.setIn = {self.lyrIn}
        #     for i in range(1, self.nReservoirs):
        #         getattr(self, 'lyrRes{}'.format(i)).setIn = {getattr(self, 'lyrRes{}'.format(i-1))}
        #     self.lyrOut.setIn = {getattr(self, 'lyrRes{}'.format(self.nReservoirs-1))}
        # elif sResConn == 'parallel':
        #     for i in Range(self.nReservoirs):
        #         getattr(self, 'lyrRes{}'.format(i)).setIn = {self.lyrIn}
        #     self.lyrOut.setIn = {getattr(self, 'lyrRes{}'.format(i))
        #                        for i in range(self.nReservoirs)}
        # else:
        #     raise NetworkError('Connection type "{}" not understood'.format(sResConn))

        ""Add feedforward or recurrent layer to network and maintain 
        set containing all layers.""
        sLyrName = 'lyr'+sName
        if hasattr(self, sLyrName):
            raise NameError('There already exists a layer with this name')

        if sKind == 'ffrate':
            setattr(self, sLyrName, FeedForward.FFRate(sName=sName, fDt=self.__fDt, **kwargs))
            print('Rate-based feedforward layer "{}" has been added to network.'.format(sName))
        
        elif sKind == 'pass':
            setattr(self, sLyrName, FeedForward.PassThrough(sName=sName, fDt=self.__fDt, **kwargs))
            print('PassThrough layer "{}" has been added to network.'.format(sName))
        
        elif sKind in ['Reservoir', 'reservoir', 'Recurrent', 'recurrent', 'res', 'rec']:
            setattr(self, sLyrName, Recurrent.RecLayer(sName=sName, fDt=self.__fDt, **kwargs))
            print('Recurrent layer "{}" has been added to network.'.format(sName))

        else:
            raise NetworkError('Layer type not understood')
"""