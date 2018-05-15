import numpy as np

from layers import FeedForward, Recurrent

class Network():
    def __init__(self, fDt=1, kwInput={}, kwReservoir={}, kwOutput={}):
        """
        Create Network instance consisting of input layer, output
        layer and an arbitrary number of reservoirs. Create forward
        connections between input, reservoirs and output.
        To overwrite default parameters, pass them inside dict to
        corresponding parameter (e.g. parameters concerning reservoirs
        are passed to kwReservoir).
        """
        
        self.__fDt = fDt

        # Default parameters for layers
        dParamsIn = {'nSize' : 1,
                     'bSpiking' : True}
        dParamsOut = {'nSize' : 1}
        dParamsRes = {'nSize' : 256,
                      # 'nReservoirs' : 1,
                      # 'sResConn' : 'serial',
                      'bSpiking' : True}
        dParamsIn.update(kwInput)
        dParamsRes.update(kwReservoir)
        dParamsOut.update(kwOutput)
        try:
            dParamsRes['vTau_n'] = kwReservoir['vTau_n']  
        except KeyError:
            dParamsRes['vTau_n'] = np.random.rand(dParamsRes['nSize'])

        # Maintain set of all layers
        self.setLayers = set()
        
        # - Generate layers
        self.add_layer('ffinput', 'In', **dParamsIn)
        self.add_layer('feedforward', 'Out', **dParamsOut)
        self.add_layer('reservoir', 'Res', **dParamsRes)

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
        self.connect(source=self.lyrIn, target=self.lyrRes)
        self.connect(source=self.lyrRes, target=self.lyrOut)
        
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
    
    def add_layer(self, sKind, sName, **kwargs):
        """Add feedforward or recurrent layer to network and maintain 
        set containing all layers."""
        sLyrName = 'lyr'+sName
        if hasattr(self, sLyrName):
            raise NameError('There already exists a layer with this name')
        if sKind in ['FeedForward', 'Feedforward', 'feedforward', 'ff']:
            setattr(self, sLyrName, FeedForward.FFLayer(sName=sName, fDt=self.__fDt, **kwargs))
            print('Feedforward layer "{}" has been added to network.'.format(sName))
        if sKind == 'ffinput':
            setattr(self, sLyrName, FeedForward.FFInput(sName=sName, fDt=self.__fDt, **kwargs))
            print('FFInput layer "{}" has been added to network.'.format(sName))
        elif sKind in ['Reservoir', 'reservoir', 'Recurrent', 'recurrent', 'res', 'rec']:
            setattr(self, sLyrName, Recurrent.RecLayer(sName=sName, fDt=self.__fDt, **kwargs))
            print('Recurrent layer "{}" has been added to network.'.format(sName))
        self.setLayers.add(getattr(self, sLyrName))

    def connect(self, source, target):
        try:
            target.setIn.add(source)
        except AttributeError:
            target.setIn = {source}
        try:
            self.lEvolOrder = self.evolution_order()
            print('Layer "{}" now receives input from layer "{}" '.format(
                  target.sName, source.sName),
                  'and new layer evolution order has been set.')
        except NetworkError as e:
            target.setIn.remove(source)
            raise e 

    def disconnect(self, source, target):
        try:
            target.setIn.remove(source)
            print('Layer {} does no longer receive input from layer "{}"'.format(
                  target.sName, source.sName))
        except KeyError:
            print('There is no connection from layer "{}" to layer "{}"'.format(
                  source.sName, target.sName))

    def evolve(self, tTime, mInput):
        for lyr in self.lEvolOrder:
            print('Evolving layer "{}"'.format(lyr.sName))
            lyr.evolve(tTime, mInput)

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

    @property
    def fDt(self):
        return self.__fDt

    @fDt.setter
    def fDt(self, fNewDt):
        self.__fDt = fNewDt
        for lyr in self.setLayers:
            lyr.fDt = self.__fDt

class NetworkError(Exception):
    pass