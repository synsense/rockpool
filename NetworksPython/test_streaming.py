import numpy as np
from layers import Layer
from network import Network
from TimeSeries import *
from copy import deepcopy

import holoviews as hv
hv.extension('bokeh')


class TestLayer(Layer):
    def evolve(self):
        pass

    def stream(self, tDuration, tDt, bVerbose = False):
        # - Initialise simulation, determine how many tDt to evolve fot
        if bVerbose: print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, tDuration, tDt)
        nNumSteps = np.size(vtTimeTrace)
        mfNoise = np.random.rand(nNumSteps) * self.fNoiseStd

        nStep = 0
        vfInput = None
        self.state = np.zeros(self.nSize, 'float')
        if bVerbose: print("Layer: Prepared")

        # - Loop over tDt steps
        for nStep in range(nNumSteps):
            # - Convert from current state to output
            if bVerbose: print('Layer: Yielding state: ', self.state)
            if bVerbose: print('Layer: step', nStep)
            if bVerbose: print('Layer: Waiting for input...')

            # - Yield current output, receive input for next time step
            tInput = yield (self._t, self.state)

            if bVerbose: print('Layer: Input was: ', tInput)

            # - Process input, if supplied
            if tInput is not None:
                mfSamples = np.atleast_2d(tInput[1])
                self.state += mfSamples[0, :]

            # - Incorporate noise trace
            self.state += mfNoise[nStep]

            # - Increment time
            self._t += tDt

        # - Retrun final state
        return (self._t, self.state)


class TestSpikeLayer(Layer):
    def evolve(self):
        pass

    def stream(self, tDuration, tDt, bVerbose = False):
        # - Initialise simulation, determine how many tDt to evolve for
        if bVerbose: print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, tDuration, tDt)
        nNumSteps = np.size(vtTimeTrace)
        mfNoise = np.random.rand(nNumSteps)

        nStep = 0
        vfInput = None
        self.state = 0
        if bVerbose: print("Layer: Prepared")

        for nStep in range(nNumSteps):
            # - Convert from current state to output
            nNumSpikes = np.random.randint(0, 2)
            if nNumSpikes > 0:
                vnChannels = np.random.randint(0, 10, nNumSpikes)
                vtSpikeTimes = np.repeat(self._t, nNumSpikes)
                vfSamples = None
            else:
                vnChannels = None
                vtSpikeTimes = None
                vfSamples = None

            if bVerbose: print('Layer: Yielding {} spikes: '.format(nNumSpikes))
            if bVerbose: print('Layer: step', nStep)

            if bVerbose: print('Layer: Waiting for input spikes...')

            # - Yield output for current time step, receive input for next time step
            tInput = yield (vtSpikeTimes, vnChannels, vfSamples)

            if bVerbose: print('Layer: Input was: ', tInput)

            # - Process input, if supplied
            if tInput is not None:
                self.state += np.size(tInput[0])

            # - Incorporate noise trace
            self.state += mfNoise[nStep]

            # - Increment time
            self._t += tDt

        # - Return final state
        return (self._t, self.state, None)

    @property
    def cInput(self):
        return TSEvent

    @property
    def cOutput(self):
        return TSEvent


class TestNetwork(Network):
    def stream(self, tsExtInput, tDuration, bVerbose = False):
        # - Find the largest common tDt
        self.ltDts = [lyr.tDt for lyr in self.setLayers]
        self.tCommonDt = max(self.ltDts)
        if bVerbose: print('Net: Common time base: ', self.tCommonDt)

        # - Prepare time base
        vtTimeBase = np.arange(0, tDuration + self.tCommonDt, self.tCommonDt) + self._t
        vtTimeBase = vtTimeBase[vtTimeBase <= self._t + tDuration]
        tDuration = vtTimeBase[-1] - vtTimeBase[0]
        nNumSteps = np.size(vtTimeBase)

        # - Prepare all layers
        self.lStreamers = [lyr.stream(tDuration, self.tCommonDt, bVerbose = bVerbose)
                           for lyr in self.lEvolOrder]
        nNumLayers = np.size(self.lEvolOrder)

        # - Prepare external input
        if tsExtInput is not None:
            lInput = [tsExtInput.find((t, t + self.tCommonDt))
                      for t in vtTimeBase]
        else:
            lInput = [None] * nNumSteps

        # - Get initial state of all layers
        if bVerbose: print('Net: getting initial state')

        # - Determine input state size, obtain initial layer state
        tInputState = lInput[0]
        vtLastState = [tInputState] + [deepcopy(lyr.send(None)) for lyr in self.lStreamers]

        # - Initialise layer output variables with initial state, convert to lists
        tLayerOutputs = tuple(tuple([x] for x in state) for state in vtLastState[1:])

        if bVerbose:
            print('Net: got initial state:')
            print(tLayerOutputs)

        # - Streaming loop
        vtState = deepcopy(vtLastState)
        for nStep in range(nNumSteps - 1):
            if bVerbose: print('Net: Start of step', nStep)

            # - Set up external input
            vtLastState[0] = (lInput[nStep])

            for nLayerInd in range(nNumLayers):
                if bVerbose: print('Net: Evolving layer {}'.format(nLayerInd))
                try:
                    vtState[nLayerInd + 1] = deepcopy(self.lStreamers[nLayerInd].send(vtLastState[nLayerInd]))
                except StopIteration as e:
                    vtState[nLayerInd + 1] = e.args[0]

            # - Save layer outputs
            for nLayer in range(nNumLayers):
                for nTupleIndex in range(len(tLayerOutputs[nLayer])):
                    tLayerOutputs[nLayer][nTupleIndex].append(vtState[nLayer + 1][nTupleIndex])

            # - Save last state to use as input for next step
            vtLastState = deepcopy(vtState)

        # - Build return dictionary
        dReturn = dict()
        for nLayer in range(nNumLayers):
            # - Concatenate time series
            lvoData = [np.stack(np.array(data, 'float')) for data in tLayerOutputs[nLayer]]

            # - Filter out nans in time trace (always first data element)
            vbUseSamples = ~np.isnan(lvoData[0])
            tvoData = tuple(data[vbUseSamples] for data in lvoData)

            if bVerbose: print(tvoData[0])

            # - Build dictionary
            dReturn[self.lEvolOrder[nLayer].strName] = TSEvent(*tvoData)

        # - Increment time
        self._t += tDuration

        return dReturn

if __name__ == '__main__':
    lyr1 = TestLayer(0)
    lyr2 = TestLayer(0)
    lyr3 = TestLayer(0)
    net = TestNetwork(lyr1, lyr2, lyr3)

    # - Linear input
    tsIn = TSContinuous(np.arange(0, 11, 1), np.linspace(0, 1, 11).reshape(-1, 1)) + .5

    dOut = net.stream(tsIn, 10, bVerbose = True)

    print(dOut)

    #
    # dOut = net.stream(None, 10, bVerbose = True)
    #
    # dOut['unnamed'].plot() + dOut['unnamed_0'].plot() + dOut['unnamed_1'].plot()
    #
    # lyr1 = TestSpikeLayer(0)
    # lyr2 = TestSpikeLayer(0)
    # lyr3 = TestSpikeLayer(0)
    # net = TestNetwork(lyr1, lyr2, lyr3)
    #
    # tsSpikes = TSEvent([0, 1, 2, 3, 4]).delay(.1)
    #
    # dOutSpikes = net.stream(tsSpikes, 5, bVerbose = True)