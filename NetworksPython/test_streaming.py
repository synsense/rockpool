import numpy as np
from layers import Layer
from network import Network
from TimeSeries import *

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
        self.state = 0
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
                self.state += tInput[1]

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
    def stream(self, tsExtInput, tDuration):
        # - Find the largest common tDt
        self.ltDts = [lyr.tDt for lyr in self.setLayers]
        self.tCommonDt = max(self.ltDts)
        print('Net: Common time base: ', self.tCommonDt)

        # - Prepare time base
        vtTimeBase = np.arange(0, tDuration + self.tCommonDt, self.tCommonDt) + self._t
        vtTimeBase = vtTimeBase[vtTimeBase <= self._t + tDuration]
        tDuration = vtTimeBase[-1] - vtTimeBase[0]
        nNumSteps = np.size(vtTimeBase)

        # - Prepare all layers
        self.lStreamers = [lyr.stream(tDuration, self.tCommonDt) for lyr in self.lEvolOrder]
        nNumLayers = np.size(self.lEvolOrder)

        # - Prepare external input
        if tsExtInput is not None:
            mfExtInputStep = tsExtInput(vtTimeBase)
        else:
            mfExtInputStep = np.zeros((np.size(vtTimeBase), self.lEvolOrder[0].nDimIn))

            # - Prepare output storage
        dLayerOutputs = {nLayer: {'vtTime':   np.zeros(nNumSteps),
                                  'mfOutput': np.zeros((nNumSteps, self.lEvolOrder[nLayer].nSize)),
                                  'nPointer': 1,
                                  } for nLayer in range(nNumLayers)}

        # - Get initial state of all layers
        print('Net: getting initial state')
        vtLastState = [(0, 0)] + [lyr.send(None) for lyr in self.lStreamers]

        # - Save initial state
        for nLayer in range(nNumLayers):
            # - Get initial state from layers
            (dLayerOutputs[nLayer]['vtTime'][0],
             dLayerOutputs[nLayer]['mfOutput'][0],
             ) = vtLastState[nLayer]

        print('Net: got initial state')

        # - Streaming loop
        vtState = vtLastState.copy()
        nOutputPointer = 0
        for nStep in range(nNumSteps - 1):
            print('Net: Start of step', nStep)

            # - Set up external input
            vtLastState[0] = (vtTimeBase[nStep], mfExtInputStep[nStep, :])

            for nLayerInd in range(nNumLayers):
                try:
                    vtState[nLayerInd + 1] = self.lStreamers[nLayerInd].send(vtLastState[nLayerInd])
                except StopIteration as e:
                    vtState[nLayerInd + 1] = e.args[0]

            # - Save layer outputs
            for nLayer in range(nNumLayers):
                nOutputPointer = dLayerOutputs[nLayer]['nPointer']
                (dLayerOutputs[nLayer]['vtTime'][nOutputPointer],
                 dLayerOutputs[nLayer]['mfOutput'][nOutputPointer],
                 ) = vtState[nLayer + 1]

                # - Increment output pointer
                dLayerOutputs[nLayer]['nPointer'] += 1

            # - Save last state to use as input for next step
            vtLastState = vtState.copy()

        return {self.lEvolOrder[nLayer].strName: TSContinuous(
                dLayerOutputs[nLayer]['vtTime'][:dLayerOutputs[nLayer]['nPointer']],
                dLayerOutputs[nLayer]['mfOutput'][:dLayerOutputs[nLayer]['nPointer']])
                for nLayer in range(nNumLayers)
                }


class TestSpikeNetwork(Network):
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
        self.lStreamers = [lyr.stream(tDuration, self.tCommonDt) for lyr in self.lEvolOrder]
        nNumLayers = np.size(self.lEvolOrder)

        # - Prepare external input
        if tsExtInput is not None:
            lInputSpikes = [tsExtInput.find((t, t + self.tCommonDt))
                            for t in vtTimeBase]
        else:
            lInputSpikes = [None] * nNumSteps

        # - Get initial state of all layers
        if bVerbose: print('Net: getting initial state')

        vtLastState = [(0, 0, 0)] + [lyr.send(None) for lyr in self.lStreamers]

        # - Get initial state from layers
        ldLayerOutputs = [{'vtTime':     [vtLastState[nLayer + 1][0]],
                           'vnChannels': [vtLastState[nLayer + 1][1]],
                           'mfSamples':  [vtLastState[nLayer + 1][2]]}
                          for nLayer in range(nNumLayers)]

        if bVerbose: print('Net: got initial state')

        # - Streaming loop
        vtState = vtLastState.copy()
        nOutputPointer = 0
        for nStep in range(nNumSteps - 1):
            if bVerbose: print('Net: Start of step', nStep)

            # - Set up external input
            vtLastState[0] = (lInputSpikes[nStep])

            for nLayerInd in range(nNumLayers):
                try:
                    vtState[nLayerInd + 1] = self.lStreamers[nLayerInd].send(vtLastState[nLayerInd])
                except StopIteration as e:
                    vtState[nLayerInd + 1] = e.args[0]

            # - Save layer outputs
            for nLayer in range(nNumLayers):
                ldLayerOutputs[nLayer]['vtTime'].append(vtState[nLayer + 1][0])
                ldLayerOutputs[nLayer]['vnChannels'].append(vtState[nLayer + 1][1])
                ldLayerOutputs[nLayer]['mfSamples'].append(vtState[nLayer + 1][2])

            # - Save last state to use as input for next step
            vtLastState = vtState.copy()

        # - Build return dictionary
        dReturn = dict()
        for nLayer in range(nNumLayers):
            # - Concatenate time series
            vtTimeTrace = np.array(ldLayerOutputs[nLayer]['vtTime'], 'float')
            vnChannels = np.array(ldLayerOutputs[nLayer]['vnChannels'], 'float')
            mfSamples = np.array(ldLayerOutputs[nLayer]['mfSamples'], 'float')

            # - Filter out nans in time trace
            vbUseSamples = np.invert(np.isnan(vtTimeTrace))

            if bVerbose: print(vtTimeTrace[vbUseSamples])

            # - Build dictionary
            dReturn[self.lEvolOrder[nLayer].strName] = TSEvent(vtTimeTrace[vbUseSamples],
                                                               vnChannels[vbUseSamples],
                                                               mfSamples[vbUseSamples]
                                                               )

        return dReturn

if __name__ == '__main__':
    lyr1 = TestLayer(0)
    lyr2 = TestLayer(0)
    lyr3 = TestLayer(0)
    net = TestNetwork(lyr1, lyr2, lyr3)

    dOut = net.stream(None, 10)

    dOut['unnamed'].plot() + dOut['unnamed_0'].plot() + dOut['unnamed_1'].plot()

    lyr1 = TestSpikeLayer(0)
    lyr2 = TestSpikeLayer(0)
    lyr3 = TestSpikeLayer(0)
    net = TestSpikeNetwork(lyr1, lyr2, lyr3)

    tsSpikes = TSEvent([0, 1, 2, 3, 4]).delay(.1)

    dOutSpikes = net.stream(tsSpikes, 5)