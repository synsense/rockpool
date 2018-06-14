import numpy as np
from layers import Layer
from network import Network
from TimeSeries import *

import holoviews as hv
hv.extension('bokeh')


class TestLayer(Layer):
    def evolve(self):
        pass

    def stream(self, tDuration, tDt):
        print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, tDuration, tDt)
        nNumSteps = np.size(vtTimeTrace)
        mfNoise = np.random.rand(nNumSteps)

        nStep = 0
        vfInput = None
        self.state = 0
        print("Layer: Prepared")
        for nStep in range(nNumSteps):
            print('Layer: Yielding state: ', self.state)
            print('Layer: step', nStep)
            print('Layer: Waiting for input...')
            tInput = yield (self._t, self.state)

            print('Layer: Input was: ', tInput)

            if tInput is not None:
                self.state += tInput[1]

            self.state += mfNoise[nStep]
            self._t += tDt

        return (self._t, self.state)


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


if __name__ == '__main__':
    lyr1 = TestLayer(0)
    lyr2 = TestLayer(0)
    lyr3 = TestLayer(0)
    net = TestNetwork(lyr1, lyr2, lyr3)

    dOut = net.stream(None, 10)

    dOut[0].plot() + dOut[1].plot() + dOut[2].plot()