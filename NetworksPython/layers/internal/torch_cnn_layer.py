import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

# Internal class dependencies
from ...timeseries import TSEvent
from .spiking_conv2d_torch import CNNWeightTorch
from .iaf_cl import FFCLIAF

from typing import Optional, Union, List, Tuple, Generator
from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Absolute tolerance, e.g. for comparing float values
fTolAbs = 1e-9


class FFCLIAFTorch(FFCLIAF):
    """
    FFCLIAFTorch - Feedforward layer of integrate and fire neurons with constant leak
    Implemented using pytorch for speed and is meant to be used for convolutions
    """

    def __init__(
        self,
        mfW: CNNWeightTorch,
        vfVBias: float = 0,
        fVThresh: float = 8,
        fVReset: float = 0,
        fVSubtract: float = 8,
        tDt: float = 1,
        vnIdMonitor: Union[bool, int, None, ArrayLike] = [],
        strName: str = "unnamed",
    ):
        """
        FFCLIAFTorch - Feedforward layer of integrate and fire neurons with constant leak

        :param mfW:         array-like  Input weight matrix
        :param vfVBias:     array-like  Constant bias to be added to state at each time step
        :param vfVThresh:   array-like  Spiking threshold
        :param vfVReset:    array-like  Reset potential after spike (also see param bSubtract)
        :param vfVSubtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.
        :vnIdMonitor:       array-like  IDs of neurons to be recorded
        :param strName:     str  Name of this layer.
        """

        # Call parent constructor
        FFCLIAF.__init__(
            self,
            mfW=mfW,
            vfVBias=vfVBias,
            vfVThresh=fVThresh,
            vfVReset=fVReset,
            vfVSubtract=fVSubtract,
            tDt=tDt,
            vnIdMonitor=vnIdMonitor,
            strName=strName,
        )

        self.fVThresh = fVThresh
        self.fVSubtract = fVSubtract
        self.fVReset = fVReset

        # Placeholder variable
        self._lyrTorch = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.reset_state()

    @property
    def lyrTorch(self):
        if self._lyrTorch is None:
            self._init_torch_layer()
        return self._lyrTorch

    @lyrTorch.setter
    def lyrTorch(self, lyrNewTorch):
        self._lyrTorch = lyrNewTorch

    def _init_torch_layer(self):
        # Initialize torch layer
        self.lyrTorch = TorchSpikingConv2dLayer(
            nInChannels=self.mfW.nInChannels,
            nOutChannels=self.mfW.nKernels,
            kernel_size=self.mfW.kernel_size,
            strides=self.mfW.strides,
            padding=self.mfW.padding,
            fVThresh=self.fVThresh,
            fVSubtract=self.fVSubtract,
            fVReset=self.fVReset,
        )
        # Set torch weights and bias
        if self.mfW.img_data_format == "channels_first":
            self.lyrTorch.conv.weight.data = torch.from_numpy(self.mfW.data).float()
            self.lyrTorch.conv.bias.data = torch.from_numpy(
                np.zeros((self.mfW.nKernels,))
            ).float()
        elif self.mfW.img_data_format == "channels_last":
            weights = self.mfW.data
            self.lyrTorch.conv.weight.data = torch.from_numpy(
                weights.transpose((3, 2, 0, 1))
            ).float()
            self.lyrTorch.conv.bias.data = torch.from_numpy(
                np.zeros((self.mfW.nKernels,))
            ).float()
            self.lyrTorch.conv.bias.data = torch.from_numpy(
                np.zeros((self.mfW.nKernels,))
            ).float()
            pass
        else:
            raise Exception(
                "img_data_format(={}) not understood".format(self.mfW.img_data_format)
            )

        # Transfer layer to appropriate device
        self.lyrTorch.to(self.device)
        return

    def _prepare_input(
        self, tsInput: Optional[TSEvent] = None, nNumTimeSteps: int = 1
    ) -> np.ndarray:
        """
        Prepare input stream and return a binarized vector of spikes
        """
        # - End time of evolution
        tFinal = self.t + nNumTimeSteps * self.tDt
        # - Extract spike timings and channels
        if tsInput is not None:
            if tsInput.isempty():
                # Return an empty list with all zeros
                vbSpikeRaster = np.zeros((self.nSizeIn), bool)
            else:
                # Ensure number of channels is atleast as many as required
                try:
                    assert tsInput.nNumChannels >= self.nSizeIn
                except AssertionError as err:
                    warn(
                        self.strName
                        + ": Expanding input dimensions to match layer size."
                    )
                    tsInput.nNumChannels = self.nSizeIn

                # Extract spike data from the input variable
                mfSpikeRaster = tsInput.xraster(
                    tDt=self.tDt, tStart=self.t, tStop=tFinal
                )

                ## - Make sure size is correct
                # mfSpikeRaster = mfSpikeRaster[:nNumTimeSteps, :]
                # assert mfSpikeRaster.shape == (nNumTimeSteps, self.nSizeIn)
                yield from mfSpikeRaster  # Yield a single time step
                return
        else:
            # Return an empty list with all zeros
            vbSpikeRaster = np.zeros((self.nSizeIn), bool)

        while True:
            yield vbSpikeRaster

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSEvent  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """
        # Compute number of simulation time steps
        if nNumTimeSteps is None:
            nNumTimeSteps = int((tDuration + fTolAbs) // self.tDt)

        # - Generate input in rasterized form
        mfInptSpikeRaster = self._prepare_input(tsInput, nNumTimeSteps=nNumTimeSteps)

        # Convert input to torch tensors
        mfInptSpikeRaster = [next(mfInptSpikeRaster) for i in range(nNumTimeSteps)]
        tsrIn = torch.from_numpy(np.array(mfInptSpikeRaster, np.uint8)).type(
            torch.float
        )
        # Reshape flat data to images and channels
        tsrInReshaped = tsrIn.reshape(-1, *self.mfW.inShape)
        print(tsrInReshaped.shape)
        # Restructure input
        if self.mfW.img_data_format == "channels_last":
            tsrInReshaped = tsrInReshaped.permute((0, 3, 1, 2))
        elif self.mfW.img_data_format == "channels_first":
            pass

        # Process data
        tsrInReshaped = tsrInReshaped.to(self.device)
        tsrOut = self.lyrTorch(tsrInReshaped)

        # Reshape data again to the class's format
        if self.mfW.img_data_format == "channels_last":
            tsrOut = tsrOut.permute((0, 2, 3, 1))
        elif self.mfW.img_data_format == "channels_first":
            pass
        # Flatten output and return
        mbOutRaster = tsrOut.numpy()
        mbOutRaster = mbOutRaster.reshape((nNumTimeSteps, -1))

        # Create time series from raster
        vnTimeSteps, vnChannels = np.nonzero(mbOutRaster)
        vtTimeTrace = self.t + (vnTimeSteps + 1) * self.tDt

        # Update time
        self._nTimeStep += nNumTimeSteps

        evOut = TSEvent(
            vtTimeTrace, vnChannels, nNumChannels=self.nSize, strName=self.strName
        )
        return evOut


class TorchSpikingConv2dLayer(nn.Module):
    def __init__(
        self,
        nInChannels: int = 1,
        nOutChannels: int = 1,
        kernel_size: ArrayLike = (1, 1),
        strides: ArrayLike = (1, 1),
        padding: ArrayLike = (0, 0, 0, 0),
        fVThresh: float = 8,
        fVSubtract: Optional[float] = None,
        fVReset: float = 0,
    ):
        """
        Pytorch implementation of a spiking neuron with convolutional inputs
        SUBTRACT superseeds Reset value
        """
        super(TorchSpikingConv2dLayer, self).__init__()  # Init nn.Module
        self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(
            nInChannels, nOutChannels, kernel_size=kernel_size, stride=strides
        )

        # Initialize neuron states
        self.tsrState = nn.Parameter(None)
        self.fVSubtract = fVSubtract
        self.fVReset = fVReset
        self.fVThresh = fVThresh

    def forward(self, tsrBinaryInput):
        # Determine no. of time steps from input
        nNumTimeSteps = len(tsrBinaryInput)

        # Convolve all inputs at once
        tsrConvOut = self.conv(self.pad(tsrBinaryInput))

        # - Count number of spikes for each neuron in each time step
        vnNumSpikes = np.zeros(tsrConvOut.shape[1:], int)

        # Local variables
        tsrState = self.tsrState
        fVSubtract = self.fVSubtract
        fVThresh = self.fVThresh
        fVReset = self.fVReset

        # Create a vector to hold all output spikes
        tsrNumSpikes = torch.zeros(
            nNumTimeSteps, *tsrConvOut.shape[1:], dtype=torch.uint8
        )

        # Loop over time steps
        tsrState = self.tsrState.data
        if len(tsrState) == 0:
            tsrState = tsrConvOut[0].new_zeros(tsrConvOut.shape[1:])

        for iCurrentTimeStep in tqdm(range(nNumTimeSteps)):
            # if len(tsrState) == 0:
            #    tsrState = tsrConvOut[iCurrentTimeStep]
            # else:
            #    tsrState = tsrState + tsrConvOut[iCurrentTimeStep]
            tsrState = tsrState + tsrConvOut[iCurrentTimeStep]

            # - Reset spike counter
            vnNumSpikes[:] = 0

            # - Check threshold crossings for spikes
            vbRecSpikeRaster = tsrState >= fVThresh

            # - Reset or subtract from membrane state after spikes
            if fVSubtract is not None:
                while vbRecSpikeRaster.any():
                    # - Subtract from states
                    tsrState[vbRecSpikeRaster] -= fVSubtract
                    # - Add to spike counter
                    tsrNumSpikes[iCurrentTimeStep][vbRecSpikeRaster] += 1
                    # - Neurons that are still above threshold will emit another spike
                    vbRecSpikeRaster = tsrState >= fVThresh
            else:
                # - Add to spike counter
                tsrNumSpikes[iCurrentTimeStep] = vbRecSpikeRaster
                # - Reset neuron states
                tsrState[vbRecSpikeRaster] = fVReset

            # Record spikes

        self.tsrState.data = tsrState

        return tsrNumSpikes
