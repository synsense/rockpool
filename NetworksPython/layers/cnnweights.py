import numpy as np
import warnings
from scipy import signal
from collections import UserList
from functools import reduce


class CNNWeight(UserList):
    def __init__(
        self,
        inShape=None,
        nKernels=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        mode="same",
        img_data_format="channels_last",
    ):
        """
        CNNWeight class is virtual array that allows convolutions on the input through indexing
        :param inShape:     tuple Shape of input
        :param nKernels:    int No. of kernels for this convolutial weight matrix
        :param kernel_size: tuple Shape of each kernel
        :param strides:     tuple strides in each dimension for convolution
        :param mode:        str 'same' or 'valid' or 'full'
        :param img_data_format: str 'channels_first' or 'channels_last'
        (For more information on convolution parameters look into scipy.convolve2d documentation

        Important Note: inShape needs to be set before the use of indexing on this object.
        If left undefine/None, the input dimension is inferred from input dimensions, if the data is binary.
        If it is an integer index, then an IndexError is raised
        """
        self.nKernels = nKernels
        self.kernel_size = kernel_size
        self.strides = strides
        self.img_data_format = img_data_format
        self.mode = mode
        self.data = None  # Initialized when inShape is assigned
        self._inShape = None
        self.inShape = inShape
        self.ndim = 2  # Because the input and output is always flattened

    def __len__(self):
        return len(self.data)

    def __contains__(self, x):
        return self.data.__contains__(x)

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, index):
        # NOTE: This approach is optimal when there is dense spiking activity.
        # If this is used on a per spike basis, the simulations will be pretty
        # slow. So this will need some optimization
        img_data_format = self.img_data_format  # Local variable
        data = self.data  # Local variable
        try:
            if (type(index) is int) or (type(index) is list):
                # indexed by integer
                bIndex = np.zeros(self.inShape).astype(bool).flatten()
                bIndex[index] = True
                return self.__getitem__(bIndex)
            elif index.dtype is np.dtype(int):
                bIndex = np.zeros(self.inShape).astype(bool).flatten()
                bIndex[index] = True
                return self.__getitem__(bIndex)
            elif index.dtype is np.dtype("bool"):
                # Meat of this fuction
                # Reshape input
                bIndex = index
                bIndexReshaped = bIndex.reshape(self.inShape)
                if self.inShape is None:
                    self.inShape == bIndexReshaped.shape

                # The actual convolution happens here
                aConvolution = []
                # Convolve each kernel individually
                for nKernelIndex in range(self.nKernels):
                    if img_data_format == "channels_last":
                        kernel = data[..., nKernelIndex]
                    elif img_data_format == "channels_first":
                        kernel = data[nKernelIndex]
                    fmConvolution = None  # Reset value
                    if bIndexReshaped.ndim == 3:
                        if img_data_format == "channels_last":
                            nFeatureMaps = bIndexReshaped.shape[-1]
                        elif img_data_format == "channels_first":
                            nFeatureMaps = bIndexReshaped.shape[0]
                        # Convolve each feature of input individually
                        for nFeatureIndex in range(nFeatureMaps):
                            if img_data_format == "channels_last":
                                img = bIndexReshaped[..., nFeatureIndex]
                                kern = kernel[..., nFeatureIndex]
                            elif img_data_format == "channels_first":
                                img = bIndexReshaped[nFeatureIndex]
                                kern = kernel[nFeatureIndex]
                            # Do the convolution
                            fmConvolutionFeature = self._do_convolve_2d_torch(img, kern)
                            if fmConvolution is None:
                                fmConvolution = fmConvolutionFeature
                            else:
                                fmConvolution += fmConvolutionFeature
                    else:
                        # Do the convolution
                        fmConvolution = self._do_convolve_2d_torch(
                            bIndexReshaped, kernel
                        )
                    aConvolution.append(fmConvolution)

                fmConvolution = np.array(aConvolution)
                if img_data_format == "channels_last":
                    fmConvolution = np.moveaxis(fmConvolution, 0, -1)
                return fmConvolution.flatten()
            else:
                raise TypeError("Indices should be of type [bool]")
        except AttributeError as e:
            raise e
        except IndexError as e:
            raise e

    def _do_convolve_2d(self, bIndexReshaped, kernel):
        """
        Performs the actual convolution call of a 2D image with a 2D kernel
        """
        # This is the function to modify for stride, padding and other parameters
        mfConvOut = signal.convolve2d(
            bIndexReshaped, kernel, mode=self.mode, boundary="fill"
        )

        # Subsample based on strides
        mfConvOut = mfConvOut[:: self.strides[0], :: self.strides[1]]
        return mfConvOut

    def _calculatePadding(self, nDimSize, nKWidth, nStrideLength):
        # Calculate necessary padding
        if self.mode == "same":
            # We need to calculate padding such that the input and output dimensions are the same
            # Really only meaningful if stride length is 1, so we will ignore strides here
            nStrides = 0
            while True:
                if nStrides * 1 + nKWidth >= nDimSize:
                    break
                else:
                    nStrides += 1
            # nStrides defines the dimensions of the output
            if nStrides + 1 == nDimSize:
                padding = [0, 0]
            else:
                padding = nDimSize - (nStrides + 1)
                if padding % 2 == 0:
                    padding = [int(padding / 2)] * 2
                else:
                    padding = [int(padding / 2), int(padding / 2) + 1]
        elif self.mode == "valid":
            padding = [0, 0]
        elif self.mode == "full":
            padding = nKWidth - 1
        else:
            raise Exception("Unknown convolution mode")
        return padding

    def _update_torch_layers(self):
        import torch
        import torch.nn as nn

        with torch.no_grad():
            self.pad = nn.ZeroPad2d(self.padding)
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel.shape, stride=self.strides)
            # Set the correct weights
            self.conv.weight.data = torch.from_numpy(
                kernel[np.newaxis, np.newaxis, ...]
            ).float()
            # Set the correct biases
            self.conv.bias.data = torch.from_numpy(np.zeros((1,))).float()

    def _do_convolve_2d_torch(self, bIndexReshaped, kernel):
        """
        Performs the actual convolution call of a 2D image with a 2D kernel, using torch
        """
        import torch
        import torch.nn as nn

        # Calculate necessary padding
        padding = list(
            map(
                self._calculatePadding,
                bIndexReshaped.shape,
                self.kernel_size,
                self.strides,
            )
        )
        self.padding = np.array(padding).flatten().tolist()

        class TorchLayer(nn.Module):
            def __init__(self, kernel, strides, padding):
                super(TorchLayer, self).__init__()
                self.pad = nn.ZeroPad2d(padding)
                self.conv = nn.Conv2d(1, 1, kernel_size=kernel.shape, stride=strides)
                # Set the correct weights
                self.conv.weight.data = torch.from_numpy(
                    kernel[np.newaxis, np.newaxis, ...]
                ).float()
                # Set the correct biases
                self.conv.bias.data = torch.from_numpy(np.zeros((1,))).float()

            def forward(self, tsrIndexReshaped):
                tsrConvOut = self.conv(self.pad(tsrIndexReshaped))
                return tsrConvOut

        with torch.no_grad():
            device = torch.device("cpu")
            torchlayer = TorchLayer(kernel, self.strides, self.padding)

            tsrIndexReshaped = torch.from_numpy(
                bIndexReshaped[np.newaxis, np.newaxis, ...].astype(float)
            ).float()

            # Do the convolution
            torchlayer.to(device)
            tsrIndexReshaped = tsrIndexReshaped.to(device)
            tsrConvOut = torchlayer(tsrIndexReshaped)
            mfConvOut = tsrConvOut.cpu().numpy()
        return mfConvOut[0, 0]

    def __setitem__(self, index, value):
        """
        Set weights
        """
        raise Exception("Not Implemented")

    @property
    def shape(self):
        outSize = int(reduce(lambda x, y: x * y, self.outShape))
        inSize = int(reduce(lambda x, y: x * y, self.inShape))
        return (inSize, outSize)

    @property
    def size(self):
        outSize = int(reduce(lambda x, y: x * y, self.outShape))
        inSize = int(reduce(lambda x, y: x * y, self.inShape))
        return inSize * outSize

    @property
    def outShape(self):
        if self._outShape is None:
            if self.img_data_format == "channels_last":
                self._outShape = (
                    *(
                        self._do_convolve_2d(
                            np.zeros(self.inShape[:2]), np.zeros(self.kernel_size)
                        ).shape
                    ),
                    self.nKernels,
                )
            if self.img_data_format == "channels_first":
                self._outShape = (
                    self.nKernels,
                    *(
                        self._do_convolve_2d(
                            np.zeros(self.inShape[-2:]), np.zeros(self.kernel_size)
                        ).shape
                    ),
                )
        return self._outShape

    @property
    def inShape(self):
        return self._inShape

    @inShape.setter
    def inShape(self, inShape):
        """
        Set the variable inShape and initialize a corresponding kernel
        """
        if inShape is None:
            inShape = []
        if inShape == self._inShape:
            return  # No change
        self._inShape = inShape
        self._outShape = None
        self.initialize_weights()

    def initialize_weights(self):
        """
        This function reinitializes the weights of this object
        """
        # Initialize kernels
        if self.data is not None:
            warnings.warn(
                "Re-Initializing convolutional kernel because of inSize change"
            )
        if self.img_data_format == "channels_last":
            self.data = np.random.rand(
                *self.kernel_size, *self._inShape[2:], self.nKernels
            )  # Kernel
        elif self.img_data_format == "channels_first":
            self.data = np.random.rand(
                self.nKernels, *self.inShape[:-2], *self.kernel_size
            )  # Kernel

    def reverse_dot(self, vnInput):
        """
        Each element of vnInput corresponds to the number of input
        spikes from the respective channel. The method will return
        the sum of the corresponding weights, summed by the number
        of spikes.
        This yields the same result as a dot product vnInput @ M,
        where M is a matrix representing the weights.
        """
        vnInput = np.array(vnInput).flatten()
        assert vnInput.size == self.shape[0], "Input vector must be of size {}".format(
            self.shape[0]
        )
        # - Collect the respective weights, multiply them by the number of
        #   spikes and add them up
        return np.sum(
            (
                nNumSpikes * self[iNeuronID]  # Multiply weights with spike counts
                for iNeuronID, nNumSpikes in enumerate(vnInput)
            )
        )
