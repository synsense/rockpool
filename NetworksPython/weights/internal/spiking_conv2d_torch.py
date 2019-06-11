##
# spiking_conv2d_torch.py - Torch implementation of spiking convolution operation for CNNs
##

import numpy as np
import warnings
from collections import UserList
from functools import reduce
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class TorchConv2dLayer(nn.Module):
    def __init__(self, kernel, strides, padding, img_data_format="channels_last"):
        """
        TorchConv2dLayer - PyTorch Layer that does convolution
        :param kernel: numpy array kernel weights
        :param strides: tuple strides along each axis
        :param padding: padding along each axis
        :param img_data_format: str 'channels_first' or 'channels_last'
        """
        super(TorchConv2dLayer, self).__init__()

        self.img_data_format = img_data_format  # Expected image data format

        # Determine input and output channel numbers
        if img_data_format == "channels_last":
            num_in_channels, num_out_channels = kernel.shape[-2:]
            kernel_size = kernel.shape[:2]
            kernel = kernel.transpose((3, 2, 0, 1))
        elif img_data_format == "channels_first":
            num_out_channels, num_in_channels = kernel.shape[:2]
            kernel_size = kernel.shape[2:]

        self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(
            num_in_channels, num_out_channels, kernel_size=kernel_size, stride=strides
        )
        # Set the correct weights
        self.conv.weight.data = torch.from_numpy(kernel).float()
        # Set the correct biases
        # TODO: replace with actual biases
        self.conv.bias.data = torch.from_numpy(np.zeros((num_out_channels,))).float()

    def forward(self, tsrIndexReshaped):
        # This will always only be used for inference
        with torch.no_grad():
            # Restructure input
            if self.img_data_format == "channels_last":
                tsrIndexReshaped = tsrIndexReshaped.permute((0, 3, 1, 2))
            elif self.img_data_format == "channels_first":
                pass
            conv_out = self.conv(self.pad(tsrIndexReshaped))

            # Restructure output
            if self.img_data_format == "channels_last":
                conv_out = conv_out.permute((0, 2, 3, 1))
            elif self.img_data_format == "channels_first":
                pass
        return conv_out


class CNNWeightTorch(UserList):
    def __init__(
        self,
        inp_shape=None,
        kernels=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        mode="same",
        img_data_format="channels_last",
    ):
        """
        CNNWeightTorch class  - virtual array that allows convolutions on the input through indexing
        :param inp_shape:     tuple Shape of input
        :param kernels:    int No. of kernels for this convolutial weight matrix
        :param kernel_size: tuple Shape of each kernel, eg (5,5)
        :param strides:     tuple strides in each dimension for convolution
        :param mode:        str 'same' or 'valid' or 'full'
        :param img_data_format: str 'channels_first' or 'channels_last'
        (For more information on convolution parameters look into scipy.convolve2d documentation

        Important Note: inp_shape needs to be set before the use of indexing on this object.
        If left undefined/None, the input dimension is inferred from input dimensions, if the data is binary.
        If it is an integer index, then an IndexError is raised
        """
        # Initialize placeholder variables
        self._data = None  # Initialized when inp_shape is assigned
        self._inShape = None
        self.lyr_torch = None
        # Determine if there is a gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Set parameters from the initialization
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.strides = strides
        self.img_data_format = img_data_format
        self.mode = mode
        self.inp_shape = inp_shape  # This will initialize the weights
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
        try:
            if (type(index) is int) or (type(index) is list):
                # indexed by integer
                bIndex = np.zeros(self.inp_shape).astype(bool).flatten()
                bIndex[index] = True
                return self.__getitem__(bIndex)
            elif index.dtype is np.dtype(int):
                bIndex = np.zeros(self.inp_shape).astype(bool).flatten()
                bIndex[index] = True
                return self.__getitem__(bIndex)
            elif index.dtype is np.dtype("bool"):
                # Meat of this fuction
                # Reshape input
                bIndex = index
                bIndexReshaped = bIndex.reshape(self.inp_shape)
                # Ensure that the shape of input and this layer match
                assert self.inp_shape == bIndexReshaped.shape

                # The actual convolution happens here
                if bIndexReshaped.ndim == 3:

                    # Convert input to torch tensor
                    tsrIndexReshaped = torch.from_numpy(
                        bIndexReshaped.astype(float)
                    ).float()
                    tsrIndexReshaped = tsrIndexReshaped.unsqueeze(0)

                    # Move data to device
                    tsrIndexReshaped = tsrIndexReshaped.to(self.device)
                    # Do the convolution
                    tsrConvolution = self.lyr_torch(tsrIndexReshaped)
                else:
                    # Do the convolution
                    raise Exception("Incorrect dimensions")

                fmConvolution = tsrConvolution.cpu().numpy()
                return fmConvolution.flatten()
            else:
                raise TypeError("Indices should be of type [bool]")
        except AttributeError as e:
            raise e
        except IndexError as e:
            raise e

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

    def _update_torch_layer(self):
        # Determine input image shape
        if self.img_data_format == "channels_last":
            vImgShape = self.inp_shape[:2]
        elif self.img_data_format == "channels_first":
            vImgShape = self.inp_shape[-2:]
        # Calculate necessary padding
        padding = list(
            map(self._calculatePadding, vImgShape, self.kernel_size, self.strides)
        )
        self.padding = np.array(padding).flatten().tolist()
        del self.lyr_torch  # Free memory
        self.lyr_torch = TorchConv2dLayer(
            self.data.astype(float),
            self.strides,
            self.padding,
            img_data_format=self.img_data_format,
        )
        # Move to appropriate device
        self.lyr_torch.to(self.device)

    def __setitem__(self, index, value):
        """
        Set weights
        """
        raise Exception("Not Implemented")

    @property
    def shape(self):
        outSize = int(reduce(lambda x, y: x * y, self.out_shape))
        inSize = int(reduce(lambda x, y: x * y, self.inp_shape))
        return (inSize, outSize)

    @property
    def size(self):
        outSize = int(reduce(lambda x, y: x * y, self.out_shape))
        inSize = int(reduce(lambda x, y: x * y, self.inp_shape))
        return inSize * outSize

    @property
    def out_shape(self):
        if self._outShape is None:
            # create fake data
            tsrImg = torch.rand(self.inp_shape)
            # if self.img_data_format == "channels_last":
            #    tsrImg = tsrImg.unsqueeze(-1).to(self.device)
            #    tsrOutImg = self.lyr_torch(tsrImg)
            #    self._outShape = tsrOutImg.shape[:-1]
            # if self.img_data_format == "channels_first":
            #    tsrImg = tsrImg.unsqueeze(0).to(self.device)
            #    tsrOutImg = self.lyr_torch(tsrImg)
            #    self._outShape = tsrOutImg.shape[1:]
            tsrImg = tsrImg.unsqueeze(0).to(self.device)
            tsrOutImg = self.lyr_torch(tsrImg)
            self._outShape = tsrOutImg.shape[1:]
        return self._outShape

    @property
    def inp_shape(self):
        return self._inShape

    @inp_shape.setter
    def inp_shape(self, inp_shape):
        """
        Set the variable inp_shape and initialize a corresponding kernel
        """
        if inp_shape is None:
            return  # There is nothing to do
        if inp_shape == self._inShape:
            return  # No change

        # (No. of channels, hight, width in the order of img_data_format)
        assert len(inp_shape) == 3
        self._inShape = inp_shape
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
                *self.kernel_size, *self._inShape[2:], self.kernels
            )  # Kernel
            self.num_in_channels = self._inShape[2]
        elif self.img_data_format == "channels_first":
            self.data = np.random.rand(
                self.kernels, *self.inp_shape[:-2], *self.kernel_size
            )  # Kernel
            self.num_in_channels = self._inShape[0]
        # Initialize an updated torch layer with the updated weights
        self._update_torch_layer()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, mfWNewData):
        self._data = mfWNewData
        self._update_torch_layer()
        return
