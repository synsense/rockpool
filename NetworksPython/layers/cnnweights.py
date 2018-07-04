import numpy as np
from scipy import signal
from collections import UserList
from functools import reduce


class CNNWeight(UserList):
    def __init__(self, inShape=None, nKernels=1, kernel_size=(5, 5)):
        '''
        CNNWeight class is virtual array that allows convolutions on the input through indexing
        :param inShape:     tuple Shape of input
        :param nKernels:    int No. of kernels for this convolutial weight matrix
        :param kernel_size: tuple Shape of each kernel

        Important Note: inShape needs to be set before the use of indexing on this object.
        If left undefine/None, the input dimension is inferred from input dimensions, if the data is binary.
        If it is an integer index, then an IndexError is raised
        '''
        self.inShape = inShape
        self.nKernels = nKernels
        self.data = np.random.rand(nKernels, *kernel_size)  # Kernel
        self.ndim = 2


    def __len__(self):
        return len(self.data)

    def __contains__(self, x):
        return self.data.__contains__(x)

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, index):
        # Note: This approach is optimal when there is dense spiking activity.
        # If this is used on a per spike basis, the simulations will be pretty
        # slow. So this will need some optimization
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
            elif index.dtype is np.dtype('bool'):
                # Reshape input
                bIndex = index
                bIndexReshaped = bIndex.reshape(self.inShape)
                # The actual convolution happens here
                aConvolution = []
                for kernel in self.data:
                    # Convolve each kernel individually
                    fmConvolution = signal.convolve2d(bIndexReshaped,
                                                      kernel,
                                                      mode='same',
                                                      boundary='fill')
                    aConvolution.append(fmConvolution)
                fmConvolution = np.array(aConvolution)
                return fmConvolution.flatten()
            else:
                raise TypeError('Indexes should be of type [bool]')
        except AttributeError as e:
            raise e
        except IndexError as e:
            raise e

    @property
    def shape(self):
        # NOTE: This does not take strides into account
        inSize = int(reduce(lambda x, y: x*y, self.inShape))
        return (inSize, self.nKernels*inSize)

    @property
    def size(self):
        # NOTE: This does not take strides into account
        inSize = int(reduce(lambda x, y: x*y, self.inShape))
        return (inSize*self.nKernels*inSize)
