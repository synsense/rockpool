###
# weights.py -- Utility functions for generating and manipulating networks
###

from typing import Callable, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
from copy import deepcopy
import random


def combine_FF_Rec_stack(weights_ff: np.ndarray, weights_rec: np.ndarray) -> np.ndarray:
    """
    combine_FF_Rec_stack - Combine a FFwd and Recurrent weight matrix into a single recurrent weight matrix
    :param weights_ff:   MxN np.ndarray
    :param weights_rec:  NxN np.ndarray

    :return: (M+N)x(M+N) np.ndarray combined weight matrix
    """
    assert (
        weights_ff.shape[1] == weights_rec.shape[0]
    ), "FFwd and Rec weight matrices must have compatible shapes (MxN and NxN)."

    assert (
        weights_rec.shape[0] == weights_rec.shape[1]
    ), "`weights_rec` must be a square matrix."

    # - Determine matrix sizes
    nFFSize = weights_ff.shape[0]
    nRecSize = weights_rec.shape[0]
    mfCombined = np.zeros((nFFSize + nRecSize, nFFSize + nRecSize))

    # - Combine matrices
    mfCombined[-nRecSize:, -nRecSize:] = weights_rec
    mfCombined[:nFFSize, -nRecSize:] = weights_ff

    return mfCombined


def RndmSparseEINet(
    nResSize: int,
    fConnectivity: float = 1,
    fhRand: Callable[[int], np.ndarray] = np.random.rand,
    bPartitioned: bool = False,
    fRatioExc: float = 0.5,
    fScaleInh: float = 1,
    fNormalize: float = 0.95,
) -> np.ndarray:
    """
    RndmSparseEINet - Return a (sparse) matrix defining reservoir weights

    :param nResSize:        int Number of reservoir units
    :param fConnectivity:   float Ratio of non-zero weight matrix elements
                                  (must be between 0 and 1)
    :param fhRand:          Function used to draw random weights. Must accept an integer n
                            as argument and return n positive values.
    :param bPartitioned:    bool  Partition weight matrix into excitatory and inhibitory
    :param fRatioExc:       float Ratio of excitatory weights (must be between 0 and 1)
    :param fInhiblScale:    float Scale of negative weights to positive weights
    :param fNormalize:      float If not None, matrix is normalized so that
                                  its spectral radius equals fNormalize
    :return:                np.ndarray Weight matrix
    """

    # - Make sure parameters are in correct range
    fConnectivity = np.clip(fConnectivity, 0, 1)
    fRatioExc = np.clip(fRatioExc, 0, 1)

    # - Number of non-zero elements in matrix
    nNumWeights = int(fConnectivity * nResSize ** 2)

    # - Array for storing connection strengths
    mfWeights = np.zeros(nResSize ** 2)
    # - Draw non-zero weights
    mfWeights[:nNumWeights] = fhRand(nNumWeights)

    if not bPartitioned:
        # - Number inhibitory connections
        nNumInhWeights = int(nNumWeights * (1 - fRatioExc))
        # - Multiply inhibitory connections with -fScaleInh
        mfWeights[:nNumInhWeights] *= -fScaleInh

    # - Shuffle weights and bring array into 2D-form
    np.random.shuffle(mfWeights)
    mfWeights = mfWeights.reshape(nResSize, nResSize)

    if bPartitioned:
        # - Number of excitatory neurons
        nNumExcNeurons = int(nResSize * fRatioExc)

        # - All rows with index > nNumExcNeurons correspond to inhibitory neurons
        #   Multiply corresponding elements with -1 and scale with fScaleInh
        mfWeights[nNumExcNeurons:] *= -fScaleInh

    if fNormalize is not None:
        # Normalize weights matrix so that its spectral radius = fNormalize
        vEigenvalues = np.linalg.eigvals(mfWeights)
        mfWeights *= fNormalize / np.amax(np.abs(vEigenvalues))

    return mfWeights


def RandomEINet(
    nNumExc: int,
    nNumInh: int,
    fInhWFactor: float = 1,
    fhRand: Callable[[int], float] = lambda n: np.random.randn(n, n) / np.sqrt(n),
) -> np.ndarray:
    """
    RandomEINet - Generate a random nicely-tuned real-valued reservoir matrix
    :param nNumExc:         Number of excitatory neurons in the network
    :param nNumInh:         Number of inhibitory neurons in the network
    :param fInhWFactor:     Factor relating total inhibitory and excitatory weight (w_inh = fInhWFactor * w_exc) default: 1
    :param fhRand:          Function used to draw initial random weights. Default: numpy.random.randn

    :return:                Network connectivity weight matrix
    """

    # - Generate base connectivity matrix
    weights = fhRand(nNumExc + nNumInh)

    # - Enforce excitatory and inhibitory partitioning
    vnExc = range(nNumExc)
    vnInh = [n + nNumExc for n in range(nNumInh)]

    mfWE = weights[vnExc, :]
    mfWE[mfWE < 0] = 0
    mfWE /= np.sum(mfWE, 0)

    mfWI = weights[vnInh, :]
    mfWI[mfWI > 0] = 0
    mfWI = mfWI / np.sum(mfWI, 0) * -np.abs(fInhWFactor)

    weights = np.concatenate((mfWE, mfWI), 0)

    return weights


def WilsonCowanNet(
    nNumNodes: int,
    fSelfExc: float = 1,
    fSelfInh: float = 1,
    fExcSigma: float = 1,
    fInhSigma: float = 1,
    fhRand: Callable[[int], float] = lambda n: np.random.randn(n, n) / np.sqrt(n),
) -> (np.ndarray, np.ndarray):
    """
    WilsonCowanNet - FUNCTION Define a Wilson-Cowan network of oscillators

    :param nNumNodes: Number of (E+I) nodes in the network
    :param fSelfExc:  Strength of self-excitation autapse. Default 1.0
    :param fSelfInh:  Strength of self-inhibition autapse. Default 1.0
    :param fExcSigma: Sigma of random excitatory connections. Default 1.0
    :param fInhSigma: Sigma of random inhibitory connections. Default 1.0
    :param fhRand:    Function used to draw random weights. Default: numpy.random.randn
    :return:          2N x 2N weight matrix weights
    """

    # - Check arguments, enforce reasonable defaults

    # - Build random matrices
    mfEE = (
        np.clip(fhRand(nNumNodes), 0, None) * fExcSigma
        + np.identity(nNumNodes) * fSelfExc
    )
    mfIE = (
        np.clip(fhRand(nNumNodes), 0, None) * fExcSigma
        + np.identity(nNumNodes) * fSelfExc
    )
    mfEI = np.clip(fhRand(nNumNodes), None, 0) * fInhSigma + np.identity(
        nNumNodes
    ) * -np.abs(fSelfInh)
    mfII = np.clip(fhRand(nNumNodes), None, 0) * fInhSigma + np.identity(
        nNumNodes
    ) * -np.abs(fSelfInh)

    # - Normalise matrix components
    fENorm = fExcSigma * stats.norm.pdf(0, 0, fExcSigma) * nNumNodes + fSelfExc
    fINorm = fInhSigma * stats.norm.pdf(0, 0, fInhSigma) * nNumNodes + np.abs(fSelfInh)
    mfEE = mfEE / np.sum(mfEE, 0) * fENorm
    mfIE = mfIE / np.sum(mfIE, 0) * fENorm
    mfEI = mfEI / np.sum(mfEI, 0) * -fINorm
    mfII = mfII / np.sum(mfII, 0) * -fINorm

    # - Compose weight matrix
    weights = np.concatenate(
        (np.concatenate((mfEE, mfIE)), np.concatenate((mfEI, mfII))), axis=1
    )

    return weights


def WipeNonSwitchingEigs(
    weights: np.ndarray, vnInh: np.ndarray = None, fInhTauFactor: float = 1
) -> np.ndarray:
    """
    WipeNonSwitchingEigs - Eliminate eigenvectors that do not lead to a partition switching

    :param weights:             Network weight matrix [N x N]
    :param vnInh:           Vector [M x 1] of indices of inhibitory neurons. Default: None
    :param fInhTauFactor:   Factor relating inhibitory and excitatory time constants. tau_i = f * tau_e, tau_e = 1 Default: 1
    :return:                (weights, mfJ) Weight matrix and estimated Jacobian
    """

    nResSize = weights.shape[0]

    # - Compute Jacobian
    mfJ = weights - np.identity(nResSize)

    if vnInh is not None:
        mfJ[vnInh, :] /= fInhTauFactor

    # - Numerically estimate eigenspectrum
    [vfD, v] = np.linalg.eig(mfJ)

    # - Identify and wipe non-switching eigenvectors
    mfNormVec = v / np.sign(vfD)
    vbNonSwitchingPartition = np.all(mfNormVec > 0, 0)
    vbNonSwitchingPartition[0] = False
    print("Number of eigs wiped: " + str(np.sum(vbNonSwitchingPartition)))
    vfD[vbNonSwitchingPartition] = 0

    # - Reconstruct Jacobian and weight matrix
    mfJHat = np.real(v @ np.diag(vfD) @ np.linalg.inv(v))
    mfWHat = mfJHat

    if vnInh is not None:
        mfWHat[vnInh, :] *= fInhTauFactor

    mfWHat += np.identity(nResSize)

    # - Attempt to rescale weight matrix for optimal dynamics
    mfWHat *= fInhTauFactor * 30

    return mfWHat, mfJHat


def UnitLambdaNet(nResSize: int) -> np.ndarray:
    """
    UnitLambdaNet - Generate a network from Norm(0, sqrt(N))

    :param nResSize: Number of neurons in the network
    :return:    weight matrix
    """

    # - Draw from a Normal distribution
    weights = np.random.randn(nResSize, nResSize) / np.sqrt(nResSize)
    return weights


def DiscretiseWeightMatrix(
    weights: np.ndarray,
    nMaxConnections: int = 3,
    nLimitInputs: int = None,
    nLimitOutputs: int = None,
) -> (np.ndarray, np.ndarray, float, float):
    """
    DiscretiseWeightMatrix - Discretise a weight matrix by strength

    :param weights:             an arbitrary real-valued weight matrix.
    :param nMaxConnections: the integer maximum number of synaptic connections that may be made between
                            two neurons. Excitatory and inhibitory weights will be discretised
                            separately. Default: 3
    :param nLimitInputs:    integer Number of permissable inputs per neuron
    :param nLimitOutputs:   integer Number of permissable outputs per neuron
    :return: (mfWD, mnNumConns, fEUnitary, fIUnitary)
    """

    # - Make a copy of the input
    weights = deepcopy(weights)
    mfWE = weights * (weights > 0)
    mfWI = weights * (weights < 0)

    # - Select top N inputs per neuron
    if nLimitOutputs is not None:
        mfWE = np.array(
            [
                row * np.array(np.argsort(-row) < np.round(nLimitOutputs / 2), "float")
                for row in mfWE.T
            ]
        ).T
        mfWI = np.array(
            [
                row
                * np.array(np.argsort(-abs(row)) < np.round(nLimitOutputs / 2), "float")
                for row in mfWI.T
            ]
        ).T

    if nLimitInputs is not None:
        mfWE = np.array(
            [
                row * np.array(np.argsort(-row) < np.round(nLimitInputs / 2), "float")
                for row in mfWE
            ]
        )
        mfWI = np.array(
            [
                row
                * np.array(np.argsort(-abs(row)) < np.round(nLimitInputs / 2), "float")
                for row in mfWI
            ]
        )

    # - Determine unitary connection strengths. Clip at max absolute values
    fEUnitary = np.max(weights) / nMaxConnections
    fIUnitary = np.max(-weights) / nMaxConnections

    # - Determine number of unitary connections
    if np.any(weights > 0):
        mnEConns = np.round(weights / fEUnitary) * (weights > 0)
    else:
        mnEConns = 0

    if np.any(weights < 0):
        mnIConns = -np.round(weights / fIUnitary) * (weights < 0)
    else:
        mnIConns = 0

    mnNumConns = mnEConns - mnIConns

    # - Discretise real-valued weight matrix
    mfWD = mnEConns * fEUnitary - mnIConns * fIUnitary

    # - Return matrices
    return mfWD, mnNumConns


def one_dim_exc_res(size, nNeighbour, bZeroDiagonal=True):
    """one_dim_exc_res - Recurrent weight matrix where each neuron is connected
                         to its nNeighbour nearest neighbours on a 1D grid.
                         Only excitatory connections.
    """
    weights_res = np.zeros((size, size))
    nBound = int(np.floor(nNeighbour / 2))
    for iPost in range(size):
        weights_res[max(0, iPost - nBound) : min(iPost + nBound + 1, size), iPost] = 1
        if bZeroDiagonal:
            weights_res[iPost, iPost] = 0

    return weights_res


def two_dim_exc_res(
    size: int,
    nNeighbour: int,
    tupfWidthNeighbour: Union[float, Tuple[float, float]],
    tupnGridDim: Optional[Tuple[int, int]] = None,
    bMultipleConn: bool = True,
):
    # - Determine width of connection probability distribution
    if isinstance(tupfWidthNeighbour, int):
        # assume isotropic probability distribution
        tupfWidthNeighbour = (tupfWidthNeighbour, tupfWidthNeighbour)

    # - Determine grid size
    if tupnGridDim is None:
        # - Square grid
        nGridLength = int(np.ceil(np.sqrt(size)))
        tupnGridDim = (nGridLength, nGridLength)
    else:
        # - Make sure grid is large enough
        assert (
            tupnGridDim[0] * tupnGridDim[1] >= size
        ), "Grid dimensions are too small."
        # - Make sure grid dimensions are integers
        assert isinstance(tupnGridDim[0], int) and isinstance(
            tupnGridDim[1], int
        ), "tupnGridDim must be tuple of two positive integers."

    # - Matrix for determining connection probability to any location on the grid relative to given neuron
    #   Given neuron location corresponds to [tupnGridDim[0], tupnGridDim[1]]
    vx = np.arange(-tupnGridDim[0] + 1, tupnGridDim[0])
    vy = np.arange(-tupnGridDim[1] + 1, tupnGridDim[1])
    mx, my = np.meshgrid(vx, vy)
    # Gaussian Distribution of connection probability
    mfPConnect = np.exp(
        -((mx / tupfWidthNeighbour[0]) ** 2 + (my / tupfWidthNeighbour[1]) ** 2)
    )
    # No self-connections
    mfPConnect[tupnGridDim[0] - 1, tupnGridDim[1] - 1] = 0

    # - Weight matrix
    mnW = np.zeros((size, size))

    # - Iterate over neurons and assign their presynaptic connections
    for n_id in range(size):
        # Neuron coordinates in 2D grid
        tupnGridIndex = np.unravel_index(n_id, tupnGridDim)
        # Probabilities to connect to other neureons
        vfPConnectThis = (
            mfPConnect[
                (tupnGridDim[0] - 1)
                - tupnGridIndex[0] : (2 * tupnGridDim[0] - 1)
                - tupnGridIndex[0],
                (tupnGridDim[1] - 1)
                - tupnGridIndex[1] : (2 * tupnGridDim[1] - 1)
                - tupnGridIndex[1],
            ]
        ).flatten()[:size]
        # Normalize probabilities
        vfPConnectThis /= np.sum(vfPConnectThis)
        # Choose connections
        viPreSyn = np.random.choice(
            size, size=nNeighbour, p=vfPConnectThis, replace=bMultipleConn
        )
        for iPreSyn in viPreSyn:
            mnW[iPreSyn, n_id] += 1

    return mnW


def add_random_long_range(
    weights_res, nLongRange, bAvoidExisting: bool = False, bMultipleConn: bool = True
):
    assert weights_res.shape[0] == weights_res.shape[1], "weights_res must be a square matrix"

    for iPost in range(weights_res.shape[0]):
        if bAvoidExisting:
            viFarNeurons = np.where(weights_res[:, iPost] == 0)[0]
        else:
            viFarNeurons = np.arange(weights_res.shape[0])
        # - Make sure diagonal elements are excluded
        viFarNeurons = viFarNeurons[viFarNeurons != iPost]
        viPreSynConnect = np.random.choice(
            viFarNeurons, size=nLongRange, replace=bMultipleConn
        )
        for iPreSyn in viPreSynConnect:
            weights_res[iPreSyn, iPost] += 1
    return weights_res


def partitioned_2d_reservoir(
    size_in: int = 64,
    nSizeRec: int = 256,
    nSizeInhib: int = 64,
    nInToRec: int = 16,
    nRecShort: int = 24,
    nRecLong: int = 8,
    nRecToInhib: int = 64,
    nInhibToRec: int = 16,
    nMaxInputConn: int = 64,
    tupfWidthNeighbour: Union[float, Tuple[float, float]] = (16.0, 16.0),
):
    if nMaxInputConn is not None:
        assert (
            nInToRec + nRecShort + nRecLong + nInhibToRec <= nMaxInputConn
        ) and nInhibToRec <= nMaxInputConn, (
            "Maximum number of presynaptic connections per neuron exceeded."
        )

    # - Input-To-Recurrent part
    mnWInToRec = np.zeros((size_in, nSizeRec))
    viPreSynConnect = np.random.choice(size_in, size=nInToRec * nSizeRec)
    for iPreIndex, iPostIndex in zip(
        viPreSynConnect, np.repeat(np.arange(nSizeRec), nInToRec)
    ):
        mnWInToRec[iPreIndex, iPostIndex] += 1
    # - Recurrent-To-Inhibitory part
    mnWRecToInhib = np.zeros((nSizeRec, nSizeInhib))
    viPreSynConnect = np.random.choice(nSizeRec, size=nRecToInhib * nSizeInhib)
    for iPreIndex, iPostIndex in zip(
        viPreSynConnect, np.repeat(np.arange(nSizeInhib), nRecToInhib)
    ):
        mnWRecToInhib[iPreIndex, iPostIndex] += 1
    # - Inhibitory-To-Recurrent part
    mnWInhibToRec = np.zeros((nSizeInhib, nSizeRec))
    viPreSynConnect = np.random.choice(nSizeInhib, size=nInhibToRec * nSizeRec)
    for iPreIndex, iPostIndex in zip(
        viPreSynConnect, np.repeat(np.arange(nSizeRec), nInhibToRec)
    ):
        mnWInhibToRec[iPreIndex, iPostIndex] -= 1
    # - Recurrent short range connecitons
    mnWRec = two_dim_exc_res(
        nSizeRec, nNeighbour=nRecShort, tupfWidthNeighbour=tupfWidthNeighbour
    )
    # - Add long range connections
    mnWRec = add_random_long_range(
        mnWRec, nRecLong, bAvoidExisting=False, bMultipleConn=True
    )

    # - Put matrix together
    nSizeTotal = size_in + nSizeRec + nSizeInhib
    mnW = np.zeros((nSizeTotal, nSizeTotal))
    mnW[:size_in, size_in : size_in + nSizeRec] = mnWInToRec
    mnW[size_in : size_in + nSizeRec, size_in : size_in + nSizeRec] = mnWRec
    mnW[size_in : size_in + nSizeRec, -nSizeInhib:] = mnWRecToInhib
    mnW[-nSizeInhib:, size_in : size_in + nSizeRec] = mnWInhibToRec

    return mnW


def DynapseConform(
    shape,
    fConnectivity=None,
    fRatioExc=0.5,
    nLimitInputs=64,
    nLimitOutputs=1024,
    tupfWExc=(1, 2),
    tupfWInh=(1, 2),
    tupfProbWExc=(0.5, 0.5),
    tupfProbWInh=(0.5, 0.5),
    fNormalize=None,
):
    """
    DynapseConform - Create a weight matrix that conforms the specifications of the Dynapse Chip

    :param shape:        tuple Shape of the weight matrix
    :param fConnectivity:   float Ratio of non-zero vs. zero weights - limited by nLimitInputs/shape[0]
    :param fRatioExc:       float Ratio of excitatory vs. inhibitory synapses

    :param nLimitInputs:    int   Maximum fan-in per neuron
    :param nLimitOutputs:   int   Maximum fan-out per neuron

    :param tupfWEcx:        tuple Possible strengths for excitatory synapses
    :param tupfWInh:        tuple Possible strengths for inhibitory synapses
    :param tupfProbWEcx:    tuple Probabilities for excitatory synapse strengths
    :param tupfProbWInh:    tuple Probabilities for inhibitory synapse strengths

    :param fNormalize:      float If not None, matrix will be normalized wrt spectral radius

    :return weights:            2D-np.ndarray Generated weight matrix
    :return mnCount:        2D-np.ndarray Number of assigned weights per connection
    :return dmnCount:       dict Number of assigned weights, separated by weights
                                 Useful for re-construction of the matrix
    """

    # - Make sure input weights all have correct sign
    tupfWExc = tuple(abs(w) for w in tupfWExc)
    tupfWInh = tuple(-abs(w) for w in tupfWInh)

    # - Determine size of matrix
    try:
        shape = tuple(shape)
        assert (
            len(shape) == 2 or len(shape) == 1
        ), "Only 2-dimensional matrices can be created."
        if len(shape) == 1:
            shape = (shape[0], shape[0])
    except TypeError:
        assert isinstance(
            shape, int
        ), "shape must be integer or array-like of size 2."
        shape = (shape, shape)

    # - Matrix for holding weihgts
    weights = np.zeros(shape, float)
    # - Matrix for counting number of assigned connections for each synapse
    mnCount = np.zeros_like(weights, int)
    # - Dict of matrices to count assignments for each weight and synapse
    #   Can be used for re-constructing weights:
    #     mfW_reconstr = np.zeros(shape)
    #     for fWeight, miAssignments in dmnCount:
    #         mfW_reconstr += fWeight * miAssignments
    dmnCount = {weight: np.zeros_like(weights, int) for weight in (*tupfWExc, *tupfWInh)}

    # - Input synapses per neuron
    if fConnectivity is not None:
        nNumInputs = int(fConnectivity * weights.shape[0])
        assert (
            nNumInputs <= nLimitInputs
        ), "Maximum connectivity for given reservoir size and input limits is {}".format(
            float(nLimitInputs) / weights.shape[0]
        )
    else:
        nNumInputs = min(nLimitInputs, weights.shape[0])

    # - Numbers of excitatory and inhibitory inputs per neuron
    #   (could also be stochastic for each neuron....)
    nNumExcIn = int(np.round(nNumInputs * fRatioExc))
    nNumInhIn = nNumInputs - nNumExcIn

    # - Iterrate over neurons (columns of weights) and set their inputs
    #   Do so in random order. Otherwise small nLimitOutputs could result in more
    #   deterministic columns towards the end if no more weights can be assigned anymore
    for iCol in np.random.permutation(weights.shape[1]):
        # - Array holding non-zero weights
        vfWeights = np.zeros(nNumInputs)
        # - Generate excitatory weights
        vfWeights[:nNumExcIn] = np.random.choice(
            tupfWExc, size=nNumExcIn, p=tupfProbWExc, replace=True
        )
        # - Generate inhibitory weights
        vfWeights[nNumExcIn:] = np.random.choice(
            tupfWInh, size=nNumInhIn, p=tupfProbWInh, replace=True
        )
        # - Shuffle, so that inhibitory and excitatory weights are equally likely
        np.random.shuffle(vfWeights)

        # - Count how many weights can still be set for each row without exceeding nLimitOutputs
        vnFree = nLimitOutputs - np.sum(mnCount, axis=1)
        viFreeIndices = np.repeat(np.arange(vnFree.size), vnFree)

        # - Assign corresponding input neurons, according to what is available in viFreeIndices
        if viFreeIndices.size > 0:
            viInputNeurons = np.random.choice(
                viFreeIndices, size=nNumInputs, replace=False
            )

            # - Generate actual column of weight matrix and count number of assigned weights
            for fWeight, iInpNeur in zip(
                vfWeights[: np.size(viInputNeurons)], viInputNeurons
            ):
                weights[iInpNeur, iCol] += fWeight
                dmnCount[fWeight][iInpNeur, iCol] += 1
                mnCount[iInpNeur, iCol] += 1

    if fNormalize is not None:
        # - Normalize matrix according to spectral radius
        vfEigenVals = np.linalg.eigvals(weights)
        fSpectralRad = np.amax(np.abs(vfEigenVals))
        if fSpectralRad == 0:
            print("Matrix is 0, will not normalize.")
        else:
            fScale = fNormalize / fSpectralRad
            weights *= fScale
            # - Also scale keys in dmnCount accordingly
            dmnCount = {fScale * k: v for k, v in dmnCount.items()}

    return weights, mnCount, dmnCount


def In_Res_Dynapse(
    size: int,
    fInputDensity=1,
    fConnectivity=None,
    fRatioExcRec=0.5,
    fRatioExcIn=0.5,
    nLimitInputs=64,
    nLimitOutputs=1024,
    tupfWExc=(1, 2),
    tupfWInh=(1, 2),
    tupfProbWExcRec=(0.5, 0.5),
    tupfProbWInhRec=(0.5, 0.5),
    tupfProbWExcIn=(0.5, 0.5),
    tupfProbWInhIn=(0.5, 0.5),
    fNormalize=None,
    verbose=False,
    bLeaveSpaceForInput=False,
):
    """
    In_Res_Dynapse - Create input weights and recurrent weights for reservoir, respecting dynapse specifications

    :param size:           int Reservoir size
    :param fInputDensity:   float Ratio of non-zero vs. zero input connections
    :param fConnectivity:   float Ratio of non-zero vs. zero weights - limited by nLimitInputs/shape[0]
    :param fRatioExcRec:    float Ratio of excitatory vs. inhibitory recurrent synapses
    :param fRatioExcIn:     float Ratio of excitatory vs. inhibitory input synapses

    :param nLimitInputs:    int   Maximum fan-in per neuron
    :param nLimitOutputs:   int   Maximum fan-out per neuron

    :param tupfWEcx:        tuple Possible strengths for excitatory synapses
    :param tupfWInh:        tuple Possible strengths for inhibitory synapses
    :param tupfProbWEcxRes: tuple Probabilities for excitatory recurrent synapse strengths
    :param tupfProbWInhRec: tuple Probabilities for inhibitory recurrent synapse strengths
    :param tupfProbWEcxIn:  tuple Probabilities for excitatory input synapse strengths
    :param tupfProbWInhIn:  tuple Probabilities for inhibitory input synapse strengths

    :param fNormalize:      float If not None, matrix will be normalized wrt spectral radius

    :param verbose:        bool Currently unused

    :param bLeaveSpaceForInput: bool Limit number of input connections to ensure fan-in
                                     to include all all input weights

    :return weights_in:          2D-np.ndarray (Nx1) Generated input weights
    :return weights:            2D-np.ndarray (NxN) Generated weight matrix
    :return mnCount:        2D-np.ndarray (NxN) Number of assigned weights per connection
    :return dmnCount:       dict Number of assigned weights, separated by weights
                                 Useful for re-construction of the matrix
    """

    # - Matrix with weights. First row corresponds to input weights, others to reservoir matrix
    weights_res, mnCount, dmnCount = DynapseConform(
        shape=(size, size),
        fConnectivity=fConnectivity,
        fRatioExc=fRatioExcRec,
        nLimitInputs=nLimitInputs - int(bLeaveSpaceForInput),
        nLimitOutputs=nLimitOutputs,
        tupfWExc=tupfWExc,
        tupfWInh=tupfWInh,
        tupfProbWExc=tupfProbWExcRec,
        tupfProbWInh=tupfProbWInhRec,
        fNormalize=fNormalize,
    )

    # - For each reservoir neuron, check whether it still has space for an input
    vbFree = (nLimitInputs - np.sum(mnCount, axis=0)) > 0
    # - Total number of input weights to be assigned
    nNumInputs = (
        min(np.sum(vbFree), int(fInputDensity * size))
        if fInputDensity is not None
        else np.sum(vbFree)
    )

    if nNumInputs == 0:
        print(
            "WARNING: Could not assign any input weights, "
            + "try setting bLeaveSpaceForInput=True or reducing fConnectivity"
        )
        return np.zeros((size, 1)), weights_res, mnCount, dmnCount
    # - Number excitatory and inhibitory input weights
    nNumExcIn = int(nNumInputs * fRatioExcIn)
    nNumInhIn = nNumInputs - nNumExcIn
    # - Array holding non-zero weights
    vfWeights = np.zeros(nNumInputs)
    # - Extract normalized weight values from dmnCount
    lfValues = sorted(dmnCount.keys())
    tupfWInExc = tuple(lfValues[-2:])
    tupfWInInh = tuple(lfValues[:2])
    # - Generate excitatory weights
    vfWeights[:nNumExcIn] = np.random.choice(
        tupfWInExc, size=nNumExcIn, p=tupfProbWExcIn, replace=True
    )
    # - Generate inhibitory weights
    vfWeights[nNumExcIn:] = np.random.choice(
        tupfWInInh, size=nNumInhIn, p=tupfProbWInhIn, replace=True
    )
    # - Ids of neurons to which weights will be assigned
    viNeurons = np.random.choice(np.where(vbFree)[0], size=nNumInputs, replace=False)
    # - Input weight vector
    weights_in = np.zeros(size)
    for n_id, fWeight in zip(viNeurons, vfWeights):
        weights_in[n_id] = fWeight

    return weights_in.reshape(-1, 1), weights_res, mnCount, dmnCount


def In_Res_Dynapse_Flex(
    size: int,
    size_in: None,
    nMaxInputConn: int = 1,
    fInputDensity=1,
    fConnectivity=None,
    fRatioExcRec=0.5,
    fRatioExcIn=0.5,
    nLimitInputs=64,
    nLimitOutputs=1024,
    tupfWExc=(1, 1),
    tupfWInh=(1, 1),
    tupfProbWExcRec=(0.5, 0.5),
    tupfProbWInhRec=(0.5, 0.5),
    tupfProbWExcIn=(0.5, 0.5),
    tupfProbWInhIn=(0.5, 0.5),
    fNormalize=None,
    verbose=False,
    bLeaveSpaceForInput=True,
):
    """
    In_Res_Dynapse_Flex - Like In_Res_Dynapse but number of input weights can be chosen

    :param size:           int Reservoir size
    :param size_in:         int Number of neurons in input layer
    :param nMaxInputConn:   int Number of presynaptic connections that are used for input connections per neuron
    :param fInputDensity:   float Ratio of non-zero vs. zero input connections
    :param fConnectivity:   float Ratio of non-zero vs. zero weights - limited by nLimitInputs/shape[0]
    :param fRatioExcRec:    float Ratio of excitatory vs. inhibitory recurrent synapses
    :param fRatioExcIn:     float Ratio of excitatory vs. inhibitory input synapses

    :param nLimitInputs:    int   Maximum fan-in per neuron
    :param nLimitOutputs:   int   Maximum fan-out per neuron

    :param tupfWEcx:        tuple Possible strengths for excitatory synapses
    :param tupfWInh:        tuple Possible strengths for inhibitory synapses
    :param tupfProbWEcxRes: tuple Probabilities for excitatory recurrent synapse strengths
    :param tupfProbWInhRec: tuple Probabilities for inhibitory recurrent synapse strengths
    :param tupfProbWEcxIn:  tuple Probabilities for excitatory input synapse strengths
    :param tupfProbWInhIn:  tuple Probabilities for inhibitory input synapse strengths

    :param fNormalize:      float If not None, matrix will be normalized wrt spectral radius

    :param verbose:        bool Currently unused

    :param bLeaveSpaceForInput: bool Limit number of input connections to ensure fan-in
                                     to include all all input weights

    :return weights_in:          2D-np.ndarray (Nx1) Generated input weights
    :return weights:            2D-np.ndarray (NxN) Generated weight matrix
    :return mnCount:        2D-np.ndarray (NxN) Number of assigned weights per connection
    :return dmnCount:       dict Number of assigned weights, separated by weights
                                 Useful for re-construction of the matrix
    """

    # - Set input size
    size_in = size if size_in is None else size_in

    # - Matrix with weights. First row corresponds to input weights, others to reservoir matrix
    weights_res, mnCount, dmnCount = DynapseConform(
        shape=(size, size),
        fConnectivity=fConnectivity,
        fRatioExc=fRatioExcRec,
        nLimitInputs=nLimitInputs - int(bLeaveSpaceForInput) * nMaxInputConn,
        nLimitOutputs=nLimitOutputs,
        tupfWExc=tupfWExc,
        tupfWInh=tupfWInh,
        tupfProbWExc=tupfProbWExcRec,
        tupfProbWInh=tupfProbWInhRec,
        fNormalize=fNormalize,
    )

    # - For each reservoir neuron, check whether it still has space for an input
    vnFree = nLimitInputs - np.sum(mnCount, axis=0)
    # - Total number of input weights to be assigned
    nNumInputs = (
        min(np.sum(vnFree), int(fInputDensity * size * nMaxInputConn))
        if fInputDensity is not None
        else np.sum(vnFree)
    )

    if nNumInputs == 0:
        print(
            "WARNING: Could not assign any input weights, "
            + "try setting bLeaveSpaceForInput=True or reducing fConnectivity"
        )
        return np.zeros((size, 1)), weights_res, mnCount, dmnCount

    # - Number excitatory and inhibitory input weights
    nNumExcIn = int(nNumInputs * fRatioExcIn)
    nNumInhIn = nNumInputs - nNumExcIn

    # - Array holding non-zero weights
    vfWeights = np.zeros(nNumInputs)

    # - Extract normalized weight values from dmnCount
    lfValues = sorted(dmnCount.keys())
    tupfWInExc = tuple(lfValues[-2:])
    tupfWInInh = tuple(lfValues[:2])

    # - Generate excitatory weights
    vfWeights[:nNumExcIn] = np.random.choice(
        tupfWInExc, size=nNumExcIn, p=tupfProbWExcIn, replace=True
    )
    # - Generate inhibitory weights
    vfWeights[nNumExcIn:] = np.random.choice(
        tupfWInInh, size=nNumInhIn, p=tupfProbWInhIn, replace=True
    )

    # - Ids of neurons to which weights can be assigned, repeated according to number of free connections
    viAvailableNeurons = np.repeat(np.arange(size), vnFree)
    # - Ids of neurons to which weights will be assigned, repeated according to number assignments
    viAssignNeurons = np.random.choice(
        viAvailableNeurons, size=nNumInputs, replace=False
    )
    # - Ids of input neurons to be connected to reservoir neurons
    viInputNeurons = np.random.choice(np.arange(size_in), size=nNumInputs, replace=True)
    # - Input weight matrix
    weights_in = np.zeros((size_in, size))
    weights_in[viInputNeurons, viAssignNeurons] = vfWeights

    return weights_in, weights_res, mnCount, dmnCount


def digital(
    shape,
    fConnectivity=None,
    fRatioExc=0.5,
    nBitResolution=4,
    nLimitInputs=64,
    nLimitOutputs=1024,
    fRangeUse=1,
    fNormalize=None,
    fRescale: Optional[float] = None,
):
    """
    digital - Create a weight matrix that conforms the specifications of the Dynapse Chip

    :param shape:        tuple Shape of the weight matrix
    :param fConnectivity:   float Ratio of non-zero vs. zero weights - limited by nLimitInputs/shape[0]
    :param fRatioExc:       float Ratio of excitatory vs. inhibitory synapses

    :param nBitResolution:  int   Weight resolution in bits. Before normalization, weights will
                                  be integers between -2**(nBitResolution/2) and 2**(nBitResolution/2)-1

    :param nLimitInputs:    int   Maximum fan-in per neuron
    :param nLimitOutputs:   int   Maximum fan-out per neuron

    :param fRangeUse:       float Ratio of the possible value range that is actually used. If smaller than
                                  than 1, input weights can be larger than recurrent weights, later on.

    :param fNormalize:      float If not None, matrix will be normalized wrt spectral radius
    :param fRescale:        float If not None, multiply weights with this constant

    :return mnW:            2D-np.ndarray Generated weight matrix
    :return mnCount:        2D-np.ndarray Number of assigned weights per connection
    :return fScale:         float Factor used for normalization and rescaling
    """

    # - Determine size of matrix
    try:
        shape = tuple(shape)
        assert (
            len(shape) == 2 or len(shape) == 1
        ), "Only 2-dimensional matrices can be created."
        if len(shape) == 1:
            shape = (shape[0], shape[0])
    except TypeError:
        assert isinstance(
            shape, int
        ), "shape must be integer or array-like of size 2."
        shape = (shape, shape)

    # - Matrix for holding weights
    mnW = np.zeros(shape, float)
    # - Matrix for counting number of assigned connections for each synapse
    mnCount = np.zeros_like(mnW, int)
    # - Dict of matrices to count assignments for each weight and synapse
    #   Can be used for re-constructing mnW:
    #     mnW_reconstr = np.zeros(shape)
    #     for fWeight, miAssignments in dmnCount:
    #         mnW_reconstr += fWeight * miAssignments
    # dmnCount = {weight: np.zeros_like(mnW, int) for weight in (*tupfWExc, *tupfWInh)}

    # - Input synapses per neuron
    if fConnectivity is not None:
        nNumInputs = int(fConnectivity * mnW.shape[0])
        assert (
            nNumInputs <= nLimitInputs
        ), "Maximum connectivity for given reservoir size and input limits is {}".format(
            float(nLimitInputs) / mnW.shape[0]
        )
    else:
        nNumInputs = min(nLimitInputs, mnW.shape[0])

    # - Numbers of excitatory and inhibitory inputs per neuron
    #   (could also be stochastic for each neuron....)
    nNumExcIn = int(np.round(nNumInputs * fRatioExc))
    nNumInhIn = nNumInputs - nNumExcIn

    # - Determine value range of weights
    nMinWeight = int(-2 ** (nBitResolution / 2) * fRangeUse)
    nMaxWeight = int(
        2 ** (nBitResolution / 2) * fRangeUse
    )  ## Due to how np.random.randin works, max weight will be nMaxWeight-1

    # - Iterrate over neurons (columns of mnW) and set their inputs
    #   Do so in random order. Otherwise small nLimitOutputs could result in more
    #   deterministic columns towards the end if no more weights can be assigned anymore
    for iCol in np.random.permutation(mnW.shape[1]):
        # - Array holding non-zero weights
        vnWeights = np.zeros(nNumInputs)
        # - Generate excitatory weights
        vnWeights[:nNumExcIn] = np.random.randint(1, nMaxWeight, size=nNumExcIn)
        # - Generate inhibitory weights
        vnWeights[nNumExcIn:] = np.random.randint(nMinWeight, 0, size=nNumInhIn)
        # - Shuffle, so that inhibitory and excitatory weights are equally likely
        np.random.shuffle(vnWeights)

        # - Count how many weights can still be set for each row without exceeding nLimitOutputs
        vnFree = nLimitOutputs - np.sum(mnCount, axis=1)
        viFreeIndices = np.repeat(np.arange(vnFree.size), vnFree)

        if viFreeIndices.size > 0:
            # - Assign corresponding input neurons, according to what is available in viFreeIndices
            viInputNeurons = np.random.choice(
                viFreeIndices, size=nNumInputs, replace=False
            )

            # - Generate actual column of weight matrix and count number of assigned weights
            for fWeight, iInpNeur in zip(
                vnWeights[: np.size(viInputNeurons)], viInputNeurons
            ):
                mnW[iInpNeur, iCol] += fWeight
                # dmnCount[fWeight][iInpNeur, iCol] += 1
                mnCount[iInpNeur, iCol] += 1

    if fNormalize is None:
        fScale = 1
    else:
        # - Normalize matrix according to spectral radius
        vfEigenVals = np.linalg.eigvals(mnW)
        fSpectralRad = np.amax(np.abs(vfEigenVals))
        fScale = fNormalize / fSpectralRad
        mnW *= fScale
        # - Also scale keys in dmnCount accordingly
        # dmnCount = {fScale * k: v for k, v in dmnCount.items()}
    if fRescale is not None:
        # - Rescale weights by factor fRescale
        mnW *= fRescale
        fScale *= fRescale

    return mnW, mnCount, fScale  # , dmnCount


def in_res_digital(
    size: int,
    fInputDensity=1,
    fConnectivity=None,
    fRatioExcRec=0.5,
    fRatioExcIn=0.5,
    nBitResolution=4,
    nLimitInputs=64,
    nLimitOutputs=1024,
    fRatioRecIn=1,
    fNormalize=None,
    verbose=False,
    bLeaveSpaceForInput=False,
):
    """
    in_res_digital - Create input weights and recurrent weights for reservoir, respecting dynapse specifications

    :param size:           int Reservoir size
    :param fInputDensity:   float Ratio of non-zero vs. zero input connections
    :param fConnectivity:   float Ratio of non-zero vs. zero weights - limited by nLimitInputs/shape[0]

    :param fRatioExcRec:    float Ratio of excitatory vs. inhibitory recurrent synapses
    :param fRatioExcIn:     float Ratio of excitatory vs. inhibitory input synapses

    :param nBitResolution:  int   Weight resolution in bits. Before normalization, weights will
                                  be integers between -2**(nBitResolution/2) and 2**(nBitResolution/2)-1

    :param nLimitInputs:    int   Maximum fan-in per neuron
    :param nLimitOutputs:   int   Maximum fan-out per neuron

    :param fRatioRecIn:     float Ratio of input vs recurrent weight ranges
                                  smaller than 1 leads to stronger input than recurrent weights
    :param fNormalize:      float If not None, matrix will be normalized wrt spectral radius

    :param verbose:        bool Currently unused

    :param bLeaveSpaceForInput: bool Limit number of input connections to ensure fan-in
                                     to include all all input weights

    :return weights_in:          2D-np.ndarray (Nx1) Generated input weights
    :return mnW:            2D-np.ndarray (NxN) Generated weight matrix
    :return mnCount:        2D-np.ndarray (NxN) Number of assigned weights per connection
    :return dmnCount:       dict Number of assigned weights, separated by weights
                                 Useful for re-construction of the matrix
    """

    # - Matrix with weights. First row corresponds to input weights, others to reservoir matrix
    mnWRes, mnCount, fScale = digital(
        shape=(size, size),
        fConnectivity=fConnectivity,
        fRatioExc=fRatioExcRec,
        nBitResolution=nBitResolution,
        nLimitInputs=nLimitInputs - int(bLeaveSpaceForInput),
        nLimitOutputs=nLimitOutputs,
        fRangeUse=fRatioRecIn,
        fNormalize=fNormalize,
    )

    # - For each reservoir neuron, check whether it still has space for an input
    vbFree = (nLimitInputs - np.sum(mnCount, axis=0)) > 0
    # - Total number of input weights to be assigned
    nNumInputs = (
        min(np.sum(vbFree), int(fInputDensity * size))
        if fInputDensity is not None
        else np.sum(vbFree)
    )

    if nNumInputs == 0:
        print(
            "Weights: WARNING: Could not assign any input weights, "
            + "try setting bLeaveSpaceForInput=True or reducing fConnectivity"
        )
        return np.zeros((size, 1)), mnWRes, mnCount

    # - Number excitatory and inhibitory input weights
    nNumExcIn = int(nNumInputs * fRatioExcIn)
    nNumInhIn = nNumInputs - nNumExcIn
    # - Array holding non-zero weights
    vnWeights = np.zeros(nNumInputs)

    # - Determine value range of unnormalized weights
    nMinWeight = int(-2 ** nBitResolution / 2)
    nMaxWeight = int(2 ** nBitResolution / 2)
    # - Generate excitatory weights
    vnWeights[:nNumExcIn] = fScale * np.random.randint(1, nMaxWeight, size=nNumExcIn)
    # - Generate inhibitory weights
    vnWeights[nNumExcIn:] = fScale * np.random.randint(nMinWeight, 0, size=nNumInhIn)

    # - Ids of neurons to which weights will be assigned
    viNeurons = np.random.choice(np.where(vbFree)[0], size=nNumInputs, replace=False)
    # - Input weight vector
    weights_in = np.zeros(size)
    for n_id, fWeight in zip(viNeurons, vnWeights):
        weights_in[n_id] = fWeight

    return weights_in.reshape(-1, 1), mnWRes, mnCount, fScale


def IAFSparseNet(
    nResSize: int = 100, fMean: float = None, fStd: float = None, fDensity: float = 1.0
) -> np.ndarray:
    """
    IAFSparseNet - Return a random sparse reservoir, scaled for a standard IAF spiking network

    :param nResSize:    int Number of neurons in reservoir. Default: 100
    :param fMean:       float Mean connection strength. Default: -4.5mV
    :param fStd:        float Std. Dev. of connection strengths. Default: 5.5mV
    :param fDensity:    float 0..1 Density of connections. Default: 1 (full connectivity)
    :return:            [N x N] array Weight matrix
    """

    # - Check inputs
    assert (fDensity >= 0.0) and (
        fDensity <= 1.0
    ), "`fDensity` must be between 0 and 1."

    # - Set default values
    if fMean is None:
        fMean = -0.0045 / fDensity / nResSize

    if fStd is None:
        fStd = 0.0055 / np.sqrt(fDensity)

    # - Determine sparsity
    nNumConnPerRow = int(fDensity * nResSize)
    nNumNonConnPerRow = nResSize - nNumConnPerRow
    mbConnection = [
        random.sample([1] * nNumConnPerRow + [0] * nNumNonConnPerRow, nResSize)
        for _ in range(nResSize)
    ]
    vbConnection = np.array(mbConnection).reshape(-1)

    # - Randomise recurrent weights
    weights = (
        np.random.randn(nResSize ** 2) * np.asarray(fStd) + np.asarray(fMean)
    ) * vbConnection
    return weights.reshape(nResSize, nResSize)
