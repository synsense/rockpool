from typing import Callable
import numpy as np
import scipy.stats as stats
from copy import deepcopy
from brian2.units.allunits import mvolt
import random


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
        #   Multiply coresponding elements with -1 and scale with fScaleInh
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
    mfW = fhRand(nNumExc + nNumInh)

    # - Enforce excitatory and inhibitory partitioning
    vnExc = range(nNumExc)
    vnInh = [n + nNumExc for n in range(nNumInh)]

    mfWE = mfW[:, vnExc]
    mfWE[mfWE < 0] = 0
    mfWE /= np.sum(mfWE, 0)

    mfWI = mfW[:, vnInh]
    mfWI[mfWI > 0] = 0
    mfWI = mfWI / np.sum(mfWI, 0) * -np.abs(fInhWFactor)

    mfW = np.concatenate((mfWE, mfWI), 1)

    return mfW


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
    :return:          2N x 2N weight matrix mfW
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
    mfW = np.concatenate(
        (np.concatenate((mfEE, mfIE)), np.concatenate((mfEI, mfII))), axis=1
    )

    return mfW


def WipeNonSwitchingEigs(
    mfW: np.ndarray, vnInh: np.ndarray = None, fInhTauFactor: float = 1
) -> np.ndarray:
    """
    WipeNonSwitchingEigs - Eliminate eigenvectors that do not lead to a partition switching

    :param mfW:             Network weight matrix [N x N]
    :param vnInh:           Vector [M x 1] of indices of inhibitory neurons. Default: None
    :param fInhTauFactor:   Factor relating inhibitory and excitatory time constants. tau_i = f * tau_e, tau_e = 1 Default: 1
    :return:                (mfW, mfJ) Weight matrix and estimated Jacobian
    """

    nResSize = mfW.shape[0]

    # - Compute Jacobian
    mfJ = mfW - np.identity(nResSize)

    if vnInh is not None:
        mfJ[vnInh, :] /= fInhTauFactor

    # - Numerically estimate eigenspectrum
    [vfD, mfV] = np.linalg.eig(mfJ)

    # - Identify and wipe non-switching eigenvectors
    mfNormVec = mfV / np.sign(vfD)
    vbNonSwitchingPartition = np.all(mfNormVec > 0, 0)
    vbNonSwitchingPartition[0] = False
    print("Number of eigs wiped: " + str(np.sum(vbNonSwitchingPartition)))
    vfD[vbNonSwitchingPartition] = 0

    # - Reconstruct Jacobian and weight matrix
    mfJHat = np.real(mfV @ np.diag(vfD) @ np.linalg.inv(mfV))
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
    mfW = np.random.randn(nResSize, nResSize) / np.sqrt(nResSize)
    return mfW


def DiscretiseWeightMatrix(
    mfW: np.ndarray,
    nMaxConnections: int = 3,
    nLimitInputs: int = None,
    nLimitOutputs: int = None,
) -> (np.ndarray, np.ndarray, float, float):
    """
    DiscretiseWeightMatrix - Discretise a weight matrix by strength

    :param mfW:             an arbitrary real-valued weight matrix.
    :param nMaxConnections: the integer maximum number of synaptic connections that may be made between
                            two neurons. Excitatory and inhibitory weights will be discretised
                            separately. Default: 3
    :param nLimitInputs:    integer Number of permissable inputs per neuron
    :param nLimitOutputs:   integer Number of permissable outputs per neuron
    :return: (mfWD, mnNumConns, fEUnitary, fIUnitary)
    """

    # - Make a copy of the input
    mfW = deepcopy(mfW)
    mfWE = mfW * (mfW > 0)
    mfWI = mfW * (mfW < 0)

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
    fEUnitary = np.max(mfW) / nMaxConnections
    fIUnitary = np.max(-mfW) / nMaxConnections

    # - Determine number of unitary connections
    if np.any(mfW > 0):
        mnEConns = np.round(mfW / fEUnitary) * (mfW > 0)
    else:
        mnEConns = 0

    if np.any(mfW < 0):
        mnIConns = -np.round(mfW / fIUnitary) * (mfW < 0)
    else:
        mnIConns = 0

    mnNumConns = mnEConns - mnIConns

    # - Discretise real-valued weight matrix
    mfWD = mnEConns * fEUnitary - mnIConns * fIUnitary

    # - Return matrices
    return mfWD, mnNumConns

def DynapseConform(
    tupShape,
    fConnectivity=None,
    fRatioExc=0.5,
    nLimitInputs=64,
    nLimitOutputs=1024,
    tupfWExc=(1,2),
    tupfWInh=(1,2),
    tupfProbWExc=(0.5,0.5),
    tupfProbWInh=(0.5,0.5),
    fNormalize=None,
):
    """
    DynapseConform - Create a weight matrix that conforms the specifications of the Dynapse Chip

    :param tupShape:        tuple Shape of the weight matrix
    :param fConnectivity:   float Ratio of non-zero vs. zero weights - limited by nLimitInputs/tupShape[0]
    :param fRatioExc:       float Ratio of excitatory vs. inhibitory synapses
    
    :param nLimitInputs:    int   Maximum fan-in per neuron
    :param nLimitOutputs:   int   Maximum fan-out per neuron
    
    :param tupfWEcx:        tuple Possible strengths for excitatory synapses
    :param tupfWInh:        tuple Possible strengths for inhibitory synapses
    :param tupfProbWEcx:    tuple Probabilities for excitatory synapse strengths
    :param tupfProbWInh:    tuple Probabilities for inhibitory synapse strengths
    
    :param fNormalize:      float If not None, matrix will be normalized wrt spectral radius

    :return mfW:            2D-np.ndarray Generated weight matrix
    :return mnCount:        2D-np.ndarray Number of assigned weights per connection
    :return dmnCount:       dict Number of assigned weights, separated by weights
                                 Useful for re-construction of the matrix    
    """

    # - Make sure input weights all have correct sign
    tupfWExc = tuple(abs(w) for w in tupfWExc)
    tupfWInh = tuple(-abs(w) for w in tupfWInh)

    # - Determine size of matrix
    try:
        tupShape = tuple(tupShape)
        assert len(tupShape) == 2 or len(tupShape) == 1, 'Only 2-dimensional matrices can be created.'
        if len(tupShape) == 1:
            tupShape = (tupShape[0], tupShape[0])
    except TypeError:
        assert isinstance(tupShape, int), 'tupShape must be integer or array-like of size 2.'
        tupShape = (tupShape,tupShape)

    # - Matrix for holding weihgts
    mfW = np.zeros(tupShape, float)
    # - Matrix for counting number of assigned connections for each synapse
    mnCount = np.zeros_like(mfW, int)
    # - Dict of matrices to count assignments for each weight and synapse
    #   Can be used for re-constructing mfW:
    #     mfW_reconstr = np.zeros(tupShape)
    #     for fWeight, miAssignments in dmnCount:
    #         mfW_reconstr += fWeight * miAssignments
    dmnCount = {weight : np.zeros_like(mfW, int) for weight in (*tupfWExc, *tupfWInh)}
      
    # - Input synapses per neuron
    if fConnectivity is not None:
        nNumInputs = int(fConnectivity * mfW.shape[0])
        assert nNumInputs <= nLimitInputs, (
            "Maximum connectivity for given reservoir size and input limits is {}".format(float(nLimitInputs) / mfW.shape[0]))
    else:
        nNumInputs = min(nLimitInputs, mfW.shape[0])

    # - Numbers of excitatory and inhibitory inputs per neuron
    #   (could also be stochastic for each neuron....)
    nNumExcIn = int(np.round(nNumInputs * fRatioExc))
    nNumInhIn = nNumInputs - nNumExcIn

    # - Iterrate over neurons (columns of mfW) and set their inputs
    #   Do so in random order. Otherwise small nLimitOutputs could result in more
    #   deterministic columns towards the end if no more weights can be assigned anymore
    for iCol in np.random.permutation(mfW.shape[1]):
        # - Array holding non-zero weights
        vfWeights = np.zeros(nNumInputs)
        # - Generate excitatory weights
        vfWeights[ : nNumExcIn] = np.random.choice(tupfWExc, size=nNumExcIn, p=tupfProbWExc, replace=True)
        # - Generate inhibitory weights
        vfWeights[nNumExcIn : ] = np.random.choice(tupfWInh, size=nNumInhIn, p=tupfProbWInh, replace=True)

        # - Count how many weights can still be set for each row without exceeding nLimitOutputs
        vnFree = nLimitOutputs - np.sum(mnCount, axis=1)
        liFreeIndices = [index for index, num in enumerate(vnFree) for _ in range(num)]

        # - Assign corresponding input neurons, according to what is available in liFreeIndices
        viInputNeurons = np.random.choice(liFreeIndices, size=nNumInputs, replace=False)

        # - Generate actual column of weight matrix and count number of assigned weights
        for fWeight, iInpNeur in zip(vfWeights, viInputNeurons):
            mfW[iInpNeur, iCol] += fWeight
            dmnCount[fWeight][iInpNeur, iCol] += 1
            mnCount[iInpNeur, iCol] += 1

    if fNormalize is not None:
        # - Normalize matrix according to spectral radius
        vfEigenVals = np.linalg.eigvals(mfW)
        fSpectralRad = np.amax(np.abs(vfEigenVals))
        fScale = fNormalize / fSpectralRad
        mfW *= fScale
        # - Also scale keys in dmnCount accordingly
        dmnCount = {fScale*k : v for k, v in dmnCount.items()}

    return mfW, mnCount, dmnCount

def In_Res_Dynapse(
    nSize : int,
    fInputDensity=1,
    fConnectivity=None,
    fRatioExcRes=0.5,
    fRatioExcIn=0.5,
    nLimitInputs=64,
    nLimitOutputs=1024,
    tupfWExc=(1,2),
    tupfWInh=(1,2),
    tupfProbWExcRes=(0.5,0.5),
    tupfProbWInhRes=(0.5,0.5),
    tupfProbWExcIn=(0.5,0.5),
    tupfProbWInhIn=(0.5,0.5),
    fNormalize=None,
    bVerbose=False,
    bLeaveSpaceForInput=False,
):
    """
    In_Res_Dynapse - Create input weights and recurrent weights for reservoir, respecting dynapse specifications

    :param nSize:           int Reservoir size
    :param fInputDensity:   float Ratio of non-zero vs. zero input connections
    :param fConnectivity:   float Ratio of non-zero vs. zero weights - limited by nLimitInputs/tupShape[0]
    :param fRatioExcRes:    float Ratio of excitatory vs. inhibitory recurrent synapses
    :param fRatioExcIn:     float Ratio of excitatory vs. inhibitory input synapses
    
    :param nLimitInputs:    int   Maximum fan-in per neuron
    :param nLimitOutputs:   int   Maximum fan-out per neuron
    
    :param tupfWEcx:        tuple Possible strengths for excitatory synapses
    :param tupfWInh:        tuple Possible strengths for inhibitory synapses
    :param tupfProbWEcxRes: tuple Probabilities for excitatory recurrent synapse strengths
    :param tupfProbWInhRes: tuple Probabilities for inhibitory recurrent synapse strengths
    :param tupfProbWEcxIn:  tuple Probabilities for excitatory input synapse strengths
    :param tupfProbWInhIn:  tuple Probabilities for inhibitory input synapse strengths
    
    :param fNormalize:      float If not None, matrix will be normalized wrt spectral radius

    :param bVerbose:        bool Currently unused

    :param bLeaveSpaceForInput: bool Limit number of input connections to ensure fan-in
                                     to include all all input weights

    :return vfWIn:          2D-np.ndarray (Nx1) Generated input weights
    :return mfW:            2D-np.ndarray (NxN) Generated weight matrix
    :return mnCount:        2D-np.ndarray (NxN) Number of assigned weights per connection
    :return dmnCount:       dict Number of assigned weights, separated by weights
                                 Useful for re-construction of the matrix    
    """

    # - Matrix with weights. First row corresponds to input weights, others to reservoir matrix
    mfWRes, mnCount, dmnCount = DynapseConform(
        tupShape= (nSize, nSize),
        fConnectivity=fConnectivity,
        fRatioExc=fRatioExcRes,
        nLimitInputs=nLimitInputs - int(bLeaveSpaceForInput),
        nLimitOutputs=nLimitOutputs,
        tupfWExc=tupfWExc,
        tupfWInh=tupfWInh,
        tupfProbWExc=tupfProbWExcRes,
        tupfProbWInh=tupfProbWInhRes,
        fNormalize=fNormalize,
    )

    # - For each reservoir neuron, check whether it still has space for an input
    vbFree = (nLimitInputs - np.sum(mnCount, axis=0)) > 0
    # - Total number of input weights to be assigned
    nNumInputs = (min(np.sum(vbFree), int(fInputDensity*nSize))
        if fInputDensity is not None else np.sum(vbFree)
    )

    if nNumInputs == 0:
        print("WARNING: Could not assign any input weights, "
            + "try setting bLeaveSpaceForInput=True or reducing fConnectivity"
        )
        return np.zeros((nSize, 1)), mfWRes, mnCount, dmnCount
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
    vfWeights[ : nNumExcIn] = np.random.choice(tupfWInExc, size=nNumExcIn, p=tupfProbWExcIn, replace=True)
    # - Generate inhibitory weights
    vfWeights[nNumExcIn : ] = np.random.choice(tupfWInInh, size=nNumInhIn, p=tupfProbWInhIn, replace=True)
    # - Ids of neurons to which weights will be assigned
    viNeurons = np.random.choice(np.where(vbFree)[0], size=nNumInputs, replace=False)
    # - Input weight vector
    vfWIn = np.zeros(nSize)
    for iNeuron, fWeight in zip(viNeurons, vfWeights):
        vfWIn[iNeuron] = fWeight

    return vfWIn.reshape(-1,1), mfWRes, mnCount, dmnCount

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
    mfW = (
        np.random.randn(nResSize ** 2) * np.asarray(fStd) + np.asarray(fMean)
    ) * vbConnection
    return mfW.reshape(nResSize, nResSize)
