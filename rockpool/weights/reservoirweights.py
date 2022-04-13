"""
Utility functions for generating and manipulating networks
"""

import copy
from typing import Callable, Optional, Tuple, Union
from copy import deepcopy
import random
import numpy as np
import scipy.stats as stats

from rockpool.utilities.property_arrays import ArrayLike


def combine_ff_rec_stack(weights_ff: np.ndarray, weights_rec: np.ndarray) -> np.ndarray:
    """
    combine_ff_rec_stack - Combine a FFwd and Recurrent weight matrix into a single recurrent weight matrix
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


def rndm_sparse_ei_net(
    res_size: int,
    connectivity: float = 1,
    rndm_weight_fct: Callable[[int], np.ndarray] = np.random.rand,
    partitioned: bool = False,
    ratio_exc: float = 0.5,
    scale_inh: float = 1,
    normalization: float = 0.95,
) -> np.ndarray:
    """
    rndm_sparse_ei_net - Return a (sparse) matrix defining reservoir weights

    :param res_size:        int Number of reservoir units
    :param connectivity:   float Ratio of non-zero weight matrix elements
                                  (must be between 0 and 1)
    :param rndm_weight_fct:          Function used to draw random weights. Must accept an integer n
                            as argument and return n positive values.
    :param partitioned:    bool  Partition weight matrix into excitatory and inhibitory
    :param ratio_exc:       float Ratio of excitatory weights (must be between 0 and 1)
    :param fInhiblScale:    float Scale of negative weights to positive weights
    :param normalization:      float If not None, matrix is normalized so that
                                  its spectral radius equals normalization
    :return:                np.ndarray Weight matrix
    """

    # - Make sure parameters are in correct range
    connectivity = np.clip(connectivity, 0, 1)
    ratio_exc = np.clip(ratio_exc, 0, 1)

    # - Number of non-zero elements in matrix
    nNumWeights = int(connectivity * res_size**2)

    # - Array for storing connection strengths
    mfWeights = np.zeros(res_size**2)
    # - Draw non-zero weights
    mfWeights[:nNumWeights] = rndm_weight_fct(nNumWeights)

    if not partitioned:
        # - Number inhibitory connections
        nNumInhWeights = int(nNumWeights * (1 - ratio_exc))
        # - Multiply inhibitory connections with -scale_inh
        mfWeights[:nNumInhWeights] *= -scale_inh

    # - Shuffle weights and bring array into 2D-form
    np.random.shuffle(mfWeights)
    mfWeights = mfWeights.reshape(res_size, res_size)

    if partitioned:
        # - Number of excitatory neurons
        nNumExcNeurons = int(res_size * ratio_exc)

        # - All rows with index > nNumExcNeurons correspond to inhibitory neurons
        #   Multiply corresponding elements with -1 and scale with scale_inh
        mfWeights[nNumExcNeurons:] *= -scale_inh

    if normalization is not None:
        # Normalize weights matrix so that its spectral radius = normalization
        vEigenvalues = np.linalg.eigvals(mfWeights)
        mfWeights *= normalization / np.amax(np.abs(vEigenvalues))

    return mfWeights


def rndm_ei_net(
    num_exc: int,
    num_inh: int,
    ratio_inh_exc: float = 1,
    rndm_weight_fct: Callable[[int], float] = lambda n: np.random.randn(n, n)
    / np.sqrt(n),
) -> np.ndarray:
    """
    rndm_ei_net - Generate a random nicely-tuned real-valued reservoir matrix
    :param num_exc:         Number of excitatory neurons in the network
    :param num_inh:         Number of inhibitory neurons in the network
    :param ratio_inh_exc:     Factor relating total inhibitory and excitatory weight (w_inh = ratio_inh_exc * w_exc) default: 1
    :param rndm_weight_fct:          Function used to draw initial random weights. Default: numpy.random.randn

    :return:                Network connectivity weight matrix
    """

    # - Generate base connectivity matrix
    weights = rndm_weight_fct(num_exc + num_inh)

    # - Enforce excitatory and inhibitory partitioning
    vnExc = range(num_exc)
    idcs_inh = [n + num_exc for n in range(num_inh)]

    mfWE = weights[vnExc, :]
    mfWE[mfWE < 0] = 0
    mfWE /= np.sum(mfWE, 0)

    mfWI = weights[idcs_inh, :]
    mfWI[mfWI > 0] = 0
    mfWI = mfWI / np.sum(mfWI, 0) * -np.abs(ratio_inh_exc)

    weights = np.concatenate((mfWE, mfWI), 0)

    return weights


def wilson_cowan_net(
    num_nodes: int,
    self_exc: float = 1,
    self_inh: float = 1,
    exc_sigma: float = 1,
    inh_sigma: float = 1,
    rndm_weight_fct: Callable[[int], float] = lambda n: np.random.randn(n, n)
    / np.sqrt(n),
) -> (np.ndarray, np.ndarray):
    """
    wilson_cowan_net - FUNCTION Define a Wilson-Cowan network of oscillators

    :param num_nodes: Number of (E+I) nodes in the network
    :param self_exc:  Strength of self-excitation autapse. Default 1.0
    :param self_inh:  Strength of self-inhibition autapse. Default 1.0
    :param exc_sigma: Sigma of random excitatory connections. Default 1.0
    :param inh_sigma: Sigma of random inhibitory connections. Default 1.0
    :param rndm_weight_fct:    Function used to draw random weights. Default: numpy.random.randn
    :return:          2N x 2N weight matrix weights
    """

    # - Check arguments, enforce reasonable defaults

    # - Build random matrices
    mfEE = (
        np.clip(rndm_weight_fct(num_nodes), 0, None) * exc_sigma
        + np.identity(num_nodes) * self_exc
    )
    mfIE = (
        np.clip(rndm_weight_fct(num_nodes), 0, None) * exc_sigma
        + np.identity(num_nodes) * self_exc
    )
    mfEI = np.clip(rndm_weight_fct(num_nodes), None, 0) * inh_sigma + np.identity(
        num_nodes
    ) * -np.abs(self_inh)
    mfII = np.clip(rndm_weight_fct(num_nodes), None, 0) * inh_sigma + np.identity(
        num_nodes
    ) * -np.abs(self_inh)

    # - Normalise matrix components
    fENorm = exc_sigma * stats.norm.pdf(0, 0, exc_sigma) + self_exc
    fINorm = inh_sigma * stats.norm.pdf(0, 0, inh_sigma) + np.abs(self_inh)
    mfEE = mfEE / np.sum(mfEE, 0) * fENorm
    mfIE = mfIE / np.sum(mfIE, 0) * fENorm
    mfEI = mfEI / np.sum(mfEI, 0) * -fINorm
    mfII = mfII / np.sum(mfII, 0) * -fINorm

    # - Compose weight matrix
    weights = np.concatenate(
        (np.concatenate((mfEE, mfIE)), np.concatenate((mfEI, mfII))), axis=1
    )

    return weights.T


def wipe_non_switiching_eigs(
    weights: np.ndarray, idcs_inh: np.ndarray = None, inh_tau_factor: float = 1
) -> np.ndarray:
    """
    wipe_non_switiching_eigs - Eliminate eigenvectors that do not lead to a partition switching

    :param weights:             Network weight matrix [N x N]
    :param idcs_inh:           Vector [M x 1] of indices of inhibitory neurons. Default: None
    :param inh_tau_factor:   Factor relating inhibitory and excitatory time constants. tau_i = f * tau_e, tau_e = 1 Default: 1
    :return:                (weights, mfJ) Weight matrix and estimated Jacobian
    """

    res_size = weights.shape[0]

    # - Compute Jacobian
    mfJ = weights - np.identity(res_size)

    if idcs_inh is not None:
        mfJ[idcs_inh, :] /= inh_tau_factor

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

    if idcs_inh is not None:
        mfWHat[idcs_inh, :] *= inh_tau_factor

    mfWHat += np.identity(res_size)

    # - Attempt to rescale weight matrix for optimal dynamics
    mfWHat *= inh_tau_factor * 30

    return mfWHat, mfJHat


def unit_lambda_net(res_size: int) -> np.ndarray:
    """
    unit_lambda_net - Generate a network from Norm(0, sqrt(N))

    :param res_size: Number of neurons in the network
    :return:    weight matrix
    """

    # - Draw from a Normal distribution
    weights = np.random.randn(res_size, res_size) / np.sqrt(res_size)
    return weights


def DiscretiseWeightMatrix(
    weights: np.ndarray,
    max_num_connections: int = 3,
    max_num_inputs: int = None,
    max_num_outputs: int = None,
) -> (np.ndarray, np.ndarray, float, float):
    """
    DiscretiseWeightMatrix - Discretise a weight matrix by strength

    :param weights:             an arbitrary real-valued weight matrix.
    :param max_num_connections: the integer maximum number of synaptic connections that may be made between
                            two neurons. Excitatory and inhibitory weights will be discretised
                            separately. Default: 3
    :param max_num_inputs:    integer Number of permissable inputs per neuron
    :param max_num_outputs:   integer Number of permissable outputs per neuron
    :return: (mfWD, mnNumConns, fEUnitary, fIUnitary)
    """

    # - Make a copy of the input
    weights = deepcopy(weights)
    mfWE = weights * (weights > 0)
    mfWI = weights * (weights < 0)

    # - Select top N inputs per neuron
    if max_num_outputs is not None:
        mfWE = np.array(
            [
                row
                * np.array(np.argsort(-row) < np.round(max_num_outputs / 2), "float")
                for row in mfWE.T
            ]
        ).T
        mfWI = np.array(
            [
                row
                * np.array(
                    np.argsort(-abs(row)) < np.round(max_num_outputs / 2), "float"
                )
                for row in mfWI.T
            ]
        ).T

    if max_num_inputs is not None:
        mfWE = np.array(
            [
                row * np.array(np.argsort(-row) < np.round(max_num_inputs / 2), "float")
                for row in mfWE
            ]
        )
        mfWI = np.array(
            [
                row
                * np.array(
                    np.argsort(-abs(row)) < np.round(max_num_inputs / 2), "float"
                )
                for row in mfWI
            ]
        )

    # - Determine unitary connection strengths. Clip at max absolute values
    fEUnitary = np.max(weights) / max_num_connections
    fIUnitary = np.max(-weights) / max_num_connections

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


def one_dim_exc_res(size, n_neighbour, zero_diagnoal=True):
    """one_dim_exc_res - Recurrent weight matrix where each neuron is connected
    to its n_neighbour nearest neighbours on a 1D grid.
    Only excitatory connections.
    """
    weights_res = np.zeros((size, size))
    nBound = int(np.floor(n_neighbour / 2))
    for iPost in range(size):
        weights_res[max(0, iPost - nBound) : min(iPost + nBound + 1, size), iPost] = 1
        if zero_diagnoal:
            weights_res[iPost, iPost] = 0

    return weights_res


def two_dim_exc_res(
    size: int,
    n_neighbour: int,
    width_neighbour: Union[float, Tuple[float, float]],
    grid_dim: Optional[Tuple[int, int]] = None,
    multiple_conn: bool = True,
):
    # - Determine width of connection probability distribution
    if isinstance(width_neighbour, int):
        # assume isotropic probability distribution
        width_neighbour = (width_neighbour, width_neighbour)

    # - Determine grid size
    if grid_dim is None:
        # - Square grid
        nGridLength = int(np.ceil(np.sqrt(size)))
        grid_dim = (nGridLength, nGridLength)
    else:
        # - Make sure grid is large enough
        assert grid_dim[0] * grid_dim[1] >= size, "Grid dimensions are too small."
        # - Make sure grid dimensions are integers
        assert isinstance(grid_dim[0], int) and isinstance(
            grid_dim[1], int
        ), "grid_dim must be tuple of two positive integers."

    # - Matrix for determining connection probability to any location on the grid relative to given neuron
    #   Given neuron location corresponds to [grid_dim[0], grid_dim[1]]
    vx = np.arange(-grid_dim[0] + 1, grid_dim[0])
    vy = np.arange(-grid_dim[1] + 1, grid_dim[1])
    mx, my = np.meshgrid(vx, vy)
    # Gaussian Distribution of connection probability
    mfPConnect = np.exp(
        -((mx / width_neighbour[0]) ** 2 + (my / width_neighbour[1]) ** 2)
    )
    # No self-connections
    mfPConnect[grid_dim[0] - 1, grid_dim[1] - 1] = 0

    # - Weight matrix
    mnW = np.zeros((size, size))

    # - Iterate over neurons and assign their presynaptic connections
    for n_id in range(size):
        # Neuron coordinates in 2D grid
        tupnGridIndex = np.unravel_index(n_id, grid_dim)
        # Probabilities to connect to other neureons
        vfPConnectThis = (
            mfPConnect[
                (grid_dim[0] - 1)
                - tupnGridIndex[0] : (2 * grid_dim[0] - 1)
                - tupnGridIndex[0],
                (grid_dim[1] - 1)
                - tupnGridIndex[1] : (2 * grid_dim[1] - 1)
                - tupnGridIndex[1],
            ]
        ).flatten()[:size]
        # Normalize probabilities
        vfPConnectThis /= np.sum(vfPConnectThis)
        # Choose connections
        viPreSyn = np.random.choice(
            size, size=n_neighbour, p=vfPConnectThis, replace=multiple_conn
        )
        for iPreSyn in viPreSyn:
            mnW[iPreSyn, n_id] += 1

    return mnW


def add_random_long_range(
    weights_res,
    num_long_range,
    avoid_existing: bool = False,
    multiple_conn: bool = True,
):
    assert (
        weights_res.shape[0] == weights_res.shape[1]
    ), "weights_res must be a square matrix"

    for iPost in range(weights_res.shape[0]):
        if avoid_existing:
            viFarNeurons = np.where(weights_res[:, iPost] == 0)[0]
        else:
            viFarNeurons = np.arange(weights_res.shape[0])
        # - Make sure diagonal elements are excluded
        viFarNeurons = viFarNeurons[viFarNeurons != iPost]
        viPreSynConnect = np.random.choice(
            viFarNeurons, size=num_long_range, replace=multiple_conn
        )
        for iPreSyn in viPreSynConnect:
            weights_res[iPreSyn, iPost] += 1
    return weights_res


def partitioned_2d_reservoir(
    size_in: int = 64,
    size_rec: int = 256,
    size_inhib: int = 64,
    max_fanin: int = 64,
    num_inp_to_rec: int = 16,
    num_rec_to_inhib: int = 64,
    num_inhib_to_rec: int = 16,
    num_rec_short: int = 24,
    num_rec_long: int = 8,
    width_neighbour: Union[float, Tuple[float, float]] = (16.0, 16.0),
    input_sparsity: float = 1.0,
    input_sparsity_type: Optional[str] = None,
):
    if max_fanin is not None:
        assert (
            num_inp_to_rec + num_rec_short + num_rec_long + num_inhib_to_rec
            <= max_fanin
        ) and num_inhib_to_rec <= max_fanin, (
            "Maximum number of presynaptic connections per neuron exceeded."
        )

    # - Input-To-Recurrent part
    mnWInToRec = inp_to_rec(
        size_in=size_in,
        size_rec=size_rec,
        num_inp_to_rec=num_inp_to_rec,
        input_sparsity=input_sparsity,
        input_sparsity_type=input_sparsity_type,
    )

    # - Recurrent-To-Inhibitory part
    mnWRecToInhib = np.zeros((size_rec, size_inhib))
    viPreSynConnect = np.random.choice(size_rec, size=num_rec_to_inhib * size_inhib)
    for iPreIndex, iPostIndex in zip(
        viPreSynConnect, np.repeat(np.arange(size_inhib), num_rec_to_inhib)
    ):
        mnWRecToInhib[iPreIndex, iPostIndex] += 1

    # - Inhibitory-To-Recurrent part
    mnWInhibToRec = np.zeros((size_inhib, size_rec))
    viPreSynConnect = np.random.choice(size_inhib, size=num_inhib_to_rec * size_rec)
    for iPreIndex, iPostIndex in zip(
        viPreSynConnect, np.repeat(np.arange(size_rec), num_inhib_to_rec)
    ):
        mnWInhibToRec[iPreIndex, iPostIndex] -= 1

    # - Recurrent short range connecitons
    mnWRec = two_dim_exc_res(
        size_rec, n_neighbour=num_rec_short, width_neighbour=width_neighbour
    )
    # - Add long range connections
    mnWRec = add_random_long_range(
        mnWRec, num_rec_long, avoid_existing=False, multiple_conn=True
    )

    # - Put matrix together
    nSizeTotal = size_in + size_rec + size_inhib
    mnW = np.zeros((nSizeTotal, nSizeTotal))
    mnW[:size_in, size_in : size_in + size_rec] = mnWInToRec
    mnW[size_in : size_in + size_rec, size_in : size_in + size_rec] = mnWRec
    mnW[size_in : size_in + size_rec, -size_inhib:] = mnWRecToInhib
    mnW[-size_inhib:, size_in : size_in + size_rec] = mnWInhibToRec

    return mnW


def inp_to_rec(
    size_in: int = 64,
    size_rec: int = 256,
    num_inp_to_rec: int = 16,
    input_sparsity: float = 1.0,
    input_sparsity_type: Optional[str] = None,
    allow_multiples: bool = True,
):
    """
    inp_to_rec - Create an integer weight matrix that serves as input weights to the
                 recurrent population of a reservoir
    :param size_in:     int Size of presynaptic layer
    :param size_rec:    int Size of postsynaptic (recurrent) layer
    :param num_inp_to_rec:   int  Number of non-zero input connections of those postsynaptic
                             neurons that have. Total number of connections is
                             size_rec * input_sparsity * num_inp_to_rec .
    :input_sparsity:    float Ratio of postsynaptic neurons that have non-zero connections
    :input_sparsity_type:  str or None:  If None, all postsynaptic neurons will have
                           non-zero input connections. `input_sparsity` will be ignored.
                           If "random", a random subset of postsynaptic neurons will have
                           non-zero input connections. If "first" it is the neurons with
                           lowest IDs.
    :allow_multiples: If True, multiple connections can be set between the same pair
                      of neurons. Corresponding to entries > 1 in weight matrix.
    return
        np.ndarray  Weight matrix (integer entries)
    """
    mnWInToRec = np.zeros((size_in, size_rec))
    if input_sparsity_type is None:
        num_receive_input = size_rec
        input_receivers = np.arange(size_rec)
    else:
        num_receive_input = int(np.round(input_sparsity * size_rec))
        if input_sparsity_type == "random":
            input_receivers = np.random.choice(
                size_rec, size=num_receive_input, replace=False
            )
        elif input_sparsity_type == "first":
            input_receivers = np.arange(num_receive_input)
        else:
            raise ValueError(
                f"Input sparsity type ({input_sparsity_type}) not recognized."
            )
    if allow_multiples:
        viPreSynConnect = np.random.choice(
            size_in, size=num_inp_to_rec * num_receive_input
        )
        for iPreIndex, iPostIndex in zip(
            viPreSynConnect, np.repeat(input_receivers, num_inp_to_rec)
        ):
            mnWInToRec[iPreIndex, iPostIndex] += 1
    else:
        for i_post in input_receivers:
            presyn_ids = np.random.choice(size_in, size=num_inp_to_rec, replace=False)
            for i_pre in presyn_ids:
                mnWInToRec[i_pre, i_post] = 1

    return mnWInToRec


def ring_reservoir(size_in: int = 64, size_rec: int = 256, num_inp_to_rec: int = 16):
    # - Random connections from input stage to recurrent stage
    presyn_neuron_ids = np.random.randint(size_in, size=(size_rec, num_inp_to_rec))
    conn_inp_rec = np.zeros((size_in, size_rec))
    conn_signs = np.random.choice((-1, 1), size=presyn_neuron_ids.shape)
    for i_post in range(size_rec):
        for i_pre, sign in zip(presyn_neuron_ids[i_post], conn_signs[i_post]):
            conn_inp_rec[i_pre, i_post] += sign
    # - Recurrent stage is one ring
    conn_rec = np.roll(np.eye(size_rec), 1, axis=0)
    # - Put everything together to a full connection matrix
    full_size = size_in + size_rec
    connections_full = np.zeros((full_size, full_size))
    connections_full[:size_in, size_in:] = conn_inp_rec
    connections_full[size_in:, size_in:] = conn_rec
    return connections_full


def dynapse_conform(
    shape,
    connectivity=None,
    ratio_exc=0.5,
    max_num_inputs=64,
    max_num_outputs=1024,
    weights_exc=(1, 2),
    weights_inh=(1, 2),
    probs_w_exc=(0.5, 0.5),
    probs_w_inh=(0.5, 0.5),
    normalization=None,
):
    """
    dynapse_conform - Create a weight matrix that conforms the specifications of the Dynapse Chip

    :param shape:        tuple Shape of the weight matrix
    :param connectivity:   float Ratio of non-zero vs. zero weights - limited by max_num_inputs/shape[0]
    :param ratio_exc:       float Ratio of excitatory vs. inhibitory synapses

    :param max_num_inputs:    int   Maximum fan-in per neuron
    :param max_num_outputs:   int   Maximum fan-out per neuron

    :param tupfWEcx:        tuple Possible strengths for excitatory synapses
    :param weights_inh:        tuple Possible strengths for inhibitory synapses
    :param tupfProbWEcx:    tuple Probabilities for excitatory synapse strengths
    :param probs_w_inh:    tuple Probabilities for inhibitory synapse strengths

    :param normalization:      float If not None, matrix will be normalized wrt spectral radius

    :return weights:            2D-np.ndarray Generated weight matrix
    :return mnCount:        2D-np.ndarray Number of assigned weights per connection
    :return dmnCount:       dict Number of assigned weights, separated by weights
                                 Useful for re-construction of the matrix
    """

    # - Make sure input weights all have correct sign
    weights_exc = tuple(abs(w) for w in weights_exc)
    weights_inh = tuple(-abs(w) for w in weights_inh)

    # - Determine size of matrix
    try:
        shape = tuple(shape)
        assert (
            len(shape) == 2 or len(shape) == 1
        ), "Only 2-dimensional matrices can be created."
        if len(shape) == 1:
            shape = (shape[0], shape[0])
    except TypeError:
        assert isinstance(shape, int), "shape must be integer or array-like of size 2."
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
    dmnCount = {
        weight: np.zeros_like(weights, int) for weight in (*weights_exc, *weights_inh)
    }

    # - Input synapses per neuron
    if connectivity is not None:
        nNumInputs = int(connectivity * weights.shape[0])
        assert (
            nNumInputs <= max_num_inputs
        ), "Maximum connectivity for given reservoir size and input limits is {}".format(
            float(max_num_inputs) / weights.shape[0]
        )
    else:
        nNumInputs = min(max_num_inputs, weights.shape[0])

    # - Numbers of excitatory and inhibitory inputs per neuron
    #   (could also be stochastic for each neuron....)
    nNumExcIn = int(np.round(nNumInputs * ratio_exc))
    nNumInhIn = nNumInputs - nNumExcIn

    # - Iterrate over neurons (columns of weights) and set their inputs
    #   Do so in random order. Otherwise small max_num_outputs could result in more
    #   deterministic columns towards the end if no more weights can be assigned anymore
    for iCol in np.random.permutation(weights.shape[1]):
        # - Array holding non-zero weights
        vfWeights = np.zeros(nNumInputs)
        # - Generate excitatory weights
        vfWeights[:nNumExcIn] = np.random.choice(
            weights_exc, size=nNumExcIn, p=probs_w_exc, replace=True
        )
        # - Generate inhibitory weights
        vfWeights[nNumExcIn:] = np.random.choice(
            weights_inh, size=nNumInhIn, p=probs_w_inh, replace=True
        )
        # - Shuffle, so that inhibitory and excitatory weights are equally likely
        np.random.shuffle(vfWeights)

        # - Count how many weights can still be set for each row without exceeding max_num_outputs
        vnFree = max_num_outputs - np.sum(mnCount, axis=1)
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

    if normalization is not None:
        # - Normalize matrix according to spectral radius
        vfEigenVals = np.linalg.eigvals(weights)
        fSpectralRad = np.amax(np.abs(vfEigenVals))
        if fSpectralRad == 0:
            print("Matrix is 0, will not normalize.")
        else:
            fScale = normalization / fSpectralRad
            weights *= fScale
            # - Also scale keys in dmnCount accordingly
            dmnCount = {fScale * k: v for k, v in dmnCount.items()}

    return weights, mnCount, dmnCount


def in_res_dynapse(
    size: int,
    input_density=1,
    connectivity=None,
    ratio_exc_rec=0.5,
    ratio_exc_in=0.5,
    max_num_inputs=64,
    max_num_outputs=1024,
    weights_exc=(1, 2),
    weights_inh=(1, 2),
    probs_w_exc_rec=(0.5, 0.5),
    probs_w_inh_rec=(0.5, 0.5),
    tupfProbWExcIn=(0.5, 0.5),
    tupfProbWInhIn=(0.5, 0.5),
    normalization=None,
    verbose=False,
    leave_space_for_input=False,
):
    """
    in_res_dynapse - Create input weights and recurrent weights for reservoir, respecting dynapse specifications

    :param size:           int Reservoir size
    :param input_density:   float Ratio of non-zero vs. zero input connections
    :param connectivity:   float Ratio of non-zero vs. zero weights - limited by max_num_inputs/shape[0]
    :param ratio_exc_rec:    float Ratio of excitatory vs. inhibitory recurrent synapses
    :param ratio_exc_in:     float Ratio of excitatory vs. inhibitory input synapses

    :param max_num_inputs:    int   Maximum fan-in per neuron
    :param max_num_outputs:   int   Maximum fan-out per neuron

    :param tupfWEcx:        tuple Possible strengths for excitatory synapses
    :param weights_inh:        tuple Possible strengths for inhibitory synapses
    :param tupfProbWEcxRes: tuple Probabilities for excitatory recurrent synapse strengths
    :param probs_w_inh_rec: tuple Probabilities for inhibitory recurrent synapse strengths
    :param tupfProbWEcxIn:  tuple Probabilities for excitatory input synapse strengths
    :param tupfProbWInhIn:  tuple Probabilities for inhibitory input synapse strengths

    :param normalization:      float If not None, matrix will be normalized wrt spectral radius

    :param verbose:        bool Currently unused

    :param leave_space_for_input: bool Limit number of input connections to ensure fan-in
                                     to include all all input weights

    :return weights_in:          2D-np.ndarray (Nx1) Generated input weights
    :return weights:            2D-np.ndarray (NxN) Generated weight matrix
    :return mnCount:        2D-np.ndarray (NxN) Number of assigned weights per connection
    :return dmnCount:       dict Number of assigned weights, separated by weights
                                 Useful for re-construction of the matrix
    """

    # - Matrix with weights. First row corresponds to input weights, others to reservoir matrix
    weights_res, mnCount, dmnCount = dynapse_conform(
        shape=(size, size),
        connectivity=connectivity,
        ratio_exc=ratio_exc_rec,
        max_num_inputs=max_num_inputs - int(leave_space_for_input),
        max_num_outputs=max_num_outputs,
        weights_exc=weights_exc,
        weights_inh=weights_inh,
        probs_w_exc=probs_w_exc_rec,
        probs_w_inh=probs_w_inh_rec,
        normalization=normalization,
    )

    # - For each reservoir neuron, check whether it still has space for an input
    vbFree = (max_num_inputs - np.sum(mnCount, axis=0)) > 0
    # - Total number of input weights to be assigned
    nNumInputs = (
        min(np.sum(vbFree), int(input_density * size))
        if input_density is not None
        else np.sum(vbFree)
    )

    if nNumInputs == 0:
        print(
            "WARNING: Could not assign any input weights, "
            + "try setting leave_space_for_input=True or reducing connectivity"
        )
        return np.zeros((size, 1)), weights_res, mnCount, dmnCount
    # - Number excitatory and inhibitory input weights
    nNumExcIn = int(nNumInputs * ratio_exc_in)
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


def in_res_dynapse_flex(
    size: int,
    size_in: None,
    max_fanin: int = 1,
    input_density=1,
    connectivity=None,
    ratio_exc_rec=0.5,
    ratio_exc_in=0.5,
    max_num_inputs=64,
    max_num_outputs=1024,
    weights_exc=(1, 1),
    weights_inh=(1, 1),
    probs_w_exc_rec=(0.5, 0.5),
    probs_w_inh_rec=(0.5, 0.5),
    tupfProbWExcIn=(0.5, 0.5),
    tupfProbWInhIn=(0.5, 0.5),
    normalization=None,
    verbose=False,
    leave_space_for_input=True,
):
    """
    in_res_dynapse_flex - Like in_res_dynapse but number of input weights can be chosen

    :param size:           int Reservoir size
    :param size_in:         int Number of neurons in input layer
    :param max_fanin:   int Number of presynaptic connections that are used for input connections per neuron
    :param input_density:   float Ratio of non-zero vs. zero input connections
    :param connectivity:   float Ratio of non-zero vs. zero weights - limited by max_num_inputs/shape[0]
    :param ratio_exc_rec:    float Ratio of excitatory vs. inhibitory recurrent synapses
    :param ratio_exc_in:     float Ratio of excitatory vs. inhibitory input synapses

    :param max_num_inputs:    int   Maximum fan-in per neuron
    :param max_num_outputs:   int   Maximum fan-out per neuron

    :param tupfWEcx:        tuple Possible strengths for excitatory synapses
    :param weights_inh:        tuple Possible strengths for inhibitory synapses
    :param tupfProbWEcxRes: tuple Probabilities for excitatory recurrent synapse strengths
    :param probs_w_inh_rec: tuple Probabilities for inhibitory recurrent synapse strengths
    :param tupfProbWEcxIn:  tuple Probabilities for excitatory input synapse strengths
    :param tupfProbWInhIn:  tuple Probabilities for inhibitory input synapse strengths

    :param normalization:      float If not None, matrix will be normalized wrt spectral radius

    :param verbose:        bool Currently unused

    :param leave_space_for_input: bool Limit number of input connections to ensure fan-in
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
    weights_res, mnCount, dmnCount = dynapse_conform(
        shape=(size, size),
        connectivity=connectivity,
        ratio_exc=ratio_exc_rec,
        max_num_inputs=max_num_inputs - int(leave_space_for_input) * max_fanin,
        max_num_outputs=max_num_outputs,
        weights_exc=weights_exc,
        weights_inh=weights_inh,
        probs_w_exc=probs_w_exc_rec,
        probs_w_inh=probs_w_inh_rec,
        normalization=normalization,
    )

    # - For each reservoir neuron, check whether it still has space for an input
    vnFree = max_num_inputs - np.sum(mnCount, axis=0)
    # - Total number of input weights to be assigned
    nNumInputs = (
        min(np.sum(vnFree), int(input_density * size * max_fanin))
        if input_density is not None
        else np.sum(vnFree)
    )

    if nNumInputs == 0:
        print(
            "WARNING: Could not assign any input weights, "
            + "try setting leave_space_for_input=True or reducing connectivity"
        )
        return np.zeros((size, 1)), weights_res, mnCount, dmnCount

    # - Number excitatory and inhibitory input weights
    nNumExcIn = int(nNumInputs * ratio_exc_in)
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
    connectivity=None,
    ratio_exc=0.5,
    bit_resolution=4,
    max_num_inputs=64,
    max_num_outputs=1024,
    use_range=1,
    normalization=None,
    f_rescale: Optional[float] = None,
):
    """
    digital - Create a weight matrix that conforms the specifications of the Dynapse Chip

    :param shape:        tuple Shape of the weight matrix
    :param connectivity:   float Ratio of non-zero vs. zero weights - limited by max_num_inputs/shape[0]
    :param ratio_exc:       float Ratio of excitatory vs. inhibitory synapses

    :param bit_resolution:  int   Weight resolution in bits. Before normalization, weights will
                                  be integers between -2**(bit_resolution/2) and 2**(bit_resolution/2)-1

    :param max_num_inputs:    int   Maximum fan-in per neuron
    :param max_num_outputs:   int   Maximum fan-out per neuron

    :param use_range:       float Ratio of the possible value range that is actually used. If smaller than
                                  than 1, input weights can be larger than recurrent weights, later on.

    :param normalization:      float If not None, matrix will be normalized wrt spectral radius
    :param f_rescale:        float If not None, multiply weights with this constant

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
        assert isinstance(shape, int), "shape must be integer or array-like of size 2."
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
    # dmnCount = {weight: np.zeros_like(mnW, int) for weight in (*weights_exc, *weights_inh)}

    # - Input synapses per neuron
    if connectivity is not None:
        nNumInputs = int(connectivity * mnW.shape[0])
        assert (
            nNumInputs <= max_num_inputs
        ), "Maximum connectivity for given reservoir size and input limits is {}".format(
            float(max_num_inputs) / mnW.shape[0]
        )
    else:
        nNumInputs = min(max_num_inputs, mnW.shape[0])

    # - Numbers of excitatory and inhibitory inputs per neuron
    #   (could also be stochastic for each neuron....)
    nNumExcIn = int(np.round(nNumInputs * ratio_exc))
    nNumInhIn = nNumInputs - nNumExcIn

    # - Determine value range of weights
    nMinWeight = int(-(2 ** (bit_resolution / 2)) * use_range)
    nMaxWeight = int(
        2 ** (bit_resolution / 2) * use_range
    )  ## Due to how np.random.randin works, max weight will be nMaxWeight-1

    # - Iterrate over neurons (columns of mnW) and set their inputs
    #   Do so in random order. Otherwise small max_num_outputs could result in more
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

        # - Count how many weights can still be set for each row without exceeding max_num_outputs
        vnFree = max_num_outputs - np.sum(mnCount, axis=1)
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

    if normalization is None:
        fScale = 1
    else:
        # - Normalize matrix according to spectral radius
        vfEigenVals = np.linalg.eigvals(mnW)
        fSpectralRad = np.amax(np.abs(vfEigenVals))
        fScale = normalization / fSpectralRad
        mnW *= fScale
        # - Also scale keys in dmnCount accordingly
        # dmnCount = {fScale * k: v for k, v in dmnCount.items()}
    if f_rescale is not None:
        # - Rescale weights by factor f_rescale
        mnW *= f_rescale
        fScale *= f_rescale

    return mnW, mnCount, fScale  # , dmnCount


def in_res_digital(
    size: int,
    input_density=1,
    connectivity=None,
    ratio_exc_rec=0.5,
    ratio_exc_in=0.5,
    bit_resolution=4,
    max_num_inputs=64,
    max_num_outputs=1024,
    ratio_rec_in=1,
    normalization=None,
    verbose=False,
    leave_space_for_input=False,
):
    """
    in_res_digital - Create input weights and recurrent weights for reservoir, respecting dynapse specifications

    :param size:           int Reservoir size
    :param input_density:   float Ratio of non-zero vs. zero input connections
    :param connectivity:   float Ratio of non-zero vs. zero weights - limited by max_num_inputs/shape[0]

    :param ratio_exc_rec:    float Ratio of excitatory vs. inhibitory recurrent synapses
    :param ratio_exc_in:     float Ratio of excitatory vs. inhibitory input synapses

    :param bit_resolution:  int   Weight resolution in bits. Before normalization, weights will
                                  be integers between -2**(bit_resolution/2) and 2**(bit_resolution/2)-1

    :param max_num_inputs:    int   Maximum fan-in per neuron
    :param max_num_outputs:   int   Maximum fan-out per neuron

    :param ratio_rec_in:     float Ratio of input vs recurrent weight ranges
                                  smaller than 1 leads to stronger input than recurrent weights
    :param normalization:      float If not None, matrix will be normalized wrt spectral radius

    :param verbose:        bool Currently unused

    :param leave_space_for_input: bool Limit number of input connections to ensure fan-in
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
        connectivity=connectivity,
        ratio_exc=ratio_exc_rec,
        bit_resolution=bit_resolution,
        max_num_inputs=max_num_inputs - int(leave_space_for_input),
        max_num_outputs=max_num_outputs,
        use_range=ratio_rec_in,
        normalization=normalization,
    )

    # - For each reservoir neuron, check whether it still has space for an input
    vbFree = (max_num_inputs - np.sum(mnCount, axis=0)) > 0
    # - Total number of input weights to be assigned
    nNumInputs = (
        min(np.sum(vbFree), int(input_density * size))
        if input_density is not None
        else np.sum(vbFree)
    )

    if nNumInputs == 0:
        print(
            "Weights: WARNING: Could not assign any input weights, "
            + "try setting leave_space_for_input=True or reducing connectivity"
        )
        return np.zeros((size, 1)), mnWRes, mnCount

    # - Number excitatory and inhibitory input weights
    nNumExcIn = int(nNumInputs * ratio_exc_in)
    nNumInhIn = nNumInputs - nNumExcIn
    # - Array holding non-zero weights
    vnWeights = np.zeros(nNumInputs)

    # - Determine value range of unnormalized weights
    nMinWeight = int(-(2**bit_resolution) / 2)
    nMaxWeight = int(2**bit_resolution / 2)
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


def iaf_sparse_net(
    res_size: int = 100, mean: float = None, std: float = None, density: float = 1.0
) -> np.ndarray:
    """
    iaf_sparse_net - Return a random sparse reservoir, scaled for a standard IAF spiking network

    :param res_size:    int Number of neurons in reservoir. Default: 100
    :param mean:       float Mean connection strength. Default: -4.5mV
    :param std:        float Std. Dev. of connection strengths. Default: 5.5mV
    :param density:    float 0..1 Density of connections. Default: 1 (full connectivity)
    :return:            [N x N] array Weight matrix
    """

    # - Check inputs
    assert (density >= 0.0) and (density <= 1.0), "`density` must be between 0 and 1."

    # - Set default values
    if mean is None:
        mean = -0.0045 / density / res_size

    if std is None:
        std = 0.0055 / np.sqrt(density)

    # - Determine sparsity
    nNumConnPerRow = int(density * res_size)
    nNumNonConnPerRow = res_size - nNumConnPerRow
    mbConnection = [
        random.sample([1] * nNumConnPerRow + [0] * nNumNonConnPerRow, res_size)
        for _ in range(res_size)
    ]
    vbConnection = np.array(mbConnection).reshape(-1)

    # - Randomise recurrent weights
    weights = (
        np.random.randn(res_size**2) * np.asarray(std) + np.asarray(mean)
    ) * vbConnection
    return weights.reshape(res_size, res_size)


def gen_sparse_partitioned_network(
    partition_sizes: ArrayLike,
    num_internal_inputs: int = 0,
    num_between_inputs: int = 0,
) -> np.ndarray:
    """
    Generate weight matrices that embody sparse networks with defined partition sizes

    :Example:

    >>> w, cl = gen_sparse_partitioned_network([3, 3, 3], 2, 1)
    >>> cl

    .. raw::

        [[0, [2, 0, 7]],
         [1, [2, 0, 8]],
         [2, [2, 1, 5]],
         [3, [3, 4, 6]],
         [4, [3, 5, 2]],
         [5, [4, 5, 6]],
         [8, [8, 7, 0]],
         [6, [8, 6, 4]],
         [7, [8, 6, 5]]]

    This will generate a network with three partitions, each with three elements. Each neuron will receive
    two inputs from within its partition, and one input from outside the partition.

    The total network size is the sum of the partition sizes. Only fan-in is limited; fan-out is not limited.

    :param ArrayLike[int] partition_sizes: List of partition sizes
    :param int num_internal_inputs:        Number of inputs to each neuron from within its partition
    :param int num_between_inputs:         Number of inputs to each neuron from outside its partition

    :return (ndarray, list): (weights, conn_lists)
        weights is an NxN matrix indexed as [source, dest], with `1` between connected neurons
        conn_lists is a list of lists, with each element [dest, list_sources]
    """
    # - Get total network size
    net_size = np.sum(partition_sizes)

    # - Make the weights matrix
    weights = np.zeros((net_size, net_size))

    # - Determine partition index beginnings
    partition_start = np.cumsum(np.concatenate(([0], partition_sizes)))
    partition_end = np.concatenate((partition_start[1:], [net_size]))

    # - Loop over partitions
    conn_list = []
    for part_size, start, end in zip(partition_sizes, partition_start, partition_end):
        # - Get the partition indices
        part_indices = set(range(start, end))
        nonpart_indices = set(range(net_size)) - part_indices

        # - For each neuron in the partition, find a set of random inputs
        for dest in part_indices:
            # - Get in-partition sources
            sources_in = list(copy.deepcopy(part_indices))
            shuffle(sources_in)
            sources_in = sources_in[:num_internal_inputs]

            # - Assign in-partition connections
            weights[sources_in, dest] = 1

            # - Get out-partition sources
            sources_out = list(copy.deepcopy(nonpart_indices))
            shuffle(sources_out)
            sources_out = sources_out[:num_between_inputs]

            # - Assign out-partition connections
            weights[sources_out, dest] = 1

            # - Append to connection lists
            conn_list.append([dest, list(np.concatenate((sources_in, sources_out)))])

    return weights, conn_list
