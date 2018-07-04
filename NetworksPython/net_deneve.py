###
# net_deneve.py - Classes and functions for encapsulating Denève reservoirs
###

from .network import Network
from .layers.feedforward import PassThrough, FFExpSyn, FFExpSynBrian
from .layers.recurrent import RecFSSpikeEulerBT

import numpy as np

# - Configure exports
__all__ = ['NetworkDeneve']

class NetworkDeneve(Network):
    def __init__(self):
        # - Call super-class constructor
        super().__init__()

    @staticmethod
    def SolveLinearProblem(mfA: np.ndarray = None,
                           nNetSize: int = None,

                           mfGamma: np.ndarray = None,

                           tDt: float = None,

                           fMu: float = 1e-4,
                           fNu: float = 1e-3,

                           fNoiseStd: float = 0.0,

                           tTauN: float = 20e-3,
                           tTauSynFast: float = 1e-3,
                           tTauSynSlow: float = 100e-3,

                           vfVThresh: np.ndarray = -55e-3,
                           vfVRest: np.ndarray = -65e-3,

                           tRefractoryTime = -np.finfo(float).eps,
                           ):
        """
        SolveLinearProblem - Static method Direct implementation of a linear dynamical system

        :param mfA:             np.ndarray [PxP] Matrix defining linear dynamical system
        :param nNetSize:        int Desired size of recurrent reservoir layer (Default: 100)

        :param mfGamma:         np.ndarray [PxN] Input kernel (Default: 50 * Normal / nNetSize)

        :param tDt:             float Nominal time step (Default: 0.1 ms)

        :param fMu:             float Linear cost parameter (Default: 1e-4)
        :param fNu:             float Quadratic cost parameter (Default: 1e-3)

        :param fNoiseStd:       float Noise std. dev. (Default: 0)

        :param tTauN:           float Neuron membrane time constant (Default: 20 ms)
        :param tTauSynFast:     float Fast synaptic time constant (Default: 1 ms)
        :param tTauSynSlow:     float Slow synaptic time constant (Default: 100 ms)

        :param vfVThresh:       float Threshold membrane potential (Default: -55 mV)
        :param vfVRest:         float Rest potential (Default: -65 mV)
        :param tRefractoryTime: float Refractory time for neuron spiking (Default: 0 ms)

        :return: A configured NetworkDeneve object, containing input, reservoir and output layers
        """
        # - Get input parameters
        nJ = mfA.shape[0]

        # - Generate random input weights if not provided
        if mfGamma is None:
            # - Default net size
            if nNetSize is None:
                nNetSize = 100

            # - Create random input kernels
            mfGamma = np.random.randn(nJ, nNetSize)
            mfGamma /= np.sum(np.abs(mfGamma), 0, keepdims = True)
            mfGamma /= nNetSize
            mfGamma *= 50

        else:
            assert (nNetSize is None) or nNetSize == mfGamma.shape[1], \
                '`nNetSize` must match the size of `mfGamma`.'
            nNetSize = mfGamma.shape[1]

        # - Generate network
        lambda_d = np.asarray(1 / tTauSynSlow)

        vfT = (fNu * lambda_d + fMu * lambda_d ** 2 + np.sum(abs(mfGamma.T), -1, keepdims = True) ** 2) / 2

        Omega_f = mfGamma.T @ mfGamma + fMu * lambda_d ** 2 * np.identity(nNetSize)
        Omega_s = mfGamma.T @ (mfA + lambda_d * np.identity(nJ)) @ mfGamma

        # - Scale problem to arbitrary membrane potential ranges and time constants
        vfT_dash = _expand_to_size(vfVThresh, nNetSize)
        b = np.reshape(_expand_to_size(vfVRest, nNetSize), (nNetSize, -1))
        a = np.reshape(_expand_to_size(vfVThresh, nNetSize), (nNetSize, -1)) - b

        Gamma_dash = a * mfGamma.T / vfT

        Omega_f_dash = a * Omega_f / vfT
        Omega_s_dash = a * Omega_s / vfT

        # - Pull out reset voltage from fast synaptic weights
        vfVReset = vfT_dash - np.diag(Omega_f_dash)
        np.fill_diagonal(Omega_f_dash, 0)

        # - Scale Omega_f_dash by fast TC
        Omega_f_dash /= tTauSynFast

        # - Scale everything by membrane TC
        Omega_f_dash *= tTauN
        Omega_s_dash *= tTauN
        Gamma_dash *= tTauN

        # - Build and return network
        return NetworkDeneve.SpecifyNetwork(mfW_f = -Omega_f_dash, mfW_s = Omega_s_dash,
                       mfW_input = Gamma_dash.T, mfW_output = mfGamma.T,
                       tDt = tDt, fNoiseStd = fNoiseStd,
                       vfVRest = vfVRest, vfVReset = vfVReset, vfVThresh = vfVThresh,
                       vtTauN = tTauN, vtTauSynR_f = tTauSynFast, vtTauSynR_s = tTauSynSlow, tTauSynO = tTauSynSlow,
                       tRefractoryTime = tRefractoryTime,
                    )


    @staticmethod
    def SpecifyNetwork(mfW_f, mfW_s,
                       mfW_input, mfW_output,

                       tDt: float = None,
                       fNoiseStd: float = 0.0,

                       vfVThresh: np.ndarray = -55e-3,
                       vfVReset: np.ndarray = -65e-3,
                       vfVRest: np.ndarray = -65e-3,

                       vtTauN: float = 20e-3,
                       vtTauSynR_f: float = 1e-3,
                       vtTauSynR_s: float = 100e-3,
                       tTauSynO: float = 100e-3,

                       tRefractoryTime: float = -np.finfo(float).eps,
                    ):
        """
        SpecifyNetwork - Directly configure all layers of a reservoir

        :param mfW_f:       np.ndarray [NxN] Matrix of fast synaptic weights
        :param mfW_s:       np.ndarray [NxN] Matrix of slow synaptic weights
        :param mfW_input:   np.ndarray [LxN] Matrix of input kernels
        :param mfW_output:  np.ndarray [NxM] Matrix of output kernels

        :param tDt:         float Nominal time step
        :param fNoiseStd:   float Noise Std. Dev.

        :param vfVRest:     np.ndarray [Nx1] Vector of rest potentials (spiking layer)
        :param vfVReset:    np.ndarray [Nx1] Vector of reset potentials (spiking layer)
        :param vfVThresh:   np.ndarray [Nx1] Vector of threshold potentials (spiking layer)

        :param vtTauN:      float Neuron membrane time constant (spiking layer)
        :param vtTauSynR_f: float Fast recurrent synaptic time constant
        :param vtTauSynR_s: float Slow recurrent synaptic time constant
        :param tTauSynO:    float Synaptic time constant for output layer

        :param tRefractoryTime: float Refractory time for spiking layer

        :return:
        """
        # - Construct reservoir
        lyrReservoir = RecFSSpikeEulerBT(mfW_f, mfW_s, tDt = tDt, fNoiseStd = fNoiseStd,
                                         vtTauN = vtTauN, vtTauSynR_f = vtTauSynR_f, vtTauSynR_s = vtTauSynR_s,
                                         vfVThresh = vfVThresh, vfVRest = vfVRest,
                                         vfVReset = vfVReset, tRefractoryTime = tRefractoryTime,
                                         strName = 'Deneve_Reservoir')

        # - Ensure time step is consistent across layers
        if tDt is None:
            tDt = lyrReservoir.tDt

        # - Construct input layer
        lyrInput = PassThrough(mfW_input, tDt = tDt, fNoiseStd = fNoiseStd, strName = 'Input')


        # - Construct output layer
        lyrOutput = FFExpSyn(mfW_output, tDt = 0.1e-4, fNoiseStd = fNoiseStd, tTauSyn = tTauSynO, strName = 'Output')

        # - Build network
        netDeneve = NetworkDeneve()
        netDeneve.lyrInput = netDeneve.add_layer(lyrInput, bExternalInput = True)
        netDeneve.lyrRes = netDeneve.add_layer(lyrReservoir, lyrInput)
        netDeneve.lyrOutput = netDeneve.add_layer(lyrOutput, lyrReservoir)

        # - Return constructed network
        return netDeneve


def _expand_to_size(oInput,
                    nSize: int,
                    sVariableName: str = 'input') -> np.ndarray:
    """
    _expand_to_size: Replicate out a scalar to a desired size

    :param oInput:          scalar or array-like (N)
    :param nSize:           int Desired size to return (=N)
    :param sVariableName:   str Name of the variable to include in error messages
    :return:                np.ndarray (N) vector
    """
    if np.size(oInput) == 1:
        # - Expand input to vector
        oInput = np.repeat(oInput, nSize)

    assert np.size(oInput) == nSize, \
        '`{}` must be a scalar or have {} elements'.format(sVariableName, nSize)

    # - Return object of correct shape
    return np.reshape(oInput, nSize)
