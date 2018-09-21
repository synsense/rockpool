###
# net_rate_reservoir.py - Classes and functions for encapsulating simple rate-based reservoirs
###

from ..network import Network
from ...layers import FFRateEuler, RecRateEuler, PassThrough

from typing import Union, List
import numpy as np

ArrayLike = Union[List, np.ndarray, float, int]


def BuildRateReservoir(
    mfWInput: ArrayLike,
    mfWRes: ArrayLike,
    mfWOutput: ArrayLike,
    vtTauInput: ArrayLike = 1.,
    vtTauRes: ArrayLike = 1.,
    vfBiasInput: ArrayLike = 0.,
    vfBiasRes: ArrayLike = 0.,
    tDt: float = 1. / 10.,
    fNoiseStdInput: float = 0.,
    fNoiseStdRes: float = 0.,
    fNoiseStdOut: float = 0.,
):
    """
    BuildRateReservoir - Build a rate-based reservoir network, with the defined weights

    :param mfWInput:
    :param mfWRes:
    :param mfWOutput:
    :param vtTauInput:
    :param vtTauRes:
    :param vfBiasInput:
    :param vfBiasRes:
    :return:            Network A reservoir network
    """

    # - Build the input layer
    lyrInput = FFRateEuler(
        mfW=mfWInput,
        vtTau=vtTauInput,
        vfBias=vfBiasInput,
        tDt=tDt,
        fNoiseStd=fNoiseStdInput,
        strName="Input",
    )

    # - Build the recurrent layer
    lyrRes = RecRateEuler(
        mfW=mfWRes,
        vtTau=vtTauRes,
        vfBias=vfBiasRes,
        tDt=tDt,
        fNoiseStd=fNoiseStdRes,
        strName="Reservoir",
    )

    # - Build the output layer
    lyrOut = PassThrough(
        mfW=mfWOutput, tDt=tDt, fNoiseStd=fNoiseStdOut, strName="Readout"
    )

    # - Return the network
    return Network(lyrInput, lyrRes, lyrOut)


def BuildRandomReservoir(
    nInputSize: int = 1,
    nReservoirSize: int = 100,
    nOutputSize: int = 1,
    fInputWeightStd: float = 1.,
    fResWeightStd: float = 1.,
    fOutputWeightStd: float = 1.,
    fInputWeightMean: float = 0.,
    fResWeightMean: float = 0.,
    fOutputWeightMean: float = 0.,
    vtTauInput: ArrayLike = 1.,
    vtTauRes: ArrayLike = 1.,
    vfBiasInput: ArrayLike = 0.,
    vfBiasRes: ArrayLike = 0.,
    tDt: float = 1. / 10.,
    fNoiseStdInput: float = None,
    fNoiseStdRes: float = None,
    fNoiseStdOut: float = None,
):
    """
    BuildRandomReservoir - Build a randomly-generated reservoir network

    :param nInputSize:
    :param nReservoirSize:
    :param nOutputSize:
    :param fInputWeightStd:
    :param fResWeightStd:
    :param fOutputWeightStd:
    :param fInputWeightMean:
    :param fResWeightMean:
    :param fOutputWeightMean:
    :param vtTauInput:
    :param vtTauRes:
    :param vfBiasInput:
    :param vfBiasRes:
    :param tDt:
    :param fNoiseStdInput:
    :param fNoiseStdRes:
    :param fNoiseStdOut:
    :return:
    """

    # - Generate weights
    mfWInput = np.random.normal(
        fInputWeightMean, fInputWeightStd, (nInputSize, nReservoirSize)
    )
    mfWRes = np.random.normal(
        fResWeightMean, fResWeightStd, (nReservoirSize, nReservoirSize)
    )
    mfWOutput = np.random.normal(
        fOutputWeightMean, fOutputWeightStd, (nReservoirSize, nOutputSize)
    )

    # - Build reservoir network
    return BuildRandomReservoir(
        mfWInput,
        mfWRes,
        mfWOutput,
        vtTauInput,
        vtTauRes,
        vfBiasInput,
        vfBiasRes,
        tDt,
        fNoiseStdInput,
        fNoiseStdRes,
        fNoiseStdOut,
    )
