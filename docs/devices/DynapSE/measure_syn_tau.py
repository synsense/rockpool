from dataclasses import dataclass
from typing import Tuple, List

from rockpool.devices.dynapse.config.simconfig import (
    DynapSE1SimBoard,
    DynapSE1SimCore,
    DynapSELayout,
    DynapSECapacitance,
    DPIParameters,
    SynapseParameters,
    MembraneParameters,
    AHPParameters,
    NMDAParameters,
    AMPAParameters,
    GABAAParameters,
    GABABParameters,
    WeightParameters,
)

from rockpool.devices.dynapse.infrastructure.biasgen import BiasGenSE2


@dataclass
class TestBase:
    SOIF_LEAK_N: Tuple[int] = (1, 200)  # done
    SOIF_GAIN_N: Tuple[int] = (5, 254)  # done
    SOIF_REFR_N: Tuple[int] = (3, 254)  # done
    SOIF_SPKTHR_P: Tuple[int] = (5, 254)  # done
    SOIF_CC_N: Tuple[int] = (4, 254)  # NOT USED!
    SOIF_DC_P: Tuple[int] = (0, 1)  # done
    SYAM_W0_P: Tuple[int] = (5, 254)  # done
    SYAM_W1_P: Tuple[int] = (4, 60)  # done
    SYAM_W2_P: Tuple[int] = (4, 80)  # done
    SYAM_W3_P: Tuple[int] = (4, 120)  # done

    def mem_params(self):
        return MembraneParameters(
            Imem=BiasGenSE2.get_bias(*self.SOIF_GAIN_N, "N", 1.05) * 3,
            Itau=BiasGenSE2.get_bias(*self.SOIF_LEAK_N, "N", 0.61),
            tau=None,
            Ith=BiasGenSE2.get_bias(*self.SOIF_GAIN_N, "N", 1.05),
            f_gain=None,
            Iref=BiasGenSE2.get_bias(*self.SOIF_REFR_N, "N", 1.05),
            t_ref=None,
            Ispkthr=BiasGenSE2.get_bias(*self.SOIF_SPKTHR_P, "N", 0.38),
            Idc=BiasGenSE2.get_bias(*self.SOIF_DC_P, "P", 0.385),
        )

    def weight_params(self):
        return WeightParameters(
            Iw_0=BiasGenSE2.get_bias(*self.SYAM_W0_P, "P", 0.22),
            Iw_1=BiasGenSE2.get_bias(*self.SYAM_W1_P, "P", 0.22),
            Iw_2=BiasGenSE2.get_bias(*self.SYAM_W2_P, "P", 0.22),
            Iw_3=BiasGenSE2.get_bias(*self.SYAM_W3_P, "P", 0.22),
        )

    def syn_params(self):
        raise NotImplementedError("Should be implemented synapse specific!")

    def sim_core(self):
        raise NotImplementedError("Should be implemented synapse specific!")

    def _sim_core(self, i, *args, **kwargs):
        return DynapSE1SimCore(
            size=1,
            neuron_idx_map={i: 1},
            mem=self.mem_params(),
            weights=self.weight_params(),
            ahp=AHPParameters(Iw=0.0),
            *args,
            **kwargs
        )


@dataclass
class AMPABase(TestBase):
    SOIF_GAIN_N: Tuple[int] = (5, 254)  # done
    DEAM_ETAU_P: Tuple[int] = (1, 140)  # done
    DEAM_EGAIN_P: Tuple[int] = (1, 20)  # done
    SYPD_EXT_N: Tuple[int] = (4, 80)  # NOT USED!

    def syn_params(self):
        return AMPAParameters(
            Isyn=BiasGenSE2.get_bias(*self.DEAM_EGAIN_P, "P", 0.68) * 3,
            Itau=BiasGenSE2.get_bias(*self.DEAM_ETAU_P, "P", 0.2),
            tau=None,
            Ith=BiasGenSE2.get_bias(*self.DEAM_EGAIN_P, "P", 0.68),
            f_gain=None,
        )

    def sim_core(self, i=0):
        return self._sim_core(i=i, ampa=self.syn_params())


@dataclass
class NMDABase(TestBase):
    SOIF_GAIN_N: Tuple[int] = (5, 254)
    DENM_ETAU_P: Tuple[int] = (1, 140)
    DENM_EGAIN_P: Tuple[int] = (1, 20)
    SYPD_EXT_N: Tuple[int] = (4, 80)

    def syn_params(self):
        return NMDAParameters(
            Itau=BiasGenSE2.get_bias(*self.DENM_ETAU_P, "P", 0.2),
            tau=None,
            Ith=BiasGenSE2.get_bias(*self.DENM_EGAIN_P, "P", 0.68),
            f_gain=None,
        )

    def sim_core(self, i=0):
        return self._sim_core(i=i, nmda=self.syn_params())


@dataclass
class GABABase(TestBase):
    SOIF_GAIN_N: Tuple[int] = (5, 80)
    DEGA_ITAU_P: Tuple[int] = (1, 140)
    DEGA_IGAIN_P: Tuple[int] = (1, 40)
    SYPD_EXT_N: Tuple[int] = (4, 170)

    def syn_params(self):
        return GABABParameters(
            Itau=BiasGenSE2.get_bias(*self.DEGA_ITAU_P, "P", 0.2),
            tau=None,
            Ith=BiasGenSE2.get_bias(*self.DEGA_IGAIN_P, "P", 0.68),
            f_gain=None,
        )

    def sim_core(self, i=0):
        return self._sim_core(i=i, gaba_b=self.syn_params())


@dataclass
class ShuntBase(TestBase):
    SOIF_GAIN_N: Tuple[int] = (5, 80)
    DESC_ITAU_P: Tuple[int] = (1, 140)
    DESC_IGAIN_P: Tuple[int] = (1, 40)
    SYPD_EXT_N: Tuple[int] = (4, 80)

    def syn_params(self):
        return GABAAParameters(
            Itau=BiasGenSE2.get_bias(*self.DESC_ITAU_P, "P", 0.2),
            tau=None,
            Ith=BiasGenSE2.get_bias(*self.DESC_IGAIN_P, "P", 0.68),
            f_gain=None,
        )

    def sim_core(self, i=0):
        return self._sim_core(i=i, gaba_a=self.syn_params())


def get_board(
    test_base: TestBase,
    syn_tau: List[Tuple[int]] = [(1, 240), (1, 120), (1, 80), (1, 60), (1, 48)],
):
    if hasattr(test_base, "DEAM_ETAU_P"):
        tau = "DEAM_ETAU_P"
    elif hasattr(test_base, "DENM_ETAU_P"):
        tau = "DENM_ETAU_P"
    elif hasattr(test_base, "DEGA_ITAU_P"):
        tau = "DEGA_ITAU_P"
    elif hasattr(test_base, "DESC_ITAU_P"):
        tau = "DESC_ITAU_P"
    else:
        raise AttributeError("ATTRIBUTE NOT FOUND!")

    DynapSE1SimCore.reset(DynapSE1SimCore)
    cores = []

    for i, TAU_P in enumerate(syn_tau):
        test_base.__setattr__(tau, TAU_P)
        cores.append(test_base.sim_core(i))

    sim_board = DynapSE1SimBoard(len(syn_tau), cores=cores)
    return sim_board


# def inh_test(
#     test_base: TestBase,
#     dc: List[Tuple[int]] = [(2, 30), (2, 35), (2, 40), (2, 45), (2, 50)],
#     syn_tau: List[Tuple[int]] = [(1, 240), (1, 120), (1, 80), (1, 60), (1, 48)],
# ):
#     if hasattr(test_base, "DEGA_ITAU_P"):
#         tau = "DEGA_ITAU_P"
#     elif hasattr(test_base, "DESC_ITAU_P"):
#         tau = "DESC_ITAU_P"
#     else:
#         raise AttributeError("GABA or SHUNT!")

#     for SOIF_DC_P in dc:
#         test_base.SOIF_DC_P = SOIF_DC_P
#         for ITAU_P in syn_tau:
#             test_base.__setattr__(tau, ITAU_P)
#             print(test_base)


# exc_test(AMPABase())
# inh_test(GABABase())

