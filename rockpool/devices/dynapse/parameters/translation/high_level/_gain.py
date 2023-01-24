"""
Dynap-SE2 gain ratio conversion and computation utilities

* Non User Facing *
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, replace
from rockpool.devices.dynapse.lookup import default_gain_ratios

import numpy as np

from rockpool.typehints import FloatVector
from .high import DynapSimCoreHigh, DynapSimCore

__all__ = ["DynapSimGain"]


@dataclass
class DynapSimGain(DynapSimCoreHigh):
    """
    DynapSimGain stores the ratio between gain and tau current values
    """

    r_gain_ahp: FloatVector = default_gain_ratios["r_gain_ahp"]
    """spike frequency adaptation block gain ratio :math:`Igain_ahp/Itau_ahp`"""

    r_gain_ampa: FloatVector = default_gain_ratios["r_gain_ampa"]
    """excitatory AMPA synpse gain ratio :math:`Igain_ampa/Itau_ampa`"""

    r_gain_gaba: FloatVector = default_gain_ratios["r_gain_gaba"]
    """inhibitory GABA synpse gain ratio :math:`Igain_gaba/Itau_gaba `"""

    r_gain_nmda: FloatVector = default_gain_ratios["r_gain_nmda"]
    """excitatory NMDA synpse gain ratio :math:`Igain_nmda/Itau_nmda`"""

    r_gain_shunt: FloatVector = default_gain_ratios["r_gain_shunt"]
    """inhibitory SHUNT synpse gain ratio :math:`Igain_shunt/Itau_shunt`"""

    r_gain_mem: FloatVector = default_gain_ratios["r_gain_mem"]
    """neuron membrane gain ratio :math:`Igain_mem/Itau_mem`"""

    @classmethod
    def from_DynapSimCore(cls, core: DynapSimCore) -> DynapSimGain:
        """
        from_DynapSimCore is a class factory method using DynapSimCore object

        :param core: the `DynapSimCore` object contatining the current values setting the gain ratios
        :type core: DynapSimCore
        :return: a `DynapSimGain` object, that stores the gain ratios set by a `DynapSimCore`
        :rtype: DynapSimGain
        """
        _r_gain = lambda name: cls.ratio_gain(
            Igain=core.__getattribute__(f"Igain_{name}"),
            Itau=core.__getattribute__(f"Itau_{name}"),
        )

        # Construct the object
        _mod = cls(
            r_gain_ahp=_r_gain("ahp"),
            r_gain_ampa=_r_gain("ampa"),
            r_gain_gaba=_r_gain("gaba"),
            r_gain_nmda=_r_gain("nmda"),
            r_gain_shunt=_r_gain("shunt"),
            r_gain_mem=_r_gain("mem"),
        )
        return _mod

    def update_DynapSimCore(self, core: DynapSimCore) -> DynapSimCore:
        """
        update_DynapSimCore updates a `DynapSimCore` object using the defined attirbutes in `DynapSimGain` object
        It does not change the original core object and returns an updated copy

        :param core: a `DynapSimCore` object to be updated
        :type core: DynapSimCore
        :return: an updated copy of DynapSimCore object
        :rtype: DynapSimCore
        """
        _I_gain = lambda name: self.gain_current(
            Igain=_core.__getattribute__(f"Igain_{name}"),
            r_gain=self.__getattribute__(f"r_gain_{name}"),
            Itau=_core.__getattribute__(f"Itau_{name}"),
        )
        _core = replace(core)

        for syn in ["ahp", "ampa", "gaba", "nmda", "shunt", "mem"]:
            _core.__setattr__(f"Igain_{syn}", _I_gain(syn))

        return _core

    @staticmethod
    def ratio_gain(
        Igain: Optional[FloatVector], Itau: Optional[FloatVector]
    ) -> FloatVector:
        """
        ratio_gain checks the parameters and divide Igain by Itau

        :param Igain: any gain bias current in Amperes
        :type Igain: Optional[FloatVector]
        :param Itau: any leakage current in Amperes
        :type Itau: Optional[FloatVector]
        :return: the ratio between the currents if the currents are properly set
        :rtype: FloatVector
        """

        if (
            Itau is not None
            and (np.array(Itau) > 0).all()
            and Igain is not None
            and (np.array(Igain) > 0).all()
        ):
            return Igain / Itau
        else:
            return None

    @staticmethod
    def gain_current(
        Igain: Optional[FloatVector],
        r_gain: Optional[FloatVector],
        Itau: Optional[FloatVector],
    ) -> FloatVector:
        """
        gain_current checks the ratio and Itau to deduce Igain out of them

        :param Igain: any gain bias current in Amperes
        :type Igain: Optional[FloatVector]
        :param r_gain: the ratio between Igain and Itau
        :type r_gain: Optional[FloatVector]
        :param Itau: any leakage current in Amperes
        :type Itau: Optional[FloatVector]
        :return: the gain bias current Igain in Amperes obtained from r_gain and Itau
        :rtype: FloatVector
        """
        if r_gain is None:
            return Igain
        elif Itau is not None:
            return Itau * r_gain
        else:
            return Igain
