"""
Dynap-SE1/SE2 full board configuration classes and methods

renamed : dynapse1_simconfig.py -> simconfig.py @ 211208
split_from : simconfig.py -> layout.py @ 220114
split_from : simconfig.py -> circuits.py @ 220114
merged from : layout.py -> simcore.py @ 220505
merged from : circuits.py -> simcore.py @ 220505
merged from : board.py -> simcore.py @ 220531
renamed : simcore.py -> simconfig.py @ 220531

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
03/05/2022

[] TODO : Add r_spkthr to gain
[] TODO : add from_bias methods to samna aliases
"""

from dataclasses import dataclass

from rockpool.devices.dynapse.lookup import default_weights
from rockpool.typehints import FloatVector

import numpy as np

from ..base import DynapSimProperty

__all__ = ["DynapSimWeightBits"]


@dataclass
class DynapSimWeightBits(DynapSimProperty):
    """
    DynapSimWeightBits encapsulates weight bit current parameters of Dynap-SE chips

    :param Iw_0: weight bit 0 current of the neurons of the core in Amperes
    :type Iw_0: FloatVector
    :param Iw_1: weight bit 1 current of the neurons of the core in Amperes
    :type Iw_1: FloatVector
    :param Iw_2: weight bit 2 current of the neurons of the core in Amperes
    :type Iw_2: FloatVector
    :param Iw_3: weight bit 3 current of the neurons of the core in Amperes
    :type Iw_3: FloatVector
    """

    Iw_0: FloatVector = default_weights["Iw_0"]
    Iw_1: FloatVector = default_weights["Iw_1"]
    Iw_2: FloatVector = default_weights["Iw_2"]
    Iw_3: FloatVector = default_weights["Iw_3"]

    @property
    def Iw(self) -> np.ndarray:
        return np.stack([self.Iw_0, self.Iw_1, self.Iw_2, self.Iw_3]).T
