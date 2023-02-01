"""
Dynap-SE2 weight bit current parameters

* Non User Facing *
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
    """

    Iw_0: FloatVector = default_weights["Iw_0"]
    """weight bit 0 current of the neurons of the core in Amperes"""

    Iw_1: FloatVector = default_weights["Iw_1"]
    """weight bit 1 current of the neurons of the core in Amperes"""

    Iw_2: FloatVector = default_weights["Iw_2"]
    """weight bit 2 current of the neurons of the core in Amperes"""

    Iw_3: FloatVector = default_weights["Iw_3"]
    """weight bit 3 current of the neurons of the core in Amperes"""

    @property
    def Iw(self) -> np.ndarray:
        """Weight bits stacked together"""
        return np.stack([self.Iw_0, self.Iw_1, self.Iw_2, self.Iw_3]).T
