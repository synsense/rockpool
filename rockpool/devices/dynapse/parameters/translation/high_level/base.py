"""
Dynap-SE2 high level property conversion abstract base class

* Non User Facing *
"""

from __future__ import annotations
from typing import Any
from dataclasses import dataclass

DynapSimCore = Any


@dataclass
class DynapSimCoreHigh:
    """
    DynapSimCoreHigh is an abstract class to be used as a boiler-plate for high-level projection classes
    """

    @classmethod
    def from_DynapSimCore(cls, core: DynapSimCore) -> DynapSimCoreHigh:
        NotImplementedError("Abstract method not implemented!")

    def update_DynapSimCore(self, core: DynapSimCore) -> DynapSimCore:
        NotImplementedError("Abstract method not implemented!")
