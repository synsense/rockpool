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
