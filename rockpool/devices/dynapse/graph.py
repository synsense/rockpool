"""
Dynap-SE graph modules implementing conversion and translation methods

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
"""
from typing import Optional
from rockpool.graph import GenericNeurons

from dataclasses import dataclass, field

__all__ = ["DynapseNeurons"]


@dataclass(eq=False, repr=False)
class DynapseNeurons(GenericNeurons):
    dt: Optional[float] = None
    """ float: The ``dt`` time step used for this neuron module """
    
