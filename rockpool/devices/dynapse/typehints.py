"""
Dynap-SE2 common typehint definitions
"""

from typing import Tuple
import numpy as np

__all__ = ["NeuronKey", "CoreKey", "DynapSimRecord", "DynapSimState"]

NeuronKey = Tuple[np.uint8, np.uint8, np.uint16]
CoreKey = Tuple[np.uint8, np.uint8]

DynapSimRecord = Tuple[
    np.ndarray,  # iahp
    np.ndarray,  # iampa
    np.ndarray,  # igaba
    np.ndarray,  # imem
    np.ndarray,  # inmda
    np.ndarray,  # ishunt
    np.ndarray,  # spikes
    np.ndarray,  # vmem
]


DynapSimState = Tuple[
    np.ndarray,  # iahp
    np.ndarray,  # iampa
    np.ndarray,  # igaba
    np.ndarray,  # imem
    np.ndarray,  # inmda
    np.ndarray,  # ishunt
    np.ndarray,  # rng_key
    np.ndarray,  # spikes
    np.ndarray,  # timer_ref
    np.ndarray,  # vmem
]
