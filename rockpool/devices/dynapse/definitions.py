"""
Dynap-SE1 common definitions and methods

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
22/12/2021
"""

from typing import Dict, List, Tuple, Union
import numpy as np

from jax import numpy as jnp


ArrayLike = Union[np.ndarray, List, Tuple]
NeuronKey = Tuple[np.uint8, np.uint8, np.uint16]
CoreKey = Tuple[np.uint8, np.uint8]

DynapSimRecord = Tuple[
    jnp.DeviceArray,  # iahp
    jnp.DeviceArray,  # iampa
    jnp.DeviceArray,  # igaba
    jnp.DeviceArray,  # imem
    jnp.DeviceArray,  # inmda
    jnp.DeviceArray,  # ishunt
    jnp.DeviceArray,  # spikes
    jnp.DeviceArray,  # vmem
]


DynapSimState = Tuple[
    jnp.DeviceArray,  # iahp
    jnp.DeviceArray,  # iampa
    jnp.DeviceArray,  # igaba
    jnp.DeviceArray,  # imem
    jnp.DeviceArray,  # inmda
    jnp.DeviceArray,  # ishunt
    jnp.DeviceArray,  # rng_key
    jnp.DeviceArray,  # spikes
    jnp.DeviceArray,  # timer_ref
    jnp.DeviceArray,  # vmem
]

WeightRecord = Tuple[
    jnp.DeviceArray,  # loss
    jnp.DeviceArray,  # w_en
    jnp.DeviceArray,  # w_dec
]

NUM_CHIPS = 4
NUM_CORES = 4
NUM_NEURONS = 256
NUM_SYNAPSES = 64
NUM_DESTINATION_TAGS = 4
NUM_POISSON_SOURCES = 1024


class DRCError(ValueError):
    pass


class DRCWarning(Warning, DRCError):
    pass
