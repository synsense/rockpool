"""
DynapSE-family device simulations, deployment and HDK support

JAX-backend Dynap-SE1/SE2 simulator support which allows people to 

* Configure their own networks either custom configuration or importing from existing device configuration 
* Run a simulation with any spiking input
* Observe the current changes through time
* Optimize the circuit level bias parameters using standard gradient-based optimization techniques
* Export the simulation configuration in the form of a device configuraiton for deployment

"""

from . import samna_alias as sa

from .simulation import DynapSim, dynapse_mismatch_generator
from .mapping import DynapseNeurons, mapper
from .parameters import parameter_clustering, BiasGenSE2, DynapSimCore
from .quantization import autoencoder_quantization
from .hardware import (
    config_from_specification,
    DynapseSamna,
    find_dynapse_boards,
    dynapsim_from_config,
)
