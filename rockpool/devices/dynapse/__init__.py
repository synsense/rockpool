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

from .dynapsim import DynapSim
from .mapping import DynapseNeurons
from .parameters import parameter_clustering, BiasGenSE2, DynapSimCore
