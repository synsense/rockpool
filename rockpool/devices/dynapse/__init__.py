"""
DynapSE-family device simulations, deployment and HDK support

JAX-backend Dynap-SE1/SE2 simulator support which allows people to 

* Configure their own networks either custom configuration or importing from existing device configuration 
* Run a simulation with any spiking input
* Observe the current changes through time
* Optimize the circuit level bias parameters using standard gradient-based optimization techniques
* Export the simulation configuration in the form of a device configuraiton for deployment

"""
from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

try:
    from .virtual_dynapse import *
except:
    if not backend_available("numpy", "nest"):
        VirtualDynapse = missing_backend_shim("VirtualDynapse", "numpy, nest")
