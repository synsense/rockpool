"""
Dynap-SE2 Application Programming Interface (API)

This package provides an abstract Dynap-SE2 machine that operates in the same parameter space as Dynap-SE2 processors. 
More, all the required tools to convert a simulation setting to an hardware configuration and a hardware configuration to a simulation setting.

JAX-backend Dynap-SE2 simulator named `DynapSim` does not simulate the hardware numerically precisely but executes a fast and an approximate simulation.
It uses forward Euler updates to predict the time-dependent dynamics and solves the characteristic circuit transfer functions in time.
In principle, the device and the simulator do not react exactly the same to the same input.
Since two physical chips would not react the same as well, it should not create a problem in application development.
The networks to be deployed to a chip should be robust against parameter variations.
To stress this, `DynapSim` provides a mismatch simulation feature that can alter the parameter projection.

A user can 

* define a rockpool network 
* map this network to a hardware specification
* quantize the specified paramters
* obtain a samna configuration
* connect and configure a Dynap-SE2 chip 
* run a real-time, hardware simulation

```python
net = Sequential(
    LinearJax(shape=(Nin, Nrec), has_bias=False),
    DynapSim((Nrec, Nrec), has_rec=True, percent_mismatch=0.05),
)

spec = mapper(net.as_graph())
spec.update(autoencoder_quantization(**spec))
config = config_from_specification(**spec)

# Connect to device
se2_devices = find_dynapse_boards()
se2 = DynapseSamna(se2_devices[0], **config)
out, state, rec = se2(raster, record=True)
```

Or 
* resolve a samna configuration to obtain a rockpool network

```python
model : JaxModule = dynapsim_net_from_config(**config)
out, state, rec = model(raster, record=True)
```

Please check the tutorials provided in :ref:`/docs/devices/DynapSE` for more.
"""

from .simulation import DynapSim, frozen_mismatch_prototype, dynamic_mismatch_prototype
from .mapping import DynapseNeurons, mapper
from .parameters import DynapSimCore
from .quantization import autoencoder_quantization
from .hardware import DynapseSamna, find_dynapse_boards, config_from_specification
from .dynapsim_net import dynapsim_net_from_config, dynapsim_net_from_spec
