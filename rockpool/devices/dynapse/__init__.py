"""
Dynap-SE2 Application Programming Interface (API)

This package provides an abstract Dynap-SE2 machine that operates in the same parameter space as Dynap-SE2 processors. 
More, all the required tools to convert a simulation setting to an hardware configuration and a hardware configuration to a simulation setting.

It's possible to go from simulation to deployment:

* Define a rockpool network 
* Map this network to a hardware specification
* Quantize the parameters
* Obtain a samna configuration
* Connect and configure a Dynap-SE2 chip 
* Run a real-time, hardware simulation

..  code-block:: python
    :caption: Simulation -> Device (pseudo-code)

    # Define
    net = Sequential(LinearJax((Nin, Nrec)), DynapSim((Nrec, Nrec)))

    # Map
    spec = mapper(net.as_graph())
    spec.update(autoencoder_quantization(**spec))
    config = config_from_specification(**spec)

    # Connect 
    se2_devices = find_dynapse_boards()
    se2 = DynapseSamna(se2_devices[0], **config)
    out, state, rec = se2(raster, record=True)


It's also possible to go from hardware configuration to simulation:

..  code-block:: python
    :caption: Device -> Simulation (pseudo-code)

    net = dynapsim_net_from_config(**config)
    out, state, rec = net(raster, record=True)


See Also:
    See the tutorials

    * :ref:`/devices/DynapSE/dynapse-overview.ipynb`
    * :ref:`/devices/DynapSE/post-training.ipynb`
    * :ref:`/devices/DynapSE/neuron-model.ipynb`
    * :ref:`/devices/DynapSE/jax-training.ipynb`
"""

from .simulation import DynapSim, frozen_mismatch_prototype, dynamic_mismatch_prototype
from .mapping import DynapseNeurons, mapper
from .parameters import DynapSimCore
from .quantization import autoencoder_quantization
from .hardware import DynapseSamna, find_dynapse_boards, config_from_specification
from .dynapsim_net import dynapsim_net_from_config, dynapsim_net_from_spec
