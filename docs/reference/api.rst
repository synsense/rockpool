Full API summary for |project|
==============================

Package structure summary
-------------------------

.. py:currentmodule::rockpool

.. autosummary::
    :toctree: _autosummary
    :recursive:

    rockpool


Base classes
------------

.. seealso:: :ref:`/basics/getting_started.ipynb` and :ref:`/basics/time_series.ipynb`.

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.modules.Module
    nn.modules.TimedModule

Attribute types
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    parameters.ParameterBase
    parameters.Parameter
    parameters.State
    parameters.SimulationParameter

.. autosummary::
    :toctree: _autosummary

    parameters.Constant


Alternative base classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.modules.JaxModule
    nn.modules.TorchModule

Combinator modules
------------------

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.combinators.FFwdStack
    nn.combinators.Sequential
    nn.combinators.Residual


Time series classes
-------------------

.. seealso:: :ref:`/basics/time_series.ipynb`.

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    timeseries.TimeSeries
    timeseries.TSContinuous
    timeseries.TSEvent

:py:class:`Module` subclasses
-----------------------------

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.modules.Rate
    nn.modules.RateJax
    nn.modules.RateTorch

    nn.modules.LIF
    nn.modules.LIFJax
    nn.modules.LIFTorch

    nn.modules.aLIFTorch

    nn.modules.LIFNeuronTorch
    nn.modules.UpDownTorch

    nn.modules.Linear
    nn.modules.LinearJax
    nn.modules.LinearTorch

    nn.modules.Instant
    nn.modules.InstantJax
    nn.modules.InstantTorch

    nn.modules.ExpSyn
    nn.modules.ExpSynJax
    nn.modules.ExpSynTorch

    nn.modules.SoftmaxJax
    nn.modules.LogSoftmaxJax

    nn.modules.ButterMelFilter
    nn.modules.ButterFilter

    nn.modules.LIFExodus
    nn.modules.LIFMembraneExodus
    nn.modules.ExpSynExodus


:py:class:`Layer` subclasses from Rockpool v1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These classes are deprecated, but are still usable via the high-level API, until they are converted to the v2 API.

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.layers.Layer

    nn.layers.FFIAFBrian
    nn.layers.FFIAFSpkInBrian
    nn.layers.RecIAFBrian
    nn.layers.RecIAFSpkInBrian
    nn.layers.FFExpSynBrian


Standard networks
------------------

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.networks.WaveSenseNet


Conversion utilities
--------------------

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.modules.timed_module.TimedModuleWrapper
    nn.modules.timed_module.LayerToTimedModule
    nn.modules.timed_module.astimedmodule


``Jax`` training utilities
---------------------------

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    training.jax_loss
    training.adversarial_jax

.. autosummary::
    :toctree: _autosummary

    training.adversarial_jax.pga_attack
    training.adversarial_jax.adversarial_loss


``PyTorch`` training utilities
-------------------------------

.. autosummary::
    :toctree: _autosummary
    :recursive:
    :template: module.rst

    training.torch_loss

``PyTorch`` transformation API (beta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    :recursive:
    :template: module.rst

    transform.torch_transform


Xylo hardware support and simulation
------------------------------------

Support modules
~~~~~~~~~~~~~~~

.. autosummary:: 
    :toctree: _autosummary

    devices.xylo.find_xylo_hdks

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    devices.xylo

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    devices.xylo.syns61300
    devices.xylo.syns61201
    devices.xylo.syns65300
    devices.xylo.syns63300

.. autosummary::
    :toctree: _autosummary

    transform.quantize_methods.global_quantize
    transform.quantize_methods.channel_quantize


Xylo Audio support
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    devices.xylo.syns61201.mapper
    devices.xylo.syns61201.config_from_specification
    devices.xylo.syns61201.load_config
    devices.xylo.syns61201.save_config
    devices.xylo.syns61201.cycles_model
    devices.xylo.syns61201.est_clock_freq

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns61201.XyloSim
    devices.xylo.syns61201.XyloSamna
    devices.xylo.syns61201.XyloMonitor
    devices.xylo.syns61201.AFESim
    devices.xylo.syns61201.AFESamna
    devices.xylo.syns61201.DivisiveNormalisation
    devices.xylo.syns61201.Xylo2HiddenNeurons
    devices.xylo.syns61201.Xylo2OutputNeurons


Xylo IMU support
~~~~~~~~~~~~~~~~~~~

.. seealso::
    * :ref:`/devices/xylo-imu/xylo-imu-intro.ipynb`

.. autosummary::
    :toctree: _autosummary

    devices.xylo.syns63300.mapper
    devices.xylo.syns63300.config_from_specification
    devices.xylo.syns63300.load_config
    devices.xylo.syns63300.save_config
    devices.xylo.syns63300.cycles_model
    devices.xylo.syns63300.est_clock_freq

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns63300.XyloSim
    devices.xylo.syns63300.XyloSamna
    devices.xylo.syns63300.XyloIMUMonitor

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns63300.XyloIMUHiddenNeurons
    devices.xylo.syns63300.XyloIMUOutputNeurons

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns63300.XyloSamna
    devices.xylo.syns63300.XyloSim
    devices.xylo.syns63300.XyloIMUMonitor


**IMU Preprocessing Interface**

.. seealso::
    * :ref:`/devices/xylo-imu/imu-if.ipynb`

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns63300.IMUIFSim
    devices.xylo.syns63300.IMUIFSamna
    devices.xylo.syns63300.IMUData

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    devices.xylo.syns63300.imuif

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns63300.IMUIFSim
    devices.xylo.syns63300.imuif.RotationRemoval
    devices.xylo.syns63300.imuif.BandPassFilter
    devices.xylo.syns63300.imuif.FilterBank
    devices.xylo.syns63300.imuif.ScaleSpikeEncoder
    devices.xylo.syns63300.imuif.IAFSpikeEncoder

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns63300.Quantizer


Dynap-SE2 hardware support and simulation
-----------------------------------------

.. seealso::
    * :ref:`/devices/DynapSE/dynapse-overview.ipynb`
    * :ref:`/devices/DynapSE/post-training.ipynb`
    * :ref:`/devices/DynapSE/neuron-model.ipynb`
    * :ref:`/devices/DynapSE/jax-training.ipynb`

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    devices.dynapse

**Simulation**

.. autosummary::
    :toctree: _autosummary
    
    :template: module.rst
    devices.dynapse.simulation

    :template: class.rst
    devices.dynapse.DynapSim


**Mismatch**

.. autosummary::
    :toctree: _autosummary

    transform.mismatch_generator
    devices.dynapse.frozen_mismatch_prototype
    devices.dynapse.dynamic_mismatch_prototype

**Device to Simulation**

.. autosummary::
    :toctree: _autosummary

    devices.dynapse.mapper
    devices.dynapse.autoencoder_quantization
    devices.dynapse.config_from_specification

**Computer Interface**

.. autosummary::
    :toctree: _autosummary

    devices.dynapse.find_dynapse_boards

    :template: class.rst
    devices.dynapse.DynapseSamna

**Simulation to Device**

.. autosummary::
    :toctree: _autosummary

    devices.dynapse.dynapsim_net_from_spec
    devices.dynapse.dynapsim_net_from_config

**More**

.. autosummary::
    :toctree: _autosummary
    
    :template: class.rst
    devices.dynapse.DynapseNeurons

    :template: class.rst
    devices.dynapse.DynapSimCore

Graph tracing and mapping
-------------------------

Base modules

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    graph.GraphModuleBase
    graph.GraphModule
    graph.GraphNode
    graph.GraphHolder

.. autosummary::
    :toctree: _autosummary

    graph.graph_base.as_GraphHolder

Computational graph modules

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    graph.LinearWeights
    graph.GenericNeurons
    graph.AliasConnection
    graph.LIFNeuronWithSynsRealValue
    graph.RateNeuronWithSynsRealValue

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    graph.utils

General Utilities
-----------------

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    utilities.backend_management
    utilities.tree_utils
    utilities.jax_tree_utils
    utilities.type_handling

NIR import and export
---------------------

.. autosummary::
    :toctree: _autosummary

    rockpool.nn.modules.to_nir
    rockpool.nn.modules.from_nir
