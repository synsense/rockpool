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

    nn.layers.FFIAFNest
    nn.layers.RecIAFSpkInNest
    nn.layers.RecAEIFSpkInNest

Standard networks
------------------

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.networks.wavesense.WaveSenseNet
    nn.networks.wavesense.WaveBlock


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

.. autosummary::
    :toctree: _autosummary

    devices.xylo.syns61201.config_from_specification
    devices.xylo.syns61201.load_config
    devices.xylo.syns61201.save_config

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns61201.XyloSim
    devices.xylo.syns61201.XyloSamna
    devices.xylo.syns61201.XyloMonitor
    devices.xylo.syns61201.AFESim
    devices.xylo.syns61201.AFESamna
    devices.xylo.syns61201.DivisiveNormalisation

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

.. autosummary::
    :toctree: _autosummary

    devices.xylo.syns61201.mapper

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns61300.Xylo1HiddenNeurons
    devices.xylo.syns61300.Xylo1OutputNeurons

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.syns61201.Xylo2HiddenNeurons
    devices.xylo.syns61201.Xylo2OutputNeurons

Dynap-SE2 hardware support and simulation
-----------------------------------------

.. seealso::
    Tutorials:

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
    utilities.jax_tree_utils
    utilities.type_handling

