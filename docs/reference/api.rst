Full API summary for |project|
==============================

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

    parameters.Parameter
    parameters.State
    parameters.SimulationParameter


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

.. .. seealso:: :ref:`layerssummary`, :ref:`/tutorials/building_reservoir.ipynb` and other tutorials.

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    nn.modules.RateEulerJax
    nn.modules.LIFJax
    nn.modules.LIFTorch
    nn.modules.LIFNeuronTorch
    nn.modules.UpDownTorch

    nn.modules.Linear
    nn.modules.LinearJax
    nn.modules.LinearTorch

    nn.modules.Instant
    nn.modules.InstantJax

    nn.modules.ExpSmoothJax
    nn.modules.ExpSynTorch

    nn.modules.SoftmaxJax
    nn.modules.LogSoftmaxJax

:py:class:`Layer` subclasses from Rockpool v1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. seealso:: :ref:`layerssummary`, :ref:`/tutorials/building_reservoir.ipynb` and other tutorials.

.. autosummary::
    :toctree: _autosummary
    :template: class.rst


    nn.layers.RecRateEuler
    nn.layers.FFRateEuler
    nn.layers.PassThrough

    nn.layers.ButterFilter
    nn.layers.ButterMelFilter

    nn.layers.FFIAFBrian
    nn.layers.FFIAFSpkInBrian
    nn.layers.RecIAFBrian
    nn.layers.RecIAFSpkInBrian
    nn.layers.PassThroughEvents
    nn.layers.FFExpSynBrian
    nn.layers.RecDIAF
    nn.layers.FFUpDown

    nn.layers.FFIAFNest
    nn.layers.RecIAFSpkInNest
    nn.layers.RecAEIFSpkInNest

    nn.layers.FFCLIAF
    nn.layers.RecCLIAF

    .. nn.layers.RecDynapSE
    .. nn.layers.VirtualDynapse
    .. nn.layers.RecFSSpikeEulerBT
    .. nn.layers.RecFSSpikeADS
    .. nn.layers.RecRateEulerJax
    .. nn.layers.RecRateEulerJax_IO
    .. nn.layers.FFRateEulerJax
    .. nn.layers.ForceRateEulerJax_IO
    .. nn.layers.FFExpSynTorch
    .. nn.layers.FFIAFTorch
    .. nn.layers.FFIAFRefrTorch
    .. nn.layers.FFIAFSpkInTorch
    .. nn.layers.FFIAFSpkInRefrTorch
    .. nn.layers.RecIAFTorch
    .. nn.layers.RecIAFRefrTorch
    .. nn.layers.RecIAFSpkInTorch
    .. nn.layers.RecIAFSpkInRefrTorch
    .. nn.layers.RecIAFSpkInRefrCLTorch
    .. nn.layers.CLIAF
    .. nn.layers.SoftMaxLayer
    .. nn.layers.FFExpSyn
    .. nn.layers.RecLIFJax
    .. nn.layers.RecLIFCurrentInJax
    .. nn.layers.RecLIFJax_IO
    .. nn.layers.RecLIFCurrentInJax_IO
    .. nn.layers.FFLIFJax_IO
    .. nn.layers.FFLIFCurrentInJax_SO
    .. nn.layers.FFExpSynCurrentInJax
    .. nn.layers.FFExpSynJax

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
---------------------------

.. autosummary::
    :toctree: _autosummary
    :recursive:
    :template: module.rst

    training.torch_loss

Xylo hardware support and simulation
------------------------------------

.. autosummary::
    :toctree: _autosummary

    devices.xylo.config_from_specification
    devices.xylo.load_config
    devices.xylo.save_config

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.XyloCim
    devices.xylo.XyloSamna
    devices.xylo.AFE
    devices.xylo.DivisiveNormalisation

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    devices.xylo
    devices.xylo.xylo_devkit_utils

.. autosummary::
    :toctree: _autosummary

    devices.xylo.mapper

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.xylo.XyloHiddenNeurons
    devices.xylo.XyloOutputNeurons

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
