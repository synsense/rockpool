Full API summary for |project|
==============================

.. py:currentmodule::rockpool

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

    nn.modules.Linear
    nn.modules.LinearJax
    nn.modules.LinearTorch

    nn.modules.Instant
    nn.modules.InstantJax

    nn.modules.AFE

    nn.modules.ExpSmoothJax

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
    :recursive:
    :template: module.rst

    training.jax_loss

Hardware support and simulation
-------------------------------

.. autosummary::
    :toctree: _autosummary

    devices.pollen.config_from_specification
    devices.pollen.load_config
    devices.pollen.save_config

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    devices.pollen.PollenCim
    devices.pollen.PollenSamna

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    devices.pollen.pollen_devkit_utils
