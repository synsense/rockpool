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


Layer and Network alternative base classes
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

    nn.layers.RecRateEuler
    nn.layers.FFRateEuler
    nn.layers.PassThrough

    nn.layers.FFIAFBrian
    nn.layers.FFIAFSpkInBrian
    nn.layers.RecIAFBrian
    nn.layers.RecIAFSpkInBrian
    nn.layers.PassThroughEvents
    nn.layers.FFExpSynBrian
    nn.layers.FFExpSyn
    nn.layers.RecLIFJax
    nn.layers.RecLIFCurrentInJax
    nn.layers.RecLIFJax_IO
    nn.layers.RecLIFCurrentInJax_IO
    nn.layers.FFLIFJax_IO
    nn.layers.FFLIFCurrentInJax_SO
    nn.layers.FFExpSynCurrentInJax
    nn.layers.FFExpSynJax
    nn.layers.RecDIAF
    nn.layers.RecFSSpikeEulerBT
    nn.layers.FFUpDown
    nn.layers.RecFSSpikeADS

    nn.layers.FFIAFNest
    nn.layers.RecIAFSpkInNest
    nn.layers.RecAEIFSpkInNest

    nn.layers.RecDynapSE
    nn.layers.VirtualDynapse

    nn.layers.RecRateEulerJax
    nn.layers.RecRateEulerJax_IO
    nn.layers.FFRateEulerJax
    nn.layers.ForceRateEulerJax_IO

    nn.layers.FFExpSynTorch
    nn.layers.FFIAFTorch
    nn.layers.FFIAFRefrTorch
    nn.layers.FFIAFSpkInTorch
    nn.layers.FFIAFSpkInRefrTorch
    nn.layers.RecIAFTorch
    nn.layers.RecIAFRefrTorch
    nn.layers.RecIAFSpkInTorch
    nn.layers.RecIAFSpkInRefrTorch
    nn.layers.RecIAFSpkInRefrCLTorch
    nn.layers.FFCLIAF
    nn.layers.RecCLIAF
    nn.layers.CLIAF
    nn.layers.SoftMaxLayer
