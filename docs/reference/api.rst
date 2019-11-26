Full API summary for |project|
==============================

.. py:currentmodule::rockpool

Base classes
------------

.. seealso:: :ref:`/basics/getting_started.ipynb` and :ref:`/basics/time_series.ipynb`.

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    networks.Network
    layers.Layer

Layer and Network alternative base classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    networks.NetworkDeneve
    layers.training.RRTrainedLayer


Time series classes
-------------------

.. seealso:: :ref:`/basics/time_series.ipynb`.

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    timeseries.TimeSeries
    timeseries.TSContinuous
    timeseries.TSEvent

Utility modules
---------------

:ref:`/reference/weights.rst` provides several useful functions for generating network weights.

:ref:`/reference/utils.rst` provides several useful utility functions.


:py:class:`Layer` subclasses
-----------------------------

.. seealso:: :ref:`layerssummary`, :ref:`/tutorials/building_reservoir.ipynb` and other tutorials.

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    layers.RecRateEuler
    layers.FFRateEuler
    layers.PassThrough

    layers.FFIAFBrian
    layers.FFIAFSpkInBrian
    layers.RecIAFBrian
    layers.RecIAFSpkInBrian
    layers.PassThroughEvents
    layers.FFExpSynBrian
    layers.FFExpSyn
    layers.RecLIFJax
    layers.RecLIFCurrentInJax
    layers.RecLIFJax_IO
    layers.RecLIFCurrentInJax_IO
    layers.FFCLIAF
    layers.RecCLIAF
    layers.CLIAF
    layers.SoftMaxLayer
    layers.RecDIAF
    layers.RecFSSpikeEulerBT
    layers.FFUpDown
    layers.FFExpSynTorch
    layers.FFIAFTorch
    layers.FFIAFRefrTorch
    layers.FFIAFSpkInTorch
    layers.FFIAFSpkInRefrTorch
    layers.RecIAFTorch
    layers.RecIAFRefrTorch
    layers.RecIAFSpkInTorch
    layers.RecIAFSpkInRefrTorch
    layers.RecIAFSpkInRefrCLTorch

    layers.FFIAFNest
    layers.RecIAFSpkInNest
    layers.RecAEIFSpkInNest
    layers.RecDynapSE
    layers.VirtualDynapse
    layers.RecRateEulerJax
    layers.ForceRateEulerJax
