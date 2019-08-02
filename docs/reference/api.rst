Full API summary for NetworksPython
===================================

.. py:currentmodule::NetworksPython

Main classes
------------

.. seealso:: :ref:`gettingstarted`, :ref:`basicexamples` and :ref:`networkdocs`

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    network.Network
    layers.Layer


Time series classes
-----------------------------

.. seealso:: :ref:`timeseriesdocs`

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    timeseries.TimeSeries
    timeseries.TSContinuous
    timeseries.TSEvent


:py:class:`Layer` classes
-------------------------

.. seealso:: :ref:`layersdocs` and :ref:`layerssummary`

.. autosummary::
    :toctree: _autosummary
    :template: class.rst

    layers.RecRateEuler
    layers.FFRateEuler

    layers.FFIAFBrian
    layers.FFIAFSpkInBrian
    layers.RecIAFBrian
    layers.RecIAFSpkInBrian
    layers.PassThroughEvents
    layers.FFExpSynBrian
    layers.FFExpSyn
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
    layers.RecRateEulerJax
    layers.ForceRateEulerJax
