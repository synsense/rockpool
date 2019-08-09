.. _layerssummary:

Types of `Layer` available in |project|
=======================================

Rate-based non-spiking layers
-----------------------------

.. autosummary::
    layers.PassThrough
    layers.FFRateEuler
    layers.RecRateEuler

JAX-based backend
~~~~~~~~~~~~~~~~~

.. autosummary::
    layers.RecRateEulerJax
    layers.ForceRateEulerJax


Event-driven spiking layers
---------------------------

.. autosummary::
    layers.PassThroughEvents
    layers.FFExpSyn
    layers.RecDIAF
    layers.RecFSSpikeEulerBT
    layers.FFUpDown


Layers with constant leak
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    layers.CLIAF
    layers.FFCLIAF
    layers.RecCLIAF
    layers.SoftMaxLayer

Brian-based backend
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    layers.FFIAFBrian
    layers.FFIAFSpkInBrian
    layers.RecIAFBrian
    layers.RecIAFSpkInBrian
    layers.FFExpSynBrian


Torch-based backend
~~~~~~~~~~~~~~~~~~~

.. autosummary::
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

Nest-based backend
~~~~~~~~~~~~~~~~~~

.. autosummary::

    layers.FFIAFNest
    layers.RecIAFSpkInNest
    layers.RecAEIFSpkInNest


Hardware-backed and hardware simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more information on using these layers, see :ref:`/tutorials/RecDynapSE.ipynb`

.. autosummary::

    layers.RecDynapSE
    layers.VirtualDynapse
