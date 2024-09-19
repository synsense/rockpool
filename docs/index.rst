Welcome to |project|
==============================

.. raw:: html

    <img src='_static/logo_Color_perspective.png' width=80% class='main-logo' />


|project| is a Python package for working with dynamical neural network architectures, particularly for designing event-driven
networks for Neuromorphic computing hardware. |project| provides a convenient interface for designing, training
and evaluating recurrent networks, which can operate both with continuous-time dynamics and event-driven dynamics.

|project| is an open-source project managed by SynSense.

.. toctree::
   :maxdepth: 1

   about

.. toctree::
   :maxdepth: 1
   :caption: The basics

   basics/installation
   basics/introduction_to_snns.ipynb
   basics/getting_started.ipynb
   basics/hello_MNIST.ipynb
   basics/time_series.ipynb
   basics/standard_modules.ipynb
   basics/sharp_points.ipynb

.. toctree::
   :maxdepth: 1
   :caption: In depth

   in-depth/api-low-level.ipynb
   in-depth/api-high-level.ipynb
   in-depth/api-functional.ipynb
   in-depth/jax-training.ipynb
   in-depth/torch-api.ipynb
   in-depth/torch-training.ipynb
   in-depth/howto-constrained-opt.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/sgd_recurrent_net.ipynb
   tutorials/jax_lif_sgd.ipynb
   tutorials/torch-training-spiking.ipynb

   tutorials/easter/easter-snn-images.ipynb
   tutorials/adversarial_training.ipynb

   tutorials/rockpool-shd.ipynb
   tutorials/wavesense_training.ipynb

   tutorials/synnet/synnet_architecture.ipynb

.. toctree::
   :hidden:

   tutorials/deneve_reservoirs.ipynb
   tutorials/network_ads_tutorial.ipynb
   tutorials/pytorch_lightning_mlflow.ipynb
   tutorials/pytorch_lightning_mlflow_spiking_MNIST.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Xylo™ inference processors

   devices/xylo-overview.ipynb
   devices/quick-xylo/deploy_to_xylo.ipynb
   devices/torch-training-spiking-for-xylo.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Xylo™ Audio

   devices/quick-xylo/xylo-audio-2-intro.ipynb
   devices/analog-frontend-example.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Xylo™Audio 3

   devices/xylo-a3/xylo-a3-intro.ipynb
   devices/xylo-a3/afesim.ipynb
   devices/xylo-a3/afesim_agc.ipynb
   devices/xylo-a3/afesim_pdm.ipynb

.. toctree::
   :hidden:

   devices/xylo-a3/Xylo_A3.ipynb
   devices/xylo-a3/Xylo_A3_agc.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Xylo™ IMU

   devices/xylo-imu/xylo-imu-intro.ipynb
   devices/xylo-imu/imu-if.ipynb
   devices/xylo-imu/configure_preprocessing.ipynb

.. toctree::
   :maxdepth: 1
   :caption: DYNAP-SE2 mixed-signal processor

   devices/DynapSE/dynapse-overview.ipynb
   devices/DynapSE/post-training.ipynb
   devices/DynapSE/neuron-model.ipynb
   devices/DynapSE/jax-training.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Advanced topics

   advanced/graph_overview.ipynb
   advanced/graph_mapping.ipynb
   advanced/nir_export_import.ipynb
   reference/params_types.ipynb
   reference/lif-benchmarks.ipynb
   reference/api
   advanced/CHANGELOG

.. toctree::
   :hidden:

   advanced/QuantTorch.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Developer documentation

   developer/UML-diagrams.ipynb
   developer/backend-management.ipynb
   devices/DynapSE/developer.ipynb
   developer/release_process.rst

* :ref:`genindex`
