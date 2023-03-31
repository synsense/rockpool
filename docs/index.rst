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
   .. in-depth/howto-constrained-opt.ipynb

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
   
   .. tutorials/building_reservoir.ipynb
   .. tutorials/deneve_reservoirs.ipynb
   .. tutorials/network_ads_tutorial.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Training and deploying to Xylo

   devices/xylo-overview.ipynb
   devices/quick-xylo/deploy_to_xylo.ipynb
   devices/quick-xylo/xylo-audio-2-intro.ipynb
   devices/torch-training-spiking-for-xylo.ipynb
   devices/analog-frontend-example.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Training and deploying to SE2

   devices/DynapSE/dynapse-overview.ipynb
   devices/DynapSE/post-training.ipynb
   devices/DynapSE/neuron-model.ipynb
   devices/DynapSE/jax-training.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Advanced topics

   advanced/graph_overview.ipynb
   advanced/graph_mapping.ipynb
   reference/params_types.ipynb
   reference/api
   advanced/CHANGELOG

.. toctree::
   :maxdepth: 1
   :caption: Developer documentation

   developer/UML-diagrams.ipynb
   developer/backend-management.ipynb
   devices/DynapSE/developer.ipynb
   developer/release_process.rst

* :ref:`genindex`
