.. _installation:

Installing |project|
====================

Base requirements
-----------------

|project| requires `Python 3.6`_, numpy_, scipy_ and numba_ to install. These requirements will be installed by `pip` when installing |project|. We recommend using anaconda_, miniconda_ or another environment manager to keep your Python dependencies clean.

Cloning from Gitlab in edit mode
--------------------------------

The best way to stay up to date with bug fixes and improvements is to clone the ``git`` repository and install the package from source.

* Clone the repository using your favourite ``git`` tool
* Install using ``pip``:

.. code-block:: Bash

    pip install -e .

This command installs the packages without copying them from the repository location. This means that when you pull updates to |project|, they will be applied automatically without reinstalling.

Dependencies
------------

|project| has several dependencies for various aspects of the package. However, these dependencies are compartmentalised as much as possible. For example, Jax_ is required to use the Jax_-backed layers (e.g. `.RecRateEulerJax`); PyTorch_ is required to use the Torch_-backed layers (e.g. `.RecIAFTorch`), and so on. But if these dependencies are not available, the remainder of |project| is still usable.

* NEST_ for NEST_-backed layers
* Jax_ for Jax_-backed layers
* PyTorch_ for Torch_-backed layers
* Brian2_ for Brian_-backed layers
* Matplotlib_ or HoloViews_ for plotting `.TimeSeries`
* PyTest_ for running tests
* Sphinx_, NBSphinx_ and Sphinx-autobuild_ for building documentation

To automatically install all the extra dependencies required by |project|, use the command

.. code-block:: Bash

    pip install -e .[all]

Running tests
-------------

To run all the unit tests for |project|, use `pytest`:

.. code-block:: Bash

    pytest tests

Building documentation
----------------------

The |project| documentation requires Sphinx_, NBSphinx_ and Sphinx-autobuild_. The commands

.. code-block:: Bash

    cd docs
    make livehtml

Will compile the documentation and open a web browser to the local copy of the docs.

.. _Python 3.6: https://python.org
.. _numpy: https://www.numpy.org
.. _scipy: https://www.scipy.org
.. _numba: https://numba.pydata.org
.. _Jax: https://github.com/google/jax
.. _PyTorch: https://pytorch.org/
.. _Torch: https://pytorch.org/
.. _NEST: https://www.nest-simulator.org
.. _Brian: https://github.com/brian-team/brian2
.. _Brian2: https://github.com/brian-team/brian2
.. _PyTest: https://github.com/pytest-dev/pytest
.. _Sphinx: http://www.sphinx-doc.org
.. _NBSphinx: https://github.com/spatialaudio/nbsphinx
.. _Sphinx-autobuild: https://github.com/GaretJax/sphinx-autobuild
.. _anaconda: https://www.anaconda.com
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Matplotlib: https://matplotlib.org
.. _Holoviews: http://holoviews.org
.. _tqdm: https://github.com/tqdm/tqdm
