.. _installation:

Installing |project|
====================

Base requirements
-----------------

|project| requires `Python 3.6`_, numpy_, scipy_ and numba_ to install. These requirements will be installed by `pip` when installing |project|. We recommend using anaconda_, miniconda_ or another environment manager to keep your Python dependencies clean.

Installation using `pip`
------------------------

The simplest way to install |project| is by using `pip` to download and install the latest version from PyPI.

.. code-block:: Bash

    pip install rockpool

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

    $ pip install rockpool[all]


Contributing
============

If you would like to contribute to |project|, then you should begin by forking the public repository at https://gitlab.com/ai-ctx/rockpool to your own account. Then clone your fork to your development machine

.. code-block:: Bash

    $ git clone https://gitlab.com/your-fork-location/rockpool.git rockpool


Install the package in development mode using `pip`

.. code-block:: Bash

    $ cd rockpool
    $ pip install -e . --user


or

.. code-block:: Bash

    $ pip install -e .[all] --user


The main branch is `development`. You should commit your modifications to a new feature branch.

.. code-block:: Bash

    $ git checkout -b feature/my-feature develop
    ...
    $ git commit -m 'This is a verbose commit message.'


Then push your new branch to your repository

.. code-block:: Bash

    $ git push -u origin feature/my-feature


Use the `Black code formatter`_ on your submission during your final commit. This is required for us to merge your changes. If your modifications aren't already covered by a unit test, please include a unit test with your merge request. Unit tests go in the `tests` directory.

Then when you're ready, make a merge request on gitlab.com, from the feature branch in your fork to https://gitlab.com/ai-ctx/rockpool.

.. _`Black code formatter`: https://black.readthedocs.io/en/stable/

Running tests
-------------

As part of the merge review process, we'll check that all the unit tests pass. You can check this yourself (and probably should before making your merge request), by running the unit tests locally.

To run all the unit tests for |project|, use `pytest`:

.. code-block:: Bash

    $ pytest tests


Building documentation
----------------------

The |project| documentation requires Sphinx_, NBSphinx_ and Sphinx-autobuild_. The commands

.. code-block:: Bash

    $ cd docs
    $ make livehtml


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
