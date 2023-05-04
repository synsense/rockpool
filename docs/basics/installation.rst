.. _installation:

Installing |project| and contributing
=========================================

Base requirements
-----------------

|project| requires `Python 3.7`_, numpy_ and scipy_ to install. These requirements will be installed by ``pip`` when installing |project|. We recommend using anaconda_, miniconda_ or another environment manager to keep your Python dependencies clean.

Installation using ``pip``
--------------------------

The simplest way to install |project| is by using ``pip`` to download and install the latest version from PyPI.

.. code-block:: Bash

    pip install rockpool

Installation using ``conda``
----------------------------

You can also install |project| using ``conda``, from the ``conda-forge`` channel.

.. code-block:: Bash

    conda install -c conda-forge rockpool

|project| version
-----------------

To check your |project| version, access the :py:attr:`.__version__` attribute of the module:

.. code-block:: python

    import rockpool
    rockpool.__version__


Dependencies
------------

|project| has several dependencies for various aspects of the library. However, these dependencies are compartmentalised as much as possible. For example, Jax_ is required to use the Jax_-backed modules (e.g. :py:class:`.RateJax`); PyTorch_ is required to use the Torch_-backed modules, and so on. But if these dependencies are not available, the remainder of |project| is still usable.

* scipy_ for scipy_-backed modules
* numba_ for numba_-backed modules
* NEST_ for NEST_-backed modules
* Jax_ and Jaxlib_ for Jax_-backed modules
* PyTorch_ for Torch_-backed modules
* Brian2_ for Brian_-backed modules
* Sinabs_ for Sinabs_-backed modules
* Samna_, Xylosim_, Bitstruct_ for building and deploying modules to the Xylo hardware family
* Matplotlib_ or HoloViews_ for plotting :py:class:`.TimeSeries`
* PyTest_ for running tests
* Sphinx_, pandoc_, recommonmark_, NBSphinx_, sphinx-rtd-theme_ and Sphinx-autobuild_ for building documentation

To automatically install most of the extra dependencies required by |project|, use the command

.. code-block:: Bash

    $ pip install rockpool[all]

or

.. code-block:: zsh

    $ pip install rockpool\[all\]

if using zsh. Some dependencies, such as pandoc_ and NEST_, must be installed manually.

To check which computational back-ends are available to |project|, use the :func:`.list_backends` function:

.. code-block:: python

    import rockpool
    rockpool.list_backends()



Building the documentation
--------------------------

The |project| documentation is based on sphinx, and all dependencies required for a local HTML version are installed with ``pip install rockpool[all]``.

To build a live, locally-hosted HTML version of the docs, use the command

.. code-block:: Bash

    $ cd docs
    $ make clean html

Once built, the documentation will be placed in ``rockpool\docs\_build\html``. Open ``index.html`` in a web browser to start using the documentation.

To build a PDF version of the docs, you need to install ``imagemagick`` on your system, as well as a working version of ``latex`` and ``pdflatex``. You will need to install these dependencies manually.

Once all dependencies are installed, you can build the PDF docs with

.. code-block:: Bash

    $ cd docs
    $ make clean latexpdf

Contributing
------------

If you would like to contribute to |project|, then you should begin by forking the public repository at https://github.com/synsense/rockpool to your own account. Then clone your fork to your development machine

.. code-block:: Bash

    $ git clone https://github.com/your-fork-location/rockpool.git rockpool


Install the package in development mode using ``pip``

.. code-block:: Bash

    $ cd rockpool
    $ pip install -e . --user


or

.. code-block:: Bash

    $ pip install -e .[all] --user


The main branch is ``development``. You should commit your modifications to a new feature branch.

.. code-block:: Bash

    $ git checkout -b feature/my-feature develop
    ...
    $ git commit -m 'This is a verbose commit message.'


Then push your new branch to your repository

.. code-block:: Bash

    $ git push -u origin feature/my-feature


Use the `Black code formatter`_ on your submission during your final commit. This is required for us to merge your changes. If your modifications aren't already covered by a unit test, please include a unit test with your merge request. Unit tests go in the ``tests`` directory.

Then when you're ready, make a merge request on github.com, from the feature branch in your fork to https://github.com/synsense/rockpool.

.. _`Black code formatter`: https://black.readthedocs.io/en/stable/

Running tests
~~~~~~~~~~~~~

As part of the merge review process, we'll check that all the unit tests pass. You can check this yourself (and probably should before making your merge request), by running the unit tests locally.

To run all the unit tests for |project|, use ``pytest``:

.. code-block:: Bash

    $ pytest tests

.. _Python 3.7: https://python.org
.. _numpy: https://www.numpy.org
.. _scipy: https://www.scipy.org
.. _numba: https://numba.pydata.org
.. _Jax: https://github.com/google/jax
.. _Jaxlib: https://github.com/google/jax
.. _PyTorch: https://pytorch.org/
.. _Torch: https://pytorch.org/
.. _Brian: https://github.com/brian-team/brian2
.. _Brian2: https://github.com/brian-team/brian2
.. _Sinabs: https://pypi.org/project/sinabs/
.. _PyTest: https://github.com/pytest-dev/pytest
.. _Sphinx: http://www.sphinx-doc.org
.. _pandoc: https://pandoc.org
.. _NBSphinx: https://github.com/spatialaudio/nbsphinx
.. _Sphinx-autobuild: https://github.com/GaretJax/sphinx-autobuild
.. _anaconda: https://www.anaconda.com
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Matplotlib: https://matplotlib.org
.. _Holoviews: http://holoviews.org
.. _tqdm: https://github.com/tqdm/tqdm
.. _Samna: https://pypi.org/project/samna/
.. _Xylosim: https://pypi.org/project/xylosim/
.. _Bitstruct: https://pypi.org/project/bitstruct/
.. _sphinx-rtd-theme: https://pypi.org/project/sphinx-rtd-theme/
.. _recommonmark: https://pypi.org/project/sphinx-rtd-theme/
