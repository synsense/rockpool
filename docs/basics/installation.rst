.. _installation:

How to install |project|
========================

Cloning from Gitlab in edit mode
--------------------------------

The best way to stay up to date with bug fixes and improvements is to clone the ``git`` repository and install the package from source.

* Clone the repository using your favourite ``git`` tool
* Install using ``pip``:

``
pip install . -e
``

This command installs the packages without copying them from the repository location. This means that when you pull updates to |project|, they will be applied automatically without reinstalling.