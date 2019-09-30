Rockpool
========

Installation instructions
=========================

Clone the repository.

`$ git checkout <git url>`
`$ cd network-architectures`

Use `pip` to install Rockpool and required dependencies

`$ pip install -e . --user`

The `--user` option installs an egg only for the current user.
The -e option allows you to install this as a development version so you can make changes.

If you want to install all the extra dependencies required for Brian, PyTorch and Jax layers, use the command

`$ pip install -e .[all] --user`

NEST layers
-----------

The NEST simulator cannot be installed using `pip`. Please see the documentation for NEST at [https://nest-simulator.readthedocs.io/en/latest/] for instructions on how to get NEST running on your system.

License
=======

Rockpool is released under a GPL license, unless otherwise agreed.
