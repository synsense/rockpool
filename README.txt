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

If you want to install all the extra dependencies required for NEST, Brian, PyTorch and Jax layers, use the command

`$ pip insyall -e .[all] --user`

License
=======

Rockpool is released under a GPL license, unless otherwise agreed.
