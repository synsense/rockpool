NetworksPython
==============

Installation instructions
=========================

Checkout the repository

`$ git checkout <git url>`
`$ cd network-architectures`

Install packages in requirements_conda.sh
If you are a conda user, you can simply execute this file
`$ ./requirements_conda.sh`

If not, it is advisable to install these packages using your system package manager to ensure compatibility.

The rest of the requirements can be installed using pip.

`$ pip install -r requirements.txt --user`
`$ pip install -e . --user`

The `--user` option installs an egg only for the current user. 
The -e option allows you to install this as a development version so you can make changes.

License
=======
