import logging
from distutils.core import setup
import setuptools
from setuptools import Extension

# - Read version
exec(open("rockpool/version.py").read())

setup_args = {
    "name": "rockpool",
    "author": "SynSense",
    "author_email": "dylan.muir@synsense.ai",
    "version": __version__,
    "packages": setuptools.find_packages(),
    "install_requires": ["numpy", "scipy"],
    "extras_require": {
        "all": [
            "tqdm",
            "brian2",
            "pytest>=6.0",
            "pytest-xdist",
            "torch",
            "torchvision",
            "jax>=0.2.13",
            "jaxlib>=0.1.66",
            "samna",
            "sphinx",
            "nbsphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            "recommonmark",
            "pandoc",
        ]
    },
    "description": "A Python package for developing, simulating and training spiking neural networks, and deploying on neuromorphic hardware",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
    "keywords": "spiking neural network SNN neuromorphic",
    "python_requires": ">=3.6",
    "project_urls": {
        "Source Code": "https://gitlab.com/SynSense/rockpool",
        "Documentation": "https://rockpool.ai",
        "Bug Tracker": "https://gitlab.com/SynSense/rockpool/-/issues",
    },
}

try:
    from torch.utils import cpp_extension

    # cpp extensions
    ext_modules = [
        cpp_extension.CppExtension(
            name="torch_lif_cpp",
            sources=[
                "rockpool/nn/modules/torch/cpp/lif.cpp",
                "rockpool/nn/modules/torch/cpp/threshold.cpp",
                "rockpool/nn/modules/torch/cpp/bitshift.cpp",
            ],
            extra_compile_args=["-O3"],
        ),
    ]

    cmdclass = {"build_ext": cpp_extension.BuildExtension}

    setup(ext_modules=ext_modules, cmdclass=cmdclass, **setup_args)
except:
    logging.warning("The Torch C++ extension could not be compiled")
    setup(**setup_args)
