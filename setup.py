from distutils.core import setup
import setuptools

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
        "numba": [
            "numba",
        ],
        "docs": [
            "sphinx",
            "nbsphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            "recommonmark",
            "pandoc",
        ],
        "tests": [
            "pytest>=6.0",
            "pytest-xdist>=3.2.1",
            "pytest-random-order>=1.1.0",
            "pytest-test-groups",
        ],
        "torch": [
            "torch",
            "torchvision",
        ],
        "jax": [
            "jax>=0.4.28",
            "jaxlib>=0.4.28",
        ],
        "xylo": [
            "xylosim",
            "samna>=0.30.25.0",
            "bitstruct",
        ],
        "exodus": [
            "torch",
            "sinabs>=1.0",
            "sinabs-exodus",
        ],
        "brian": [
            "brian2",
        ],
        "sinabs": [
            "sinabs>=1.0",
        ],
        "dynapse": [
            "rockpool[jax]",
            "samna>=0.32.1.0",
        ],
        "nir": [
            "nir",
            "nirtorch",
        ],
        "extras": [
            "matplotlib",
            "tqdm",
            "rich",
        ],
        "all": [
            "rockpool[numba, docs, tests, torch, jax, xylo, brian, sinabs, dynapse, extras]",
        ],
    },
    "description": "A Python package for developing, simulating and training spiking neural networks, and deploying on neuromorphic hardware",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
    "keywords": "spiking neural network SNN neuromorphic machine learning ML",
    "python_requires": ">=3.8",
    "project_urls": {
        "Source Code": "https://github.com/SynSense/rockpool",
        "Documentation": "https://rockpool.ai",
        "Bug Tracker": "https://github.com/SynSense/rockpool/issues",
    },
    "include_package_data": True,
}

setup(**setup_args)
