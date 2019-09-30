from distutils.core import setup

setup(
    name="rockpool",
    version="0.2",
    packages=[
        "rockpool",
        "rockpool.weights",
        "rockpool.weights.gpl",
        "rockpool.weights.internal",
        "rockpool.utilities",
        "rockpool.utilities.gpl",
        "rockpool.utilities.internal",
        "rockpool.networks",
        "rockpool.networks.gpl",
        "rockpool.networks.internal",
        "rockpool.layers",
        "rockpool.layers.gpl",
        "rockpool.layers.internal",
        "rockpool.layers.internal.pytorch",
        "rockpool.layers.internal.devices",
        "rockpool.layers.training",
        "rockpool.layers.training.gpl",
        "rockpool.layers.training.internal",
    ],
    license="All rights reserved aiCTX AG",
    install_requires=["numba", "numpy", "scipy"],
    extras_require={
        "all": [
            "tqdm",
            "brian2",
            "pytest",
            "torch",
            "torchvision",
            "rpyc",
            "jax",
            "jaxlib",
            "sphinx",
            "nbsphinx",
            "sphinx-autobuild",
        ]
    },
    long_description=open("README.md").read(),
)
