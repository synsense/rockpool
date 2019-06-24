from distutils.core import setup

setup(
    name="NetworksPython",
    version="0.1dev",
    packages=[
        "NetworksPython",
        "NetworksPython.weights",
        "NetworksPython.weights.gpl",
        "NetworksPython.weights.internal",
        "NetworksPython.networks",
        "NetworksPython.networks.gpl",
        "NetworksPython.networks.internal",
        "NetworksPython.layers",
        "NetworksPython.layers.gpl",
        "NetworksPython.layers.internal",
        "NetworksPython.layers.internal.pytorch",
    ],
    license="All rights reserved aiCTX AG",
    install_requires=[
        "numba",
        "numpy",
        "scipy",
        "tqdm",
        "brian2",
        "scikit-image",
        "pandas",
	"jax",
	"jaxlib"
    ],
    long_description=open("README.txt").read(),
)
