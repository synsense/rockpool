from distutils.core import setup

setup(
    name="NetworksPython",
    version="0.1dev",
    packages=[
        "NetworksPython",
        "NetworksPython.layers",
        "NetworksPython.networks",
        "NetworksPython.networks.gpl",
        "NetworksPython.networks.internal",
        "NetworksPython.layers.gpl",
        "NetworksPython.layers.internal",
        "NetworksPython.layers.internal.pytorch",
    ],
    license="All rights reserved aiCTX AG",
    install_requires=["numba", "numpy", "scipy", "tqdm", "brian2", "scikit-image"],
    long_description=open("README.txt").read(),
)
