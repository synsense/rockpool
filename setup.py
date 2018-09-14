from distutils.core import setup

setup(
    name="NetworksPython",
    version="0.1dev",
    packages=[
        "NetworksPython",
        "NetworksPython.layers",
        "NetworksPython.layers.internal",
        "NetworksPython.layers.gpl",
    ],
    license="All rights reserved aiCTX AG",
    install_requires=["numba", "numpy", "scipy", "tqdm", "brian2", "scikit-image"],
    long_description=open("README.txt").read(),
)
