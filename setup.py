from distutils.core import setup
import setuptools

setup(
    name="rockpool",
    author="aiCTX",
    author_email="dylan.muir@aictx.ai",
    version="0.9a",
    packages=setuptools.find_packages(),
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
