"""
Utility functionality for managing backends

To check a standard backend, use :py:func:`.backend_available`. To check a non-standard backend specification, use :py:func:`.check_backend`.

To build a shim class that raises an error on instantiation, for when a required backend is not available, use :py:func:`.missing_backend_shim`.
"""
import importlib
from importlib import util
from typing import List, Union, Tuple, Optional, Dict

# - Configure exports
__all__ = ["backend_available", "check_backend", "missing_backend_shim", "AbortImport"]


class AbortImport(Exception):
    pass


# - Maintain a cache of checked backends
__checked_backends: Dict[str, bool] = {}

# - Specifications for common backends
__backend_specs: Dict[str, tuple] = {
    "numpy": (),
    "numba": (),
    "nest": (),
    "jax": (["jax", "jaxlib"],),
    "torch": (),
    "sinabs": (),
    "sinabs-slayer": (["sinabs", "sinabs.slayer"],),
    "brian": (["brian2"]),
}


def check_backend(
    backend_name: str,
    required_modules: Optional[Union[Tuple[str], List[str]]] = None,
    check_flag: bool = True,
) -> bool:
    """
    Check if a backend is available, and register it in a list of available backends

    Args:
        backend_name (str): The name of this backend to check for and register
        required_modules (Optional[List[str]]): A list of required modules to search for. If ``None`` (default), check the backend name
        check_flag (bool): A manual check that can be performed externally, to see if the backend is available

    Returns:
        bool: The backend is available
    """
    # - See if the backend check is already cached
    if backend_name in __checked_backends:
        return __checked_backends[backend_name]

    # - If no list of required modules, just check the backend name
    if required_modules is None:
        required_modules = [backend_name]

    requirements_met = check_flag
    for spec in required_modules:
        try:
            # - Check the required module is installed
            requirements_met = requirements_met and (util.find_spec(spec) is not None)

            # - Try to import the module
            importlib.import_module(spec)
        except Exception as e:
            requirements_met = False

        if not requirements_met:
            break

    # - Register the backend as having been checked
    if backend_name not in __checked_backends:
        __checked_backends.update({backend_name: requirements_met})

    # - Let the caller know if we passed the check
    return requirements_met


def backend_available(*backend_names) -> bool:
    """
    Report if a backend is available for use

    This function returns immediately if the named backend has already been checked previously. Otherwise, if the backend is "standard", then it will be checked for availability. If the backend is non-standard, it cannot be checked automatically. In that case you must use :py:func:`.check_backend` directly.

    Args:
        backend_name0, backend_name1, ... (str): A backend to check

    Returns:
        bool: ``True`` iff the backend is available for use
    """

    def check_single_backend(backend_name):
        if backend_name in __checked_backends:
            return __checked_backends[backend_name]
        elif backend_name in __backend_specs:
            return check_backend(backend_name, *__backend_specs[backend_name])
        else:
            return check_backend(backend_name)

    return all([check_single_backend(be) for be in backend_names])


def missing_backend_shim(class_name: str, backend_name: str):
    """
    Make a class constructor that raises an error about a missing backend

    Examples:

        Generate a `LIFTorch` class shim, that will raise an error on instantiation.

        >>> LIFTorch = missing_backend_shim('LIFTorch', 'torch')
        >>> LIFTorch((3,), tau_syn = 10e-3)
        ModuleNotFoundError: Missing the `torch` backend. `LIFTorch` objects, and others relying on `torch` are not available.

    Args:
        class_name (str): The intended class name
        backend_name (str): The required backend that is missing

    Returns:
        Class: A class that raises an error on construction
    """

    class MBSMeta(type):
        def __getattr__(cls, *args):
            raise ModuleNotFoundError(
                f"Missing the `{backend_name}` backend. `{class_name}` objects, and others relying on `{backend_name}` are not available."
            )

    class MissingBackendShim(metaclass=MBSMeta):
        """
        BACKEND MISSING FOR THIS CLASS
        """
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                f"Missing the `{backend_name}` backend. `{class_name}` objects, and others relying on `{backend_name}` are not available."
            )

    return MissingBackendShim


def missing_backend_error(class_name, backend_name):
    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            f"Missing the `{backend_name}` backend. `{class_name}` objects, and others relying on `{backend_name}` are not available."
        )

    return __init__


def list_backends():
    """
    Print a list of computational backends available in this session
    """
    print("Backends available to Rockpool:")

    for backend in __backend_specs.keys():
        print(f"{backend:>15}: {backend_available(backend)}")
