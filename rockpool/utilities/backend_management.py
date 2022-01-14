"""
Utility functionality for managing backends
"""

from importlib import util
from typing import List, Union, Tuple

__all__ = ["missing_backend"]

__backend_available_list = []
__backend_unavailable_list = []


def check_backend(
    backend_name: str,
    required_modules: Union[Tuple[str], List[str]],
    check_flag: bool = True,
) -> bool:
    """
    Check if a backend is available, and register it in a list of available backends

    Args:
        backend_name (str): The name of this backend to check for and register
        required_modules (List[str]): A list of required modules to search for
        check_flag (bool): A manual check that can be performed externally, to see if the backend is available

    Returns:
        bool: The backend is available
    """
    # - Bypass the check if the backend is already registered as available
    if is_available(backend_name):
        return True

    requirements_met = check_flag
    for spec in required_modules:
        # - Check the required module is installed
        requirements_met = requirements_met and (util.find_spec(spec) is not None)

        # - Try to import the module
        try:
            import spec
        except Exception as e:
            requirements_met = False

        if not requirements_met:
            break

    # - Register the backend, if it is available
    if requirements_met and backend_name not in __backend_available_list:
        __backend_available_list.append(backend_name)

    # - Let the caller know if we passed the check
    return requirements_met


def is_available(backend_name: str) -> bool:
    """
    Report if a backend is available for use

    Args:
        backend_name:

    Returns:

    """
    return backend_name in __backend_available_list


def is_unavailable(backend_name: str) -> bool:
    """
    Report if a backed is unavailable for use

    Args:
        backend_name (str): The backedn name to check

    Returns:
        bool: ``True`` if the backend has been checked, and is **not** available
    """
    return backend_name in __backend_unavailable_list


def missing_backend(class_name: str, backend_name: str):
    """
    Make a class constructor that raises an error about a missing backend

    Args:
        class_name:
        backend_name:

    Returns:

    """

    class MissingBackendShim:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                f"Missing the {backend_name} backend. {class_name} objects, and others relying on {backend_name} will not be available."
            )

    return MissingBackendShim
