<<<<<<< HEAD
## __init__.py Smart importer for submodules
import importlib
from warnings import warn

# - Dictionary {module file} -> {class name to import}
dModules = {
    # ".dynapse_control_extd": "DynapseControlExtd",
    # ".dynapse_control": (
    #     "connectivity_matrix_to_prepost_lists",
    #     "connect_rpyc",
    #     "correct_argument_types",
    #     "correct_argument_types_and_teleport",
    #     "correct_type",
    #     "DynapseControl",
    #     "evaluate_firing_rates",
    #     "event_data_to_channels",
    #     "generate_event_raster",
    #     "initialize_hardware",
    #     "rectangular_neuron_arrangement",
    #     "remote_function",
    #     "setup_rpyc",
    #     "setup_rpyc_namespace",
    #     "teleport_function",
    # ),
    ".virtual_dynapse": "VirtualDynapse",
}

# - Define current package
strBasePackage = "rockpool.devices.dynapse"

# - Define docstring for module
__doc__ = """DynapSE-family device simulations, deployment and HDK support"""

# - Initialise list of available modules
__all__ = []


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

=======
"""
Package for simulating and interacting with Dynap™SE hardware
"""
>>>>>>> develop

from warnings import warn

try:
    from .virtual_dynapse import VirtualDynapse
except (ModuleNotFoundError, ImportError) as err:
    warn("Could not load package:" + str(err))
