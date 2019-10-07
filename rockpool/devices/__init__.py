## __init__.py Smart importer for submodules
import importlib
from warnings import warn

# - Dictionary {module file} -> {class name to import}
dModules = {
    ".dynapse_control_extd": "DynapseControlExtd",
    ".dynapse_control": (
        "connectivity_matrix_to_prepost_lists",
        "connect_rpyc",
        "correct_argument_types",
        "correct_argument_types_and_teleport",
        "correct_type",
        "DynapseControl",
        "evaluate_firing_rates",
        "event_data_to_channels",
        "generate_event_raster",
        "initialize_hardware",
        "rectangular_neuron_arrangement",
        "remote_function",
        "setup_rpyc",
        "setup_rpyc_namespace",
        "teleport_function",
    ),
}

# - Define current package
strBasePackage = "rockpool.devices"

# - Define docstring for module
__doc__ = """Defines classes for interacting with hardware"""

# - Initialise list of available modules
__all__ = []

# - Loop over submodules to attempt import
for strModule, classnames in dModules.items():
    try:
        if isinstance(classnames, str):
            # - Attempt to import the module, get the requested class
            strClass = classnames
            locals()[strClass] = getattr(
                importlib.import_module(strModule, strBasePackage), strClass
            )

            # - Add the resulting class to __all__
            __all__.append(strClass)

        elif isinstance(classnames, tuple):
            for strClass in classnames:
                # - Attempt to import the module
                locals()[strClass] = getattr(
                    importlib.import_module(strModule, strBasePackage), strClass
                )

                # - Add the resulting class to __all__
                __all__.append(strClass)

        elif classnames is None:
            # - Attempt to import the module alone
            locals()[strModule] = importlib.import_module(strModule, strBasePackage)

            # - Add the module to __all__
            __all__.append(strModule)

    except ModuleNotFoundError as err:
        # - Ignore ModuleNotFoundError
        warn("Could not load package " + strModule)
        print(err)
        pass

    except ImportError as err:
        # - Raise a warning if the package could not be imported for any other reason
        warn("Could not load package " + strModule)
        print(err)
