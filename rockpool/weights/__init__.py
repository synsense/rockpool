## __init__.py Smart importer for submodules
import importlib
from warnings import warn

# - Dictionary {module file} -> {class name to import}
dModules = {
    ".gpl.reservoirweights": (
        "add_random_long_range",
        "combine_ff_rec_stack",
        "digital",
        "DiscretiseWeightMatrix",
        "dynapse_conform",
        "iaf_sparse_net",
        "in_res_digital",
        "in_res_dynapse",
        "in_res_dynapse_flex",
        "one_dim_exc_res",
        "partitioned_2d_reservoir",
        "ring_reservoir",
        "rndm_sparse_ei_net",
        "rndm_ei_net",
        "two_dim_exc_res",
        "unit_lambda_net",
        "wilson_cowan_net",
        "wipe_non_switiching_eigs",
    )
}


# - Define current package
strBasePackage = "rockpool.weights"

# - Define docstring for module
__doc__ = """Defines functions for generating recurrent weight matrices"""

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


# from .internal import *

# from .internal import __all__ as suball

# __all__ += suball
