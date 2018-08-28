## __init__.py Smart importer for submodules

# - Dictionary {module file} -> {class name to import}
dModules = {
    '.rate': 'RecRateEuler',
    '.iaf_brian': 'RecIAFBrian',
    '.spike_bt': 'RecFSSpikeEulerBT',
    '.iaf_cl': 'RecCLIAF',
    '.iaf_digital': 'RecDIAF',
}

# - Define current package
strBasePackage = 'NetworksPython.layers.recurrent'

# - Required imports
import importlib
from warnings import warn

# - Initialise list of available modules
__all__ = []

# - Loop over submodules to attempt import
for strModule, strClass in dModules.items():
    try:
        # - Attempt to import the package
        locals()[strClass] = importlib.import_module(strModule, strBasePackage)

        # - Add the resulting class to __all__
        __all__.append(strClass)

    except ModuleNotFoundError:
        # - Ignore ModuleNotFoundError
        pass

    except ImportError as err:
        # - Raise a warning if the package could not be imported for any other reason
        warn('Could not load package ' + strModule)
        print(err)