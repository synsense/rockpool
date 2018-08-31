## __init__.py Smart importer for submodules
import importlib
from warnings import warn

# - Dictionary {module file} -> {class name to import}
dModules = {
    ".iaf_brian": ("FFIAFBrian", "FFIAFSpkInBrian"),
    ".rate": ("FFRateEuler", "PassThrough", "RecRateEuler"),
    ".exp_synapses_brian": "FFExpSynBrian",
    ".exp_synapses_manual": "FFExpSyn",
    ".evSpikeLayer": "EventDrivenSpikingLayer",
    ".iaf_cl": "FFCLIAF",
    ".softmaxlayer": "SoftMaxLayer",
    ".averagepooling": "AveragePooling2D",
}

# - Define current package
strBasePackage = "NetworksPython.layers.internal"


# - Initialise list of available modules
__all__ = []

# - Loop over submodules to attempt import
for strModule, classnames in dModules.items():
    try:
        strClass = classnames  # If string name
        # - Attempt to import the package
        locals()[strClass] = getattr(
            importlib.import_module(strModule, strBasePackage), strClass
        )

        # - Add the resulting class to __all__
        __all__.append(strClass)

    except TypeError:
        for strClass in classnames:  # If list of class names
            # - Attempt to import the package
            locals()[strClass] = getattr(
                importlib.import_module(strModule, strBasePackage), strClass
            )

            # - Add the resulting class to __all__
            __all__.append(strClass)

    except ModuleNotFoundError as err:
        # - Ignore ModuleNotFoundError
        warn("Could not load package " + strModule)
        print(err)
        pass

    except ImportError as err:
        # - Raise a warning if the package could not be imported for any other reason
        warn("Could not load package " + strModule)
        print(err)
