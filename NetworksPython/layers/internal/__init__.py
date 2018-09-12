## __init__.py Smart importer for submodules
import importlib
from warnings import warn

# - Dictionary {module file} -> {class name to import}
dModules = {
    ".iaf_brian": ("FFIAFBrian", "FFIAFSpkInBrian", "RecIAFBrian", "RecIAFSpkInBrian"),
    ".rate": ("FFRateEuler", "PassThrough", "RecRateEuler"),
    ".exp_synapses_brian": "FFExpSynBrian",
    ".exp_synapses_manual": "FFExpSyn",
    ".evSpikeLayer": "EventDrivenSpikingLayer",
    ".iaf_cl": ("FFCLIAF", "RecCLIAF"),
    ".iaf_cl_extd": "RecCLIAFExtd",
    ".softmaxlayer": "SoftMaxLayer",
    ".averagepooling": ("AveragePooling2D", "TorchSumPooling2dLayer"),
    ".iaf_digital": "RecDIAF",
    ".spike_bt": "RecFSSpikeEulerBT",
    ".spiking_conv2d_torch": "CNNWeightTorch",
    ".torch_cnn_layer": "FFCLIAFTorch",
}

# - Define current package
strBasePackage = "NetworksPython.layers.internal"


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
