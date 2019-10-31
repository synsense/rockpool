## __init__.py Smart importer for submodules
import importlib
from warnings import warn

# - Dictionary {module file} -> {class name to import}
dModules = {
    ".layer": "Layer",
    ".gpl.iaf_brian": (
        "FFIAFBrian",
        "FFIAFSpkInBrian",
        "RecIAFBrian",
        "RecIAFSpkInBrian",
    ),
    ".gpl.rate": ("FFRateEuler", "PassThrough", "RecRateEuler"),
    ".gpl.event_pass": "PassThroughEvents",
    ".gpl.exp_synapses_brian": "FFExpSynBrian",
    ".gpl.exp_synapses_manual": "FFExpSyn",
    ".gpl.iaf_cl": ("FFCLIAF", "RecCLIAF", "CLIAF"),
    ".gpl.softmaxlayer": "SoftMaxLayer",
    ".gpl.iaf_digital": "RecDIAF",
    ".gpl.spike_bt": "RecFSSpikeEulerBT",
    ".gpl.updown": "FFUpDown",
    ".gpl.pytorch.exp_synapses_torch": "FFExpSynTorch",
    ".gpl.pytorch.iaf_torch": (
        "FFIAFTorch",
        "FFIAFRefrTorch",
        "FFIAFSpkInTorch",
        "FFIAFSpkInRefrTorch",
        "RecIAFTorch",
        "RecIAFRefrTorch",
        "RecIAFSpkInTorch",
        "RecIAFSpkInRefrTorch",
        "RecIAFSpkInRefrCLTorch",
    ),
    ".gpl.iaf_nest": ("FFIAFNest", "RecIAFSpkInNest"),
    ".gpl.aeif_nest": "RecAEIFSpkInNest",
    ".gpl.devices.dynap_hw": "RecDynapSE",
    ".gpl.devices.virtual_dynapse": "VirtualDynapse",
    ".gpl.rate_jax": ("RecRateEulerJax", "ForceRateEulerJax", "H_ReLU", "H_tanh"),
    ".gpl.butter_mel_filter": "ButterMelFilter",
}


# - Define current package
strBasePackage = "rockpool.layers"

# - Define docstring for module
__doc__ = """Defines classes for simulating layers of neurons"""

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


