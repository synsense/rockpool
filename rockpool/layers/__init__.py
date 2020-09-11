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
    ".gpl.spike_ads": "RecFSSpikeADS",
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
    ".gpl.devices.dynap_hw": ("RecDynapSE", "RecDynapSEDemo"),
    ".gpl.devices.virtual_dynapse": "VirtualDynapse",
    ".gpl.rate_jax": (
        "RecRateEulerJax",
        "RecRateEulerJax_IO",
        "ForceRateEulerJax_IO",
        "FFRateEulerJax",
        "H_ReLU",
        "H_tanh",
    ),
    ".gpl.filter_bank": ("ButterMelFilter", "ButterFilter"),
    ".gpl.lif_jax": (
        "RecLIFJax",
        "RecLIFCurrentInJax",
        "RecLIFCurrentInJax_SO",
        "RecLIFJax_IO",
        "RecLIFCurrentInJax_IO",
        "FFLIFJax_IO",
        "FFLIFJax_SO",
        "FFLIFCurrentInJax_SO",
        "FFExpSynCurrentInJax",
        "FFExpSynJax",
    ),
}


# - Define current package
strBasePackage = "rockpool.layers"

# - Define docstring for module
__doc__ = """Defines classes for simulating layers of neurons"""

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
        print(bcolors.FAIL + bcolors.BOLD + str(err) + bcolors.ENDC)
        pass

    except ImportError as err:
        # - Raise a warning if the package could not be imported for any other reason
        warn("Could not load package " + strModule)
        print(bcolors.FAIL + bcolors.BOLD + str(err) + bcolors.ENDC)
