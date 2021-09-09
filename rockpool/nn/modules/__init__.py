## __init__.py Smart importer for submodules
import importlib
from warnings import warn

# - Dictionary {module file} -> {class name to import}
dModules = {
    ".module": "Module",
    ".timed_module": ("TimedModule", "TimedModuleWrapper"),
    ".jax.jax_module": "JaxModule",
    ".jax.lif_jax": "LIFJax",
    ".jax.rate_jax": "RateEulerJax",
    ".jax.exp_smooth_jax": "ExpSmoothJax",
    ".jax.softmax_jax": ("SoftmaxJax", "LogSoftmaxJax"),
    ".native.linear": ("Linear", "LinearJax"),
    ".native.instant": ("Instant", "InstantJax"),
    ".native.filter_bank": ("ButterMelFilter", "ButterFilter"),
    ".nest.iaf_nest": ("FFIAFNest", "RecIAFSpkInNest", "RecAEIFSpkInNest"),
    ".torch.torch_module": "TorchModule",
    ".torch.lif_torch": "LIFTorch",
    ".torch.lif_bitshift_torch": "LIFBitshiftTorch",
    ".torch.lowpass": "LowPass",
    ".torch.exp_syn_torch": "ExpSynTorch",
    ".torch.lif_neuron_torch": "LIFNeuronTorch",
    ".torch.linear_torch": "LinearTorch",
    ".torch.updown_torch": "UpDownTorch",
}


# - Define current package
strBasePackage = "rockpool.nn.modules"

# - Define docstring for module
__doc__ = """
            rockpool.nn.modules package - Contains building-block modules for networks
            
            Subpackages:
                `native`: Numpy-backed modules
                `jax`: Jax-backed modules
                `torch`: Torch-backed modules
                `nest`: NEST-backed modules
                `brian`: Brian2-backed modules
                
            The bases classes :py:class:`.Module`, :py:class:`.JaxModule` and :py:class:`TorchModule` are used to compose arbitrary networks of SNN modules.
            
            See Also:
                See :ref:`/basics/getting_started.ipynb` and :ref:`/in-depth/api-low-level.ipynb` for description and examples for building networks and writing your own :py:class:`.Module` classes. See :ref:`/in-depth/api-high-level.ipynb` for details of the :py:class:`.TimedModule` high-level API. 
            """

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

        elif isinstance(classnames, (tuple, list)):
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
