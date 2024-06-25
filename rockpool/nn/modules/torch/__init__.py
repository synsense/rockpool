"""
Modules using Torch as a backend
"""

try:
    from .torch_module import *
    from .rate_torch import *
    from .lowpass import *
    from .lif_torch import *
    from .ahp_lif_torch import *

    from .lif_bitshift_torch import *
    from .lif_neuron_torch import *
    from .exp_syn_torch import *
    from .updown_torch import *
    from .linear_torch import *
    from .bool_state import *
    from .dropout import *

except:
    from rockpool.utilities.backend_management import (
        backend_available,
        missing_backend_shim,
    )

    if not backend_available("torch"):
        TorchModule = missing_backend_shim("TorchModule", "torch")
        LIFTorch = missing_backend_shim("LIFTorch", "torch")
        aLIFTorch = missing_backend_shim("aLIFTorch", "torch")

        LowPass = missing_backend_shim("LowPass", "torch")
        RateTorch = missing_backend_shim("RateTorch", "torch")
        LIFBitshiftTorch = missing_backend_shim("LIFBitshiftTorch", "torch")
        LIFNeuronTorch = missing_backend_shim("LIFNeuronTorch", "torch")
        ExpSynTorch = missing_backend_shim("ExpSynTorch", "torch")
        UpDownTorch = missing_backend_shim("UpDownTorch", "torch")
        LinearTorch = missing_backend_shim("LinearTorch", "torch")
    else:
        raise

try:
    from .nir import *

except:
    from rockpool.utilities.backend_management import (
        backend_available,
        missing_backend_shim,
    )

    if not backend_available("nir", "nirtorch", "torch"):
        from_nir = missing_backend_shim("from_nir", "nir, nirtorch, torch")
        to_nir = missing_backend_shim("to_nir", "nir, nirtorch, torch")
    else:
        raise
