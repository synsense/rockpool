try:
    from .base_loss import *
    from .peak_loss import *
    from .mse_loss import *
except:
    from rockpool.utilities.backend_management import (
        backend_available,
        missing_backend_shim,
    )

    if not backend_available("torch"):
        PeakLoss = missing_backend_shim("PeakLoss", "torch")
        BinaryPeakLoss = missing_backend_shim("BinaryPeakLoss", "torch")
        MSELoss = missing_backend_shim("MSELoss", "torch")
    else:
        raise
