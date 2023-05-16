from rockpool.utilities.backend_management import (
    backend_available,
    missing_backend_shim,
)

if backend_available("jax"):
    from .gradient import *
    from .digital import *
    from .learn import *
    from .loss import *
    from .weight_handler import *
else:
    step_pwl_ae = missing_backend_shim("step_pwl_ae", "jax")
    DigitalAutoEncoder = missing_backend_shim("DigitalAutoEncoder", "jax")
    learn_weights = missing_backend_shim("learn_weights", "jax")
    loss_reconstruction = missing_backend_shim("loss_reconstruction", "jax")
    penalty_negative = missing_backend_shim("penalty_negative", "jax")
    penalty_reconstruction = missing_backend_shim("penalty_reconstruction", "jax")
    WeightHandler = missing_backend_shim("WeightHandler", "jax")
