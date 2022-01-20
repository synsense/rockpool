import pytest
import torch


def test_defaults():
    from rockpool.nn.modules import LIF, LIFJax, LIFTorch
    from rockpool.utilities.jax_tree_utils import tree_find
    from jax.tree_util import tree_multimap
    import numpy as np

    Module_classes = [LIF, LIFJax, LIFTorch]

    N = 2

    def get_defaults(Mod_class: type) -> dict:
        mod = Mod_class(N)
        params = mod.parameters()
        simparams = mod.simulation_parameters()
        simparams.pop("learning_window", None)
        simparams.pop("spike_generation_fn", None)

        return {"params": dict(params), "simparams": dict(simparams)}

    # - Get default parameters for each module class
    results = [get_defaults(M) for M in Module_classes]

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().numpy()
        else:
            return np.array(x)

    # - Verify that all elements are equal
    for class_index in range(1, len(Module_classes)):
        mismatch_params = tree_multimap(
            lambda a, b: not np.allclose(to_numpy(a), to_numpy(b), equal_nan=True),
            results[0],
            results[class_index],
        )

        mismatch = tree_find(mismatch_params)

        if len(mismatch) > 0:
            raise ValueError(
                f"Comparing {Module_classes[0].__name__} and {Module_classes[class_index].__name__}\nSome elements of default parameters do not match:\n{mismatch}"
            )


def test_dynamics():
    from rockpool.nn.modules import LIF, LIFJax, LIFTorch, TorchModule, JaxModule
    import numpy as np
    import torch

    from rockpool.utilities.jax_tree_utils import tree_find
    from jax.tree_util import tree_multimap

    Module_classes = [LIF, LIFJax, LIFTorch]

    Nin = 4
    Nout = 2
    batches = 3
    T = 10
    input_data = np.random.rand(batches, T, Nin)
    w_rec = np.random.randn(Nout, Nin) / np.sqrt(Nin)

    def get_dynamics(Mod_class: type):
        is_torch = issubclass(Mod_class, TorchModule)
        this_data = torch.from_numpy(input_data).float() if is_torch else input_data
        this_w_rec = torch.from_numpy(w_rec).float() if is_torch else w_rec

        mod = Mod_class((Nin, Nout), bias=0.01)
        out, ns, r_d = mod(this_data, record=True)
        ns.pop("rng_key", None)

        mod_rec = Mod_class((Nin, Nout), bias=0.01, has_rec=True, w_rec=this_w_rec)
        out_rec, ns_rec, r_d_rec = mod_rec(this_data, record=True)
        ns_rec.pop("rng_key", None)

        return {
            "output": out,
            "states": dict(ns),
            "record": dict(r_d),
            "rec_output": out_rec,
            "rec_states": dict(ns_rec),
            "rec_record": dict(r_d_rec),
        }

    # - Get dynamics for each module class
    results = [get_dynamics(M) for M in Module_classes]

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().numpy()
        else:
            return np.array(x)

    # - Verify that all elements are equal
    for class_index in range(1, len(Module_classes)):
        mismatch_dynamics = tree_multimap(
            lambda a, b: not np.allclose(
                to_numpy(a), to_numpy(b), equal_nan=True, atol=1e-4
            ),
            results[0],
            results[class_index],
        )

        mismatch = tree_find(mismatch_dynamics)

        if len(mismatch) > 0:
            raise ValueError(
                f"Comparing {Module_classes[0].__name__} and {Module_classes[class_index].__name__}\nSome elements of dynamics do not match:\n{mismatch}"
            )
