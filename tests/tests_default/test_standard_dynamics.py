import pytest

pytest.importorskip("jax")
pytest.importorskip("torch")

# - Set 64-bit mode
from jax.config import config

config.update("jax_enable_x64", True)

import torch
import numpy as np
import jax
from rockpool.utilities.tree_utils import tree_find
from jax.tree_util import tree_map

from typing import List
from rockpool.typehints import Tree


class MismatchError(ValueError):
    pass


def compare_value_tree(results: List[Tree], Classes: List[type], atol: float = 1e-4):
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().numpy()
        else:
            return np.array(x)

    # - Verify that all elements are equal
    for class_index in range(1, len(results)):
        try:
            mismatch_params = tree_map(
                lambda a, b: not np.allclose(
                    to_numpy(a), to_numpy(b), equal_nan=True, atol=atol
                )
                or not np.shape(to_numpy(a)) == np.shape(to_numpy(b)),
                results[0],
                results[class_index],
            )

            mismatch = list(tree_find(mismatch_params))

            if len(mismatch) > 0:
                raise MismatchError(
                    f"Comparing {Classes[0].__name__} and {Classes[class_index].__name__}\nSome elements of comparison values do not match:\n{mismatch}"
                )
        except MismatchError:
            raise

        except Exception as e:
            raise ValueError(
                f"Error comparing classes {Classes[0].__name__} and {Classes[class_index].__name__}:\n"
            ) from e


def get_torch_gradients(module, data):
    data = torch.as_tensor(data, dtype=torch.float)
    data.requires_grad = True

    out, _, _ = module(data)
    (out**2).sum().backward()

    def extract_grads(params):
        grads = {}
        for k, p in params.items():
            if isinstance(p, dict):
                grads.update({k: extract_grads(p)})
            else:
                grads.update({k: p.grad.detach()})
        return grads

    return extract_grads(module.parameters())


def get_jax_gradients(module, data):
    from jax.test_util import check_grads

    params, param_def = jax.tree_flatten(module.parameters())

    def grad_check(*params):
        jax.tree_unflatten(param_def, params)
        mod = module.set_attributes(params)
        out, _, _ = mod(data)
        return (out**2).sum()

    def grad_check_dict(parameters):
        mod = module.set_attributes(parameters)
        out, _, _ = mod(data)
        return (out**2).sum()

    check_grads(grad_check, params, order=2)
    return jax.grad(grad_check_dict)(module.parameters())


def test_lif_defaults():
    from rockpool.nn.modules import LIF, LIFJax, LIFTorch

    Module_classes = [LIF, LIFJax, LIFTorch]

    Nin = 4
    N = 2

    def get_defaults(Mod_class: type) -> dict:
        mod = Mod_class((Nin, N))
        params = mod.parameters()
        simparams = mod.simulation_parameters()
        simparams.pop("leak_mode", None)
        simparams.pop("learning_window", None)
        simparams.pop("spike_generation_fn", None)

        return {"params": dict(params), "simparams": dict(simparams)}

    # - Get default parameters for each module class
    results = [get_defaults(M) for M in Module_classes]

    compare_value_tree(results, Module_classes)


def test_lif_dynamics():
    from rockpool.nn.modules import LIF, LIFJax, LIFTorch, TorchModule, JaxModule
    import numpy as np
    import torch

    Module_classes = [LIF, LIFJax, LIFTorch]

    Nin = 200
    Nout = 100
    batches = 3
    T = 20
    input_data = np.random.rand(batches, T, Nin)
    w_rec = np.random.randn(Nout, Nin) / np.sqrt(Nin)
    tau_syn = np.random.uniform(20e-3, 50e-3, (Nin,))
    tau_mem = np.random.uniform(20e-3, 50e-3, (Nout,))
    bias = np.random.uniform(-1, 1, (Nout,))
    threshold = np.random.uniform(1, 2, (Nout,))

    def get_dynamics(Mod_class: type):
        is_torch = issubclass(Mod_class, TorchModule)
        this_data = torch.from_numpy(input_data).float() if is_torch else input_data
        this_w_rec = torch.from_numpy(w_rec).float() if is_torch else w_rec
        this_tau_syn = torch.from_numpy(tau_syn).float() if is_torch else tau_syn
        this_tau_mem = torch.from_numpy(tau_mem).float() if is_torch else tau_mem
        this_bias = torch.from_numpy(bias).float() if is_torch else bias
        this_threshold = torch.from_numpy(threshold).float() if is_torch else threshold

        mod = Mod_class(
            (Nin, Nout),
            bias=this_bias,
            tau_mem=this_tau_mem,
            tau_syn=this_tau_syn,
            threshold=this_threshold,
        )
        out, ns, r_d = mod(this_data, record=True)
        ns.pop("rng_key", None)

        mod_rec = Mod_class(
            (Nin, Nout),
            has_rec=True,
            w_rec=this_w_rec,
            bias=this_bias,
            tau_mem=this_tau_mem,
            tau_syn=this_tau_syn,
            threshold=this_threshold,
        )
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

    compare_value_tree(results, Module_classes, atol=1e-4 * Nin * Nout * T)


def test_linear_dynamics():
    from rockpool.nn.modules import (
        Linear,
        LinearJax,
        LinearTorch,
        TorchModule,
    )
    import numpy as np
    import torch

    Module_classes = [Linear, LinearJax, LinearTorch]

    Nin = 2
    Nout = 5
    batches = 3
    T = 10
    input_data = np.random.rand(batches, T, Nin)
    weight = np.random.randn(Nin, Nout)
    bias = np.random.randn(Nout)

    def get_dynamics(Mod_class: type):
        is_torch = issubclass(Mod_class, TorchModule)
        this_data = torch.from_numpy(input_data).float() if is_torch else input_data
        this_weight = torch.from_numpy(weight).float() if is_torch else weight
        this_bias = torch.from_numpy(bias).float() if is_torch else bias

        mod = Mod_class((Nin, Nout), bias=this_bias, weight=this_weight, has_bias=True)
        out, ns, r_d = mod(this_data, record=True)
        ns.pop("rng_key", None)

        return {
            "output": out,
            "states": dict(ns),
            "record": dict(r_d),
        }

    # - Get dynamics for each module class
    results = [get_dynamics(M) for M in Module_classes]

    compare_value_tree(results, Module_classes, atol=1e-4 * Nin * T)


def test_linear_gradients():
    from rockpool.nn.modules import LinearJax, LinearTorch
    import numpy as np

    Nin = 20
    Nout = 100
    batches = 3
    T = 10
    input_data = np.random.rand(batches, T, Nin)

    t_mod = LinearTorch((Nin, Nout), has_bias=True)
    j_mod = LinearJax(
        (Nin, Nout),
        weight=np.array(t_mod.weight.detach()),
        bias=np.array(t_mod.bias.detach()),
        has_bias=True,
    )

    t_grads = get_torch_gradients(t_mod, input_data)
    j_grads = get_jax_gradients(j_mod, input_data)

    compare_value_tree([j_grads, t_grads], [LinearTorch, LinearJax])


def test_rate_defaults():
    from rockpool.nn.modules import Rate, RateJax, RateTorch

    Module_classes = [Rate, RateJax, RateTorch]

    N = 2

    def get_defaults(Mod_class: type) -> dict:
        mod = Mod_class(N)
        params = mod.parameters()
        simparams = mod.simulation_parameters()
        simparams.pop("act_fn", None)

        return {"params": dict(params), "simparams": dict(simparams)}

    # - Get default parameters for each module class
    results = [get_defaults(M) for M in Module_classes]

    compare_value_tree(results, Module_classes)


def test_rate_dynamics():
    from rockpool.nn.modules import Rate, RateJax, RateTorch, TorchModule
    import numpy as np
    import torch

    Module_classes = [Rate, RateJax, RateTorch]

    Nin = 4
    Nout = 4
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

    compare_value_tree(results, Module_classes)


def test_expsyn_defaults():
    from rockpool.nn.modules import ExpSyn, ExpSynJax, ExpSynTorch

    Module_classes = [ExpSyn, ExpSynJax, ExpSynTorch]

    N = 2

    def get_defaults(Mod_class: type) -> dict:
        mod = Mod_class(N)
        params = mod.parameters()
        simparams = mod.simulation_parameters()
        simparams.pop("max_window_length", None)

        return {"params": dict(params), "simparams": dict(simparams)}

    # - Get default parameters for each module class
    results = [get_defaults(M) for M in Module_classes]

    compare_value_tree(results, Module_classes)


def test_expsyn_dynamics():
    from rockpool.nn.modules import ExpSyn, ExpSynJax, ExpSynTorch, TorchModule
    import numpy as np
    import torch

    Module_classes = [ExpSyn, ExpSynJax, ExpSynTorch]

    Nin = 2
    batches = 3
    T = 10
    input_data = np.random.rand(batches, T, Nin)

    def get_dynamics(Mod_class: type):
        is_torch = issubclass(Mod_class, TorchModule)
        this_data = torch.from_numpy(input_data).float() if is_torch else input_data

        mod = Mod_class((Nin,))
        out, ns, r_d = mod(this_data, record=True)
        ns.pop("rng_key", None)

        return {
            "output": out,
            "states": dict(ns),
            "record": dict(r_d),
        }

    # - Get dynamics for each module class
    results = [get_dynamics(M) for M in Module_classes]

    compare_value_tree(results, Module_classes)


def test_lif_gradients():
    from rockpool.nn.modules import LIFJax, LIFTorch
    import numpy as np

    Nin = 200
    Nout = 100
    batches = 3
    T = 20
    np.random.seed(0)
    input_data = np.random.rand(batches, T, Nin).astype("float64")
    threshold = np.ones((Nout,)).astype("float64")
    tau_syn = 100e-3 * np.ones((Nin,)).astype("float64")
    tau_mem = 150e-3 * np.ones((Nout,)).astype("float64")

    t_mod = LIFTorch(
        (Nin, Nout),
        threshold=torch.tensor(threshold),
        tau_mem=torch.tensor(tau_mem),
        tau_syn=torch.tensor(tau_syn),
    )
    j_mod = LIFJax(
        (Nin, Nout),
        threshold=threshold,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
    )

    t_grads = get_torch_gradients(t_mod, input_data)
    j_grads = get_jax_gradients(j_mod, input_data)

    print(
        "Max difference",
        np.max(np.abs(t_grads["tau_mem"].numpy() - j_grads["tau_mem"])),
    )
    atol = 1e-8 * Nin * Nout * T
    print("Tolerance", atol)

    compare_value_tree([j_grads, t_grads], [LIFJax, LIFTorch], atol=atol)


def test_linearlif_gradients():
    from rockpool.nn.modules import LIFJax, LIFTorch, LinearJax, LinearTorch
    from rockpool.nn.combinators import Sequential
    import numpy as np

    Nin = 2
    Nout = 4
    batches = 3
    T = 20
    np.random.seed(0)
    input_data = np.random.rand(batches, T, Nin).astype("float64")
    weight = np.random.uniform(-1, 1, size=(Nin, Nout)).astype("float64")
    bias = np.random.uniform(-1, 1, size=(Nout,)).astype("float64")
    threshold = np.ones((Nout,)).astype("float64")
    tau_syn = 100e-3 * np.ones((Nout,)).astype("float64")
    tau_mem = 150e-3 * np.ones((Nout,)).astype("float64")

    t_mod = Sequential(
        LinearTorch(
            (Nin, Nout),
            weight=torch.tensor(weight).float(),
            bias=torch.tensor(bias).float(),
            has_bias=True,
        ),
        LIFTorch(
            Nout,
            threshold=torch.tensor(threshold).float(),
            tau_mem=torch.tensor(tau_mem).float(),
            tau_syn=torch.tensor(tau_syn).float(),
        ),
    )
    j_mod = Sequential(
        LinearJax((Nin, Nout), weight=weight, bias=bias, has_bias=True),
        LIFJax(
            Nout,
            threshold=threshold,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
        ),
    )

    t_grads = get_torch_gradients(t_mod, torch.tensor(input_data).float())
    j_grads = get_jax_gradients(j_mod, input_data)

    print(
        "Max weight difference",
        np.max(
            np.abs(
                t_grads["0_LinearTorch"]["weight"].numpy()
                - j_grads["0_LinearJax"]["weight"]
            )
        ),
    )
    atol_weight = 1e-8 * Nin * Nout * T
    print("Tolerance", atol_weight)

    print(
        "Max tau_mem difference",
        np.max(
            np.abs(
                t_grads["1_LIFTorch"]["tau_mem"].numpy()
                - j_grads["1_LIFJax"]["tau_mem"]
            )
        ),
    )
    atol_lif = 1e-8 * Nout**2 * T**2.4
    print("Tolerance", atol_lif)

    compare_value_tree(
        [j_grads["0_LinearJax"], t_grads["0_LinearTorch"]],
        [LinearJax, LinearTorch],
        atol=atol_weight,
    )
    compare_value_tree(
        [j_grads["1_LIFJax"], t_grads["1_LIFTorch"]],
        [LIFJax, LIFTorch],
        atol=atol_lif,
    )


def test_expsyn_gradients():
    from rockpool.nn.modules import ExpSynJax, ExpSynTorch
    import numpy as np

    Nin = 2
    batches = 3
    T = 1000
    np.random.seed(0)
    input_data = np.random.rand(batches, T, Nin)

    t_mod = ExpSynTorch(Nin)
    j_mod = ExpSynJax(Nin)

    t_grads = get_torch_gradients(t_mod, input_data)
    j_grads = get_jax_gradients(j_mod, input_data)

    compare_value_tree([j_grads, t_grads], [ExpSynJax, ExpSynTorch])


def analytical_surrogate(x, t, window=0.5):
    return (x / t) * (x >= (t - window))


def analytical_dSdx(x, t, window=0.5):
    return 1 / t * (x >= (t - window))


def analytical_dSdt(x, t, window=0.5):
    return -x / (t**2) * (x >= (t - window))


def test_jax_surrogate():
    from rockpool.nn.modules.jax.lif_jax import step_pwl

    x = np.arange(-1, 10, 0.009)
    test_thresh = [1.0, 1.5, 2.0]

    analytical_grad_fn = jax.vmap(jax.grad(analytical_surrogate, argnums=(0, 1)))
    s_grad_fn = jax.vmap(jax.grad(step_pwl, argnums=(0, 1)))

    for t in test_thresh:
        # - Compare analytical output
        s_x = np.floor(analytical_surrogate(x, t))
        s_x_s = step_pwl(x, t)

        assert np.allclose(
            s_x, s_x_s
        ), f"Primal Jax spike function output does not match analytical output with t = {t}"

        # - Compare gradients
        g_x, g_t = analytical_grad_fn(x, t * np.ones(x.shape))
        g_x_s, g_t_s = s_grad_fn(x, t * np.ones(x.shape))
        ag_x = analytical_dSdx(x, t * np.ones(x.shape))
        ag_t = analytical_dSdt(x, t * np.ones(x.shape))

        assert np.allclose(
            g_x, g_x_s
        ), f"Spike function derivative w.r.t `x` does not match analytical jax gradient with t = {t}"
        assert np.allclose(
            g_t, g_t_s
        ), f"Spike function derivative w.r.t `t` does not match analytical jax gradient with t = {t}"

        assert np.allclose(
            ag_x, g_x_s
        ), f"Spike function derivative w.r.t `x` does not match analytical gradient with t = {t}"
        assert np.allclose(
            ag_t, g_t_s
        ), f"Spike function derivative w.r.t `t` does not match analytical gradient with t = {t}"


def test_torch_spike_surrogate():
    from rockpool.nn.modules.torch.lif_torch import StepPWL

    test_thresh = [1.0, 1.5, 2.0]

    for t in test_thresh:
        # - Generate input and threshold tensor
        x = torch.arange(-1, 10, 0.009)
        x.requires_grad = True
        t_t = t * torch.ones(x.shape)
        t_t.requires_grad = True

        # - Get primal outputs and gradients for analytical autograd
        s_x = analytical_surrogate(x, t_t)
        s_x.backward(torch.ones(x.shape))
        g_x, g_t = (x.grad, t_t.grad)

        # - Generate input and threshold tensor
        x = torch.arange(-1, 10, 0.009)
        x.requires_grad = True
        t_t = t * torch.ones(x.shape)
        t_t.requires_grad = True

        # - Get primal outputs and gradients for spiking surrogate
        s_x_s = StepPWL.apply(x, t_t)
        s_x_s.backward(torch.ones(x.shape))
        g_x_s, g_t_s = (x.grad, t_t.grad)

        # - Get analytical gradients
        ag_x = analytical_dSdx(x, t_t)
        ag_t = analytical_dSdt(x, t_t)

        # - Compare analytical output
        assert torch.allclose(
            torch.floor(s_x), s_x_s
        ), f"Primal Torch spike function output does not match analytical output with t = {t}"

        # - Compare gradients
        assert torch.allclose(
            g_x, g_x_s
        ), f"Torch Spike function derivative w.r.t `x` does not match torch gradient with t = {t}"
        assert torch.allclose(
            g_t, g_t_s
        ), f"Torch Spike function derivative w.r.t `t` does not match torch gradient with t = {t}"

        assert torch.allclose(
            ag_x, g_x_s
        ), f"Spike function derivative w.r.t `x` does not match analytical gradient with t = {t}"
        assert torch.allclose(
            ag_t, g_t_s
        ), f"Spike function derivative w.r.t `t` does not match analytical gradient with t = {t}"
