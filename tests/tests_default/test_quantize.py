import pytest

pytest.importorskip("torch")
pytest.importorskip("xylosim")
pytest.importorskip("samna")


def test_simple_network():
    from rockpool.devices.xylo.syns61300 import mapper
    from rockpool.devices.xylo.syns61300 import config_from_specification
    from rockpool.devices.xylo.syns61300 import XyloSim

    import numpy as np
    import torch
    import copy
    from rockpool.transform.quantize_methods import global_quantize, channel_quantize

    # from rockpool.nn.modules import LIFJax, LinearJax, JaxModule
    # from rockpool.nn.combinators import Sequential, Residual
    from rockpool.nn.modules import (
        TorchModule,
        LinearTorch,
        LIFTorch,
        LIFBitshiftTorch,
    )
    from rockpool.parameters import Constant
    from rockpool.graph import (
        AliasConnection,
        GraphHolder,
        as_GraphHolder,
        connect_modules,
        find_modules_of_subclass,
    )
    import warnings

    warnings.filterwarnings("ignore")

    class SimpleBlock(TorchModule):
        def __init__(
            self,
            Nin: int,
            Nres: int,
            Nout: int,
            dt: float,
            device: str = "cpu",
            *args,
            **kwargs,
        ):
            super().__init__(shape=(Nin, Nout), *args, **kwargs)

            self.threshold = 10.0
            self.learning_window = 0.3
            self.Nin = Nin
            self.Nres = Nres
            self.Nout = Nout
            self.dt = dt

            self.lin_res = LinearTorch(shape=(self.Nin, self.Nres), has_bias=False)
            self.spk_res = LIFBitshiftTorch(
                shape=(self.Nres, self.Nres),
                tau_mem=0.002,
                tau_syn=0.002,
                bias=Constant(0.0),
                threshold=self.threshold,
                learning_window=self.learning_window,
                dt=self.dt,
            )

            self.lin_out = LinearTorch(shape=(self.Nres, self.Nout), has_bias=False)
            self.spk_out = LIFBitshiftTorch(
                shape=(self.Nout, self.Nout),
                tau_mem=0.002,
                tau_syn=0.002,
                bias=Constant(0.0),
                threshold=self.threshold,
                learning_window=self.learning_window,
                dt=self.dt,
            )

            self._record_dict = {}

            self.submods = []
            self.submods.append(self.lin_res)
            self.submods.append(self.spk_res)
            self.submods.append(self.lin_out)
            self.submods.append(self.spk_out)

        def forward(self, inp):
            #         (n_batches, t_sim, n_neurons) = inp.shape

            out, _, self._record_dict["lin_res"] = self.lin_res(inp, record=True)
            out, _, self._record_dict["spk_res"] = self.spk_res(out, record=True)
            out, _, self._record_dict["lin_out"] = self.lin_out(out, record=True)
            out, _, self._record_dict["spk_out"] = self.spk_out(out, record=True)

            return out

        def evolve(self, inp, record: bool = False, *args, **kwargs):
            out, new_states, _ = super().evolve(inp, record, *args, **kwargs)
            rec_dict = self._record_dict if record else {}
            return out, new_states, rec_dict

        def as_graph(self):
            mod_graphs = []

            for mod in self.submods:
                mod_graphs.append(mod.as_graph())

            connect_modules(mod_graphs[0], mod_graphs[1])
            connect_modules(mod_graphs[1], mod_graphs[2])
            connect_modules(mod_graphs[2], mod_graphs[3])  # res

            return GraphHolder(
                mod_graphs[0].input_nodes,
                mod_graphs[3].output_nodes,
                f"{type(self).__name__}_{self.name}_{id(self)}",
                self,
            )

    Nin = 16
    Nres = 1000
    Nout = 8
    dt = 1e-3

    mod = SimpleBlock(Nin, Nres, Nout, dt)

    w_res = np.random.exponential(4, [Nin, Nres])
    w_res_torch = torch.tensor(w_res).float()
    mod.lin_res.weight.data = w_res_torch

    w_out = np.random.exponential(0.1, [Nres, Nout])
    w_out_torch = torch.tensor(w_out).float()
    mod.lin_out.weight.data = w_out_torch

    float_graph = mod.as_graph()
    float_specs = mapper(
        float_graph, weight_dtype="float", threshold_dtype="float", dash_dtype="float"
    )
    dt = float_specs["dt"]

    assert mod.spk_res._hw_ids == list(
        range(1000)
    ), "hardware id is not assigned correctly"

    global_specs = copy.copy(float_specs)
    channel_specs = copy.copy(float_specs)

    del float_specs["mapped_graph"]
    del float_specs["dt"]
    xylo_conf_float, is_valid, message = config_from_specification(**float_specs)
    print("Float valid config: ", is_valid, message)

    global_specs.update(global_quantize(**global_specs, fuzzy_scaling=False))
    del global_specs["mapped_graph"]
    del global_specs["dt"]
    xylo_conf_global, is_valid, message = config_from_specification(**global_specs)
    print("Global valid config: ", is_valid, message)

    channel_specs.update(channel_quantize(**channel_specs))
    del channel_specs["mapped_graph"]
    del channel_specs["dt"]
    xylo_conf_channel, is_valid, message = config_from_specification(**channel_specs)
    print("Channel valid config: ", is_valid, message)

    T = 100
    batch = 1
    inp = torch.Tensor(np.random.randint(low=0, high=3, size=(batch, T, Nin)))

    _, _, recordings_f = mod(inp, record=True)

    cim_g = XyloSim.from_config(xylo_conf_global, dt=dt)
    cim_g.reset_state()
    spk_out_g, _, rec_cim_g = cim_g(inp[0].cpu().numpy(), record=True)

    cim_c = XyloSim.from_config(xylo_conf_channel, dt=dt)
    cim_c.reset_state()
    spk_out_c, _, rec_cim_c = cim_c(inp[0].cpu().numpy(), record=True)

    spk_in_f = recordings_f["spk_res"]["spikes"].squeeze(0).detach().numpy().astype(int)
    spk_out_f = (
        recordings_f["spk_out"]["spikes"].squeeze(0).detach().numpy().astype(int)
    )

    spk_in_g = rec_cim_g["Spikes"]
    spk_in_c = rec_cim_c["Spikes"]

    in_point = spk_in_f.shape[0] * spk_in_f.shape[1]
    out_point = spk_out_f.shape[0] * spk_out_f.shape[1]

    print(
        f"\nIn LIF spike global match float percent: {np.sum(spk_in_f == spk_in_g) / in_point * 100}"
    )
    print(
        f"Out LIF spike global match float percent: {np.sum(spk_out_f == spk_out_g) / out_point * 100}"
    )

    print(
        f"\nIn LIF spike channel match float percent: {np.sum(spk_in_f == spk_in_c) / in_point * 100}"
    )
    print(
        f"Out LIF spike channel match float percent: {np.sum(spk_out_f == spk_out_c) / out_point * 100}"
    )

    print(
        f"\nIn LIF spike global match channel: {np.sum(spk_in_g == spk_in_c) / in_point * 100}"
    )
    print(
        f"Out LIF spike global match channel: {np.sum(spk_out_g == spk_out_c) / out_point * 100}"
    )


def test_complex_network():
    from rockpool.devices.xylo.syns61300 import mapper
    from rockpool.devices.xylo.syns61300 import config_from_specification
    from rockpool.devices.xylo.syns61300 import XyloSim

    import numpy as np
    import torch
    import copy
    from rockpool.transform.quantize_methods import global_quantize, channel_quantize

    from rockpool.nn.modules import (
        TorchModule,
        LinearTorch,
        LIFTorch,
        LIFBitshiftTorch,
    )
    from rockpool.parameters import Constant
    from rockpool.graph import (
        AliasConnection,
        GraphHolder,
        as_GraphHolder,
        connect_modules,
        find_modules_of_subclass,
    )
    import warnings

    warnings.filterwarnings("ignore")

    class SequentialBlock(TorchModule):
        def __init__(
            self,
            Nin: int,
            Nres1: int,
            Nres2: int,
            Nres3: int,
            Nout: int,
            dt: float,
            device: str = "cpu",
            *args,
            **kwargs,
        ):
            super().__init__(shape=(Nin, Nout), *args, **kwargs)

            self.threshold = 30.0
            self.learning_window = 0.5
            self.Nin = Nin
            self.Nres1 = Nres1
            self.Nres2 = Nres2
            self.Nres3 = Nres3
            self.Nout = Nout
            self.dt = dt

            self.lin_res1 = LinearTorch(shape=(self.Nin, self.Nres1), has_bias=False)
            self.spk_res1 = LIFBitshiftTorch(
                shape=(self.Nres1, self.Nres1),
                tau_mem=0.002,
                tau_syn=0.002,
                bias=Constant(0.0),
                threshold=self.threshold,
                learning_window=self.learning_window,
                dt=self.dt,
            )

            self.lin_res2 = LinearTorch(shape=(self.Nres1, self.Nres2), has_bias=False)
            self.spk_res2 = LIFBitshiftTorch(
                shape=(self.Nres2, self.Nres2),
                tau_mem=0.002,
                tau_syn=0.002,
                bias=Constant(0.0),
                threshold=self.threshold,
                learning_window=self.learning_window,
                dt=self.dt,
            )

            self.lin_res3 = LinearTorch(shape=(self.Nres2, self.Nres3), has_bias=False)
            self.spk_res3 = LIFBitshiftTorch(
                shape=(self.Nres3, self.Nres3),
                tau_mem=0.002,
                tau_syn=0.002,
                bias=Constant(0.0),
                threshold=self.threshold,
                learning_window=self.learning_window,
                dt=self.dt,
            )

            self.lin_out = LinearTorch(shape=(self.Nres3, self.Nout), has_bias=False)
            self.spk_out = LIFBitshiftTorch(
                shape=(self.Nout, self.Nout),
                tau_mem=0.002,
                tau_syn=0.002,
                bias=Constant(0.0),
                threshold=self.threshold,
                learning_window=self.learning_window,
                dt=self.dt,
            )

            self._record_dict = {}

            self.submods = []
            self.submods.append(self.lin_res1)
            self.submods.append(self.spk_res1)
            self.submods.append(self.lin_res2)
            self.submods.append(self.spk_res2)
            self.submods.append(self.lin_res3)
            self.submods.append(self.spk_res3)
            self.submods.append(self.lin_out)
            self.submods.append(self.spk_out)

        def forward(self, inp):
            # (n_batches, t_sim, n_neurons) = inp.shape
            out, _, self._record_dict["lin_res1"] = self.lin_res1(inp, record=True)
            out, _, self._record_dict["spk_res1"] = self.spk_res1(out, record=True)
            out, _, self._record_dict["lin_res2"] = self.lin_res2(out, record=True)
            out, _, self._record_dict["spk_res2"] = self.spk_res2(out, record=True)
            out, _, self._record_dict["lin_res3"] = self.lin_res3(out, record=True)
            out, _, self._record_dict["spk_res3"] = self.spk_res3(out, record=True)
            out, _, self._record_dict["lin_out"] = self.lin_out(out, record=True)
            out, _, self._record_dict["spk_out"] = self.spk_out(out, record=True)

            return out

        def evolve(self, inp, record: bool = False, *args, **kwargs):
            out, new_states, _ = super().evolve(inp, record, *args, **kwargs)
            rec_dict = self._record_dict if record else {}
            return out, new_states, rec_dict

        def as_graph(self):
            mod_graphs = []
            for mod in self.submods:
                mod_graphs.append(mod.as_graph())
            connect_modules(mod_graphs[0], mod_graphs[1])
            connect_modules(mod_graphs[1], mod_graphs[2])
            connect_modules(mod_graphs[2], mod_graphs[3])
            connect_modules(mod_graphs[3], mod_graphs[4])
            connect_modules(mod_graphs[4], mod_graphs[5])
            connect_modules(mod_graphs[5], mod_graphs[6])
            connect_modules(mod_graphs[6], mod_graphs[7])

            return GraphHolder(
                mod_graphs[0].input_nodes,
                mod_graphs[7].output_nodes,
                f"{type(self).__name__}_{self.name}_{id(self)}",
                self,
            )

    Nin = 16
    Nres1 = 63
    Nres2 = 63
    Nres3 = 63
    Nout = 8
    dt = 1e-3

    mod = SequentialBlock(Nin, Nres1, Nres2, Nres3, Nout, dt)

    w_res1 = np.random.exponential(4, [Nin, Nres1])
    w_res1_torch = torch.tensor(w_res1).float()
    mod.lin_res1.weight.data = w_res1_torch

    w_res2 = np.random.exponential(4, [Nres1, Nres2])
    w_res2_torch = torch.tensor(w_res2).float()
    mod.lin_res2.weight.data = w_res2_torch

    w_res3 = np.random.exponential(4, [Nres2, Nres3])
    w_res3_torch = torch.tensor(w_res3).float()
    mod.lin_res3.weight.data = w_res3_torch

    w_out = np.random.exponential(0.1, [Nres3, Nout])
    w_out_torch = torch.tensor(w_out).float()
    mod.lin_out.weight.data = w_out_torch

    float_graph = mod.as_graph()
    float_specs = mapper(
        float_graph, weight_dtype="float", threshold_dtype="float", dash_dtype="float"
    )
    dt = float_specs["dt"]

    global_specs = copy.copy(float_specs)
    channel_specs = copy.copy(float_specs)

    del float_specs["mapped_graph"]
    del float_specs["dt"]
    xylo_conf_float, is_valid, message = config_from_specification(**float_specs)
    print("Float valid config: ", is_valid, message)

    global_specs.update(global_quantize(**global_specs, fuzzy_scaling=False))
    del global_specs["mapped_graph"]
    del global_specs["dt"]
    xylo_conf_global, is_valid, message = config_from_specification(**global_specs)
    print("Global valid config: ", is_valid, message)

    channel_specs.update(channel_quantize(**channel_specs))
    del channel_specs["mapped_graph"]
    del channel_specs["dt"]
    xylo_conf_channel, is_valid, message = config_from_specification(**channel_specs)
    print("Channel valid config: ", is_valid, message)

    T = 1000
    batch = 1
    inp = torch.Tensor(np.random.randint(low=0, high=3, size=(batch, T, Nin)))

    _, _, recordings_f = mod(inp, record=True)

    cim_g = XyloSim.from_config(xylo_conf_global, dt=dt)
    cim_g.reset_state()
    spk_out_g, _, rec_cim_g = cim_g(inp[0].cpu().numpy(), record=True)

    cim_c = XyloSim.from_config(xylo_conf_channel, dt=dt)
    cim_c.reset_state()
    spk_out_c, _, rec_cim_c = cim_c(inp[0].cpu().numpy(), record=True)

    spk_res1 = (
        recordings_f["spk_res1"]["spikes"].squeeze(0).detach().numpy().astype(int)
    )
    spk_res2 = (
        recordings_f["spk_res2"]["spikes"].squeeze(0).detach().numpy().astype(int)
    )
    spk_res3 = (
        recordings_f["spk_res3"]["spikes"].squeeze(0).detach().numpy().astype(int)
    )
    spk_out_f = (
        recordings_f["spk_out"]["spikes"].squeeze(0).detach().numpy().astype(int)
    )

    spk_res_f = np.concatenate((spk_res1, spk_res2, spk_res3), axis=1)
    spk_res_g = rec_cim_g["Spikes"]
    spk_res_c = rec_cim_c["Spikes"]

    in_point = spk_res_g.shape[0] * spk_res_g.shape[1]
    out_point = spk_out_g.shape[0] * spk_out_g.shape[1]

    print(
        f"\nIn LIF spike global match float percent: {np.sum(spk_res_f == spk_res_g) / in_point * 100}"
    )
    print(
        f"Out LIF spike global match float percent: {np.sum(spk_out_f == spk_out_g) / out_point * 100}"
    )

    print(
        f"\nIn LIF spike channel match float percent: {np.sum(spk_res_f == spk_res_c) / in_point * 100}"
    )
    print(
        f"Out LIF spike channel match float percent: {np.sum(spk_out_f == spk_out_c) / out_point * 100}"
    )

    print(
        f"\nIn LIF spike global match channel: {np.sum(spk_res_f == spk_res_c) / in_point * 100}"
    )
    print(
        f"Out LIF spike global match channel: {np.sum(spk_out_g == spk_out_c) / out_point * 100}"
    )


def test_sequential_combinator():
    from rockpool.devices.xylo.syns61300 import mapper
    from rockpool.devices.xylo.syns61300 import config_from_specification
    from rockpool.devices.xylo.syns61300 import XyloSim

    import numpy as np
    import torch
    import copy
    from rockpool.transform.quantize_methods import global_quantize, channel_quantize

    # from rockpool.nn.modules import LIFJax, LinearJax, JaxModule
    from rockpool.nn.combinators import Sequential, Residual
    from rockpool.parameters import Constant
    from rockpool.nn.modules import (
        TorchModule,
        LinearTorch,
        LIFTorch,
        LIFBitshiftTorch,
    )
    from rockpool.parameters import Constant
    from rockpool.graph import (
        AliasConnection,
        GraphHolder,
        as_GraphHolder,
        connect_modules,
        find_modules_of_subclass,
    )
    import warnings

    warnings.filterwarnings("ignore")

    Nin = 1
    Nres = 100
    Nout = 2

    mod = Sequential(
        LinearTorch((Nin, Nres), has_bias=False),
        LIFBitshiftTorch(
            shape=(Nres, Nres),
            tau_mem=0.002,
            tau_syn=0.002,
            bias=Constant(0.0),
            threshold=1.0,
            learning_window=0.5,
            dt=0.001,
        ),
        LinearTorch((Nres, Nout), has_bias=False),
        LIFBitshiftTorch(
            shape=(Nout, Nout),
            tau_mem=0.02,
            tau_syn=0.02,
            bias=Constant(0.0),
            threshold=1.0,
            learning_window=0.5,
            dt=0.001,
        ),
    )

    w_in = np.random.exponential(2, [Nin, Nres])
    w_in_torch = torch.tensor(w_in).float()
    mod[0].weight.data = w_in_torch

    w_out = np.random.exponential(0.1, [Nres, Nout])
    w_out_torch = torch.tensor(w_out).float()
    mod[2].weight.data = w_out_torch

    float_graph = mod.as_graph()
    float_specs = mapper(
        float_graph, weight_dtype="float", threshold_dtype="float", dash_dtype="float"
    )
    dt = float_specs["dt"]

    global_specs = copy.copy(float_specs)
    channel_specs = copy.copy(float_specs)

    del float_specs["mapped_graph"]
    del float_specs["dt"]

    xylo_conf_float, is_valid, message = config_from_specification(**float_specs)
    print("Float valid config: ", is_valid, "message")

    global_specs.update(global_quantize(**global_specs, fuzzy_scaling=False))
    del global_specs["mapped_graph"]
    del global_specs["dt"]
    xylo_conf_global, is_valid, message = config_from_specification(**global_specs)
    print("Global valid config: ", is_valid, message)

    channel_specs.update(channel_quantize(**channel_specs))
    del channel_specs["mapped_graph"]
    del channel_specs["dt"]
    xylo_conf_channel, is_valid, message = config_from_specification(**channel_specs)
    print("Channel valid config: ", is_valid, message)

    T = 1000
    batch = 1
    inp = torch.Tensor(np.random.randint(low=0, high=3, size=(batch, T, Nin)))

    mod.reset_state()
    _, _, recordings_f = mod(inp, record=True)

    cim_g = XyloSim.from_config(xylo_conf_global, dt=dt)
    cim_g.reset_state()
    spk_out_g, _, rec_cim_g = cim_g(inp[0].cpu().numpy(), record=True)

    cim_c = XyloSim.from_config(xylo_conf_channel, dt=dt)
    cim_c.reset_state()
    spk_out_c, _, rec_cim_c = cim_c(inp[0].cpu().numpy(), record=True)

    spk_in_f = (
        recordings_f["1_LIFBitshiftTorch"]["spikes"]
        .squeeze(0)
        .detach()
        .numpy()
        .astype(int)
    )
    spk_out_f = (
        recordings_f["3_LIFBitshiftTorch"]["spikes"]
        .squeeze(0)
        .detach()
        .numpy()
        .astype(int)
    )

    spk_in_g = rec_cim_g["Spikes"]
    spk_in_c = rec_cim_c["Spikes"]

    in_point = spk_in_f.shape[0] * spk_in_f.shape[1]
    out_point = spk_out_f.shape[0] * spk_out_f.shape[1]

    print(
        f"\nIn LIF spike global match float percent: {np.sum(spk_in_f == spk_in_g) / in_point * 100}"
    )
    print(
        f"Out LIF spike global match float percent: {np.sum(spk_out_f == spk_out_g) / out_point * 100}"
    )

    print(
        f"\nIn LIF spike channel match float percent: {np.sum(spk_in_f == spk_in_c) / in_point * 100}"
    )
    print(
        f"Out LIF spike channel match float percent: {np.sum(spk_out_f == spk_out_c) / out_point * 100}"
    )

    print(
        f"\nIn LIF spike global match channel: {np.sum(spk_in_g == spk_in_c) / in_point * 100}"
    )
    print(
        f"Out LIF spike global match channel: {np.sum(spk_out_g == spk_out_c) / out_point * 100}"
    )


def test_residual_combinator():
    from rockpool.devices.xylo.syns61300 import mapper
    from rockpool.devices.xylo.syns61300 import config_from_specification
    from rockpool.devices.xylo.syns61300 import XyloSim

    import numpy as np
    import torch
    import copy
    from rockpool.transform.quantize_methods import global_quantize, channel_quantize

    # from rockpool.nn.modules import LIFJax, LinearJax, JaxModule
    from rockpool.nn.combinators import Sequential, Residual
    from rockpool.nn.modules import (
        TorchModule,
        LinearTorch,
        LIFTorch,
        LIFBitshiftTorch,
    )
    from rockpool.parameters import Constant
    import warnings

    warnings.filterwarnings("ignore")

    Nin = 16
    Nres1 = 63
    Nres2 = 63
    Nout = 8
    threshold = 20

    mod = Sequential(
        LinearTorch((Nin, Nres1), has_bias=False),
        LIFBitshiftTorch(
            shape=(Nres1, Nres1),
            tau_mem=0.002,
            tau_syn=0.002,
            bias=Constant(0.0),
            threshold=threshold,
            learning_window=0.5,
            dt=0.001,
        ),
        Residual(
            LinearTorch((Nres1, Nres2)),
            LIFBitshiftTorch(
                shape=(Nres2, Nres2),
                tau_mem=0.002,
                tau_syn=0.002,
                bias=Constant(0.0),
                threshold=threshold,
                learning_window=0.5,
                dt=0.001,
            ),
        ),
        LinearTorch((Nres2, Nout), has_bias=False),
        LIFBitshiftTorch(
            shape=(Nout, Nout),
            tau_mem=0.02,
            tau_syn=0.02,
            bias=Constant(0.0),
            threshold=threshold,
            learning_window=0.5,
            dt=0.001,
        ),
    )

    w_res1 = np.random.exponential(0.8, [Nin, Nres1])
    w_res1_torch = torch.tensor(w_res1).float()
    mod[0].weight.data = w_res1_torch

    w_res2 = np.random.exponential(0.8, [Nres1, Nres2])
    w_res2_torch = torch.tensor(w_res2).float()
    mod[2][0].weight.data = w_res2_torch

    w_out = np.random.exponential(0.1, [Nres2, Nout])
    w_out_torch = torch.tensor(w_out).float()
    mod[3].weight.data = w_out_torch

    float_graph = mod.as_graph()
    float_specs = mapper(
        float_graph, weight_dtype="float", threshold_dtype="float", dash_dtype="float"
    )
    dt = float_specs["dt"]

    global_specs = copy.copy(float_specs)
    channel_specs = copy.copy(float_specs)

    del float_specs["mapped_graph"]
    del float_specs["dt"]
    xylo_conf_float, is_valid, message = config_from_specification(**float_specs)
    print("Float valid config: ", is_valid, "message")

    global_specs.update(global_quantize(**global_specs, fuzzy_scaling=False))
    del global_specs["mapped_graph"]
    del global_specs["dt"]
    xylo_conf_global, is_valid, message = config_from_specification(**global_specs)
    print("Global valid config: ", is_valid, message)

    channel_specs.update(channel_quantize(**channel_specs))
    del channel_specs["mapped_graph"]
    del channel_specs["dt"]
    xylo_conf_channel, is_valid, message = config_from_specification(**channel_specs)
    print("Channel valid config: ", is_valid, message)

    T = 1000
    batch = 1
    inp = torch.Tensor(np.random.randint(low=0, high=3, size=(batch, T, Nin)))

    mod.reset_state()
    _, _, recordings_f = mod(inp, record=True)

    cim_g = XyloSim.from_config(xylo_conf_global, dt=dt)
    cim_g.reset_state()
    spk_out_g, _, rec_cim_g = cim_g(inp[0].cpu().numpy(), record=True)

    cim_c = XyloSim.from_config(xylo_conf_channel, dt=dt)
    cim_c.reset_state()
    spk_out_c, _, rec_cim_c = cim_c(inp[0].cpu().numpy(), record=True)

    spk_res1_f = (
        recordings_f["1_LIFBitshiftTorch"]["spikes"]
        .squeeze(0)
        .detach()
        .numpy()
        .astype(int)
    )
    spk_res2_f = (
        recordings_f["2_TorchResidual"]["1_LIFBitshiftTorch"]["spikes"]
        .squeeze(0)
        .detach()
        .numpy()
        .astype(int)
    )
    spk_in_f = np.concatenate((spk_res1_f, spk_res2_f), axis=1)

    spk_out_f = (
        recordings_f["4_LIFBitshiftTorch"]["spikes"]
        .squeeze(0)
        .detach()
        .numpy()
        .astype(int)
    )

    spk_in_g = rec_cim_g["Spikes"]
    spk_in_c = rec_cim_c["Spikes"]

    in_point = spk_in_f.shape[0] * spk_in_f.shape[1]
    out_point = spk_out_f.shape[0] * spk_out_f.shape[1]

    print(
        f"\nIn LIF spike global match float percent: {np.sum(spk_in_f == spk_in_g) / in_point * 100}"
    )
    print(
        f"Out LIF spike global match float percent: {np.sum(spk_out_f == spk_out_g) / out_point * 100}"
    )

    print(
        f"\nIn LIF spike channel match float percent: {np.sum(spk_in_f == spk_in_c) / in_point * 100}"
    )
    print(
        f"Out LIF spike channel match float percent: {np.sum(spk_out_f == spk_out_c) / out_point * 100}"
    )

    print(
        f"\nIn LIF spike global match channel: {np.sum(spk_in_g == spk_in_c) / in_point * 100}"
    )
    print(
        f"Out LIF spike global match channel: {np.sum(spk_out_g == spk_out_c) / out_point * 100}"
    )
