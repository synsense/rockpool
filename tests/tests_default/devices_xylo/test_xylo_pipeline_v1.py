import pytest

pytest.importorskip("torch")
pytest.importorskip("samna")
pytest.importorskip("xylosim")


def test_pipeline_v1():
    from rockpool.devices.xylo.syns61300 import mapper, config_from_specification
    from rockpool.devices.xylo.syns61300 import XyloSim

    import numpy as np
    import torch
    import copy
    from rockpool.transform.quantize_methods import global_quantize, channel_quantize
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules import (
        LIFBitshiftTorch,
    )
    from rockpool.parameters import Constant
    import warnings

    warnings.filterwarnings("ignore")

    dilations = [2, 4]
    Nin = 16
    Nres = 12
    Nskip = 32
    Nhid = 8
    Nout = 2
    kernel_size = 2
    tau_mem = 0.002
    base_tau_syn = 0.002
    tau_lp = 0.01
    threshold = 1.0
    dt = 0.001
    device = "cpu"

    mod = WaveSenseNet(
        dilations=dilations,
        n_classes=Nout,
        n_channels_in=Nin,
        n_channels_res=Nres,
        n_channels_skip=Nskip,
        n_hidden=Nhid,
        kernel_size=kernel_size,
        bias=Constant(0.0),
        smooth_output=False,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFBitshiftTorch,
        dt=dt,
    )

    w_1_torch = torch.nn.init.normal_(mod.lin1.weight, mean=0.0, std=5.0)
    w_w0_1_torch = torch.nn.init.normal_(mod.wave0.lin1.weight, mean=0.0, std=5.0)
    w_w1_1_torch = torch.nn.init.normal_(mod.wave1.lin1.weight, mean=0.0, std=5.0)
    w_hid_torch = torch.nn.init.normal_(mod.hidden.weight, mean=0.0, std=5.0)
    w_out_torch = torch.nn.init.normal_(mod.readout.weight, mean=0.0, std=5.0)

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
    Nin = 16
    batch = 1
    inp = torch.Tensor(np.random.randint(low=1, high=10, size=(batch, T, Nin)))

    mod.reset_state()
    _, _, recordings_f = mod(inp, record=True)

    cim_g = XyloSim.from_config(xylo_conf_global, dt=dt)
    cim_g.reset_state()
    spk_out_g, _, rec_cim_g = cim_g(inp[0].cpu().numpy(), record=True)

    cim_c = XyloSim.from_config(xylo_conf_channel, dt=dt)
    cim_c.reset_state()
    spk_out_c, _, rec_cim_c = cim_c(inp[0].cpu().numpy(), record=True)


def test_rec_rockpool():
    try:
        from rockpool.devices.xylo import mapper
        from rockpool.devices.xylo import config_from_specification
        from rockpool.devices.xylo import XyloSim
    except:
        return

    import numpy as np
    import torch
    import copy
    from rockpool.transform.quantize_methods import global_quantize, channel_quantize

    from rockpool.parameters import Constant
    from rockpool.nn.combinators import Sequential
    from rockpool.nn.modules import (
        TorchModule,
        LinearTorch,
        LIFBitshiftTorch,
    )
    import warnings

    warnings.filterwarnings("ignore")
    from rockpool.graph import (
        GraphHolder,
        connect_modules,
    )

    Nin = 3
    Nres1 = 4
    Nres2 = 4
    Nres3 = 4
    Nres4 = 4
    Nout = 2
    threshold = 1

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
        LinearTorch((Nres1, Nres2), has_bias=False),
        LIFBitshiftTorch(
            shape=(Nres2, Nres2),
            tau_mem=0.002,
            tau_syn=0.002,
            bias=Constant(0.0),
            threshold=threshold,
            learning_window=0.5,
            dt=0.001,
        ),
        LinearTorch((Nres2, Nres3), has_bias=False),
        LIFBitshiftTorch(
            shape=(Nres3, Nres3),
            tau_mem=0.002,
            tau_syn=0.002,
            bias=Constant(0.0),
            threshold=threshold,
            learning_window=0.5,
            dt=0.001,
        ),
        LinearTorch((Nres3, Nres4), has_bias=False),
        LIFBitshiftTorch(
            shape=(Nres4, Nres4),
            tau_mem=0.002,
            tau_syn=0.002,
            bias=Constant(0.0),
            threshold=threshold,
            learning_window=0.5,
            dt=0.001,
        ),
        LinearTorch((Nres4, Nout), has_bias=False),
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

    w_res1_torch = torch.nn.init.normal_(mod[0].weight, mean=0.2, std=1.0)
    w_res2_torch = torch.nn.init.normal_(mod[2].weight, mean=0.2, std=1.0)
    w_res3_torch = torch.nn.init.normal_(mod[4].weight, mean=0.2, std=1.0)
    w_res4_torch = torch.nn.init.normal_(mod[6].weight, mean=0.2, std=1.0)
    w_out_torch = torch.nn.init.normal_(mod[8].weight, mean=0.2, std=1.0)

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

    T = 10
    batch = 1
    inp = torch.Tensor(np.random.randint(low=0, high=4, size=(batch, T, Nin)))
    # inp = torch.zeros((batch, T, Nin))
    # inp[0, 1, :] = 1

    mod.reset_state()
    _, _, recordings_f = mod(inp, record=True)

    cim_g = XyloSim.from_config(xylo_conf_global, dt=dt)
    cim_g.reset_state()
    spk_out_g, _, rec_cim_g = cim_g(inp[0].cpu().numpy(), record=True)

    cim_c = XyloSim.from_config(xylo_conf_channel, dt=dt)
    cim_c.reset_state()
    spk_out_c, _, rec_cim_c = cim_c(inp[0].cpu().numpy(), record=True)

    class FloatRec(TorchModule):
        def __init__(
            self,
            Nin: int,
            Nres: int,
            Nout: int,
            weights_in: torch.Tensor,
            weights_rec: torch.Tensor,
            weights_out: torch.Tensor,
            dash_mem: torch.Tensor,
            dash_mem_out: torch.Tensor,
            dash_syn: torch.Tensor,
            dash_syn_2: torch.Tensor,
            dash_syn_out: torch.Tensor,
            threshold: torch.Tensor,
            threshold_out: torch.Tensor,
            aliases: list,
            device: str = "cpu",
            *args,
            **kwargs,
        ):
            super().__init__(shape=(Nin, Nout), *args, **kwargs)

            learning_window = 0.5
            dt = 0.001

            self.lin_res = LinearTorch(
                shape=(Nin, Nres), weight=weights_in, has_bias=False
            )

            self.spk_res = LIFBitshiftTorch(
                shape=(Nres, Nres),  # TODO: test when Nres*2
                tau_mem=dash_mem,
                tau_syn=dash_syn,
                bias=Constant(0.0),
                threshold=threshold,
                has_rec=True,
                w_rec=weights_rec,
                learning_window=learning_window,
                dt=dt,
            )

            self.lin_out = LinearTorch(
                shape=(Nres, Nout), weight=weights_out, has_bias=False
            )

            self.spk_out = LIFBitshiftTorch(
                shape=(Nout, Nout),
                tau_mem=dash_mem_out,
                tau_syn=dash_syn_out,
                bias=Constant(0.0),
                threshold=threshold_out,
                learning_window=learning_window,
                dt=dt,
            )

            self._record_dict = {}

            # self.submods = []
            # self.submods.append(self.lin_res)
            # self.submods.append(self.spk_res)
            # self.submods.append(self.lin_out)
            # self.submods.append(self.spk_out)

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

            for mod in self.modules():
                mod_graphs.append(mod.as_graph())

            connect_modules(mod_graphs[0], mod_graphs[1])
            connect_modules(mod_graphs[1], mod_graphs[2])
            connect_modules(mod_graphs[2], mod_graphs[3])

            return GraphHolder(
                mod_graphs[0].input_nodes,
                mod_graphs[3].output_nodes,
                f"{type(self).__name__}_{self.name}_{id(self)}",
            )

    mod = FloatRec(
        Nin=float_specs["weights_in"].shape[0],
        Nres=float_specs["weights_rec"].shape[1],
        Nout=float_specs["weights_out"].shape[1],
        weights_in=torch.from_numpy(float_specs["weights_in"]).to(torch.float32),
        weights_rec=torch.from_numpy(float_specs["weights_rec"]).to(torch.float32),
        weights_out=torch.from_numpy(float_specs["weights_out"]).to(torch.float32),
        dash_mem=torch.from_numpy(float_specs["dash_mem"]).to(torch.float32),
        dash_mem_out=torch.from_numpy(float_specs["dash_mem_out"]).to(torch.float32),
        dash_syn=torch.from_numpy(float_specs["dash_syn"]).to(torch.float32),
        dash_syn_2=torch.from_numpy(float_specs["dash_syn_2"]).to(torch.float32),
        dash_syn_out=torch.from_numpy(float_specs["dash_syn_out"]).to(torch.float32),
        threshold=torch.from_numpy(float_specs["threshold"]).to(torch.float32),
        threshold_out=torch.from_numpy(float_specs["threshold_out"]).to(torch.float32),
        aliases=float_specs["aliases"],
    )

    mod.reset_state()
    _, _, recordings_f = mod(inp, record=True)
