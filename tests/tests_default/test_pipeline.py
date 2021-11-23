def test_pipeline():
    import numpy as np
    import torch
    import copy
    from rockpool.transform import global_quantize, channel_quantize
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.nn.modules.torch import (
        TorchModule,
        LinearTorch,
        LIFTorch,
        LIFBitshiftTorch,
    )
    from rockpool.devices.xylo import mapper, config_from_specification
    from rockpool.devices.xylo.xylo_cimulator import XyloCim
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
        has_bias=False,
        smooth_output=False,
        tau_mem=tau_mem,
        base_tau_syn=base_tau_syn,
        tau_lp=tau_lp,
        threshold=threshold,
        neuron_model=LIFBitshiftTorch,
        dt=dt,
        device=device,
    )

    w_1_torch = torch.nn.init.normal_(mod.submods[0].weight, mean=0.0, std=5.0)
    w_w0_1_torch = torch.nn.init.normal_(
        mod.submods[2].submods[0].weight, mean=0.0, std=5.0
    )
    w_w1_1_torch = torch.nn.init.normal_(
        mod.submods[3].submods[0].weight, mean=0.0, std=5.0
    )
    w_hid_torch = torch.nn.init.normal_(mod.submods[4].weight, mean=0.0, std=5.0)
    w_out_torch = torch.nn.init.normal_(mod.submods[6].weight, mean=0.0, std=5.0)

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

    cim_g = XyloCim.from_config(xylo_conf_global, dt=dt)
    cim_g.reset_state()
    spk_out_g, _, rec_cim_g = cim_g(inp[0].cpu().numpy(), record=True)

    cim_c = XyloCim.from_config(xylo_conf_channel, dt=dt)
    cim_c.reset_state()
    spk_out_c, _, rec_cim_c = cim_c(inp[0].cpu().numpy(), record=True)
