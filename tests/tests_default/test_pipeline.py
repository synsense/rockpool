def test_pipeline():
    import numpy as np
    import torch
    import copy
    from rockpool.transform import global_quantize, channel_quantize
    from rockpool.nn.networks import WaveSenseNet
    from rockpool.devices.xylo import mapper, config_from_specification
    from rockpool.devices.xylo.xylo_cimulator import XyloCim
    import warnings
    warnings.filterwarnings('ignore')

    Net = WaveSenseNet(
        dilations=[2, 4, 8],
        n_classes=2,
        n_channels_in=16,
        n_channels_res=16,
        n_channels_skip=32,
        n_hidden=32,
        kernel_size=2,
    )

    float_graph = Net.as_graph()
    float_specs = mapper(float_graph, weight_dtype="float", threshold_dtype="float", dash_dtype="float")
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
    print('Global valid config: ', is_valid, message)

    channel_specs.update(channel_quantize(**channel_specs))
    del channel_specs["mapped_graph"]
    del channel_specs["dt"]
    xylo_conf_channel, is_valid, message = config_from_specification(**channel_specs)
    print('Channel valid config: ', is_valid, message)

    T = 100
    Nin = 16
    batch = 1
    inp = torch.Tensor(np.random.randint(low=1, high=10, size=(batch, T, Nin)))

    output_f, _, rec_rockpool_f = Net(inp, record=True)

    cim_g = XyloCim.from_config(xylo_conf_global, dt=dt)
    cim_g.reset_state()
    output_g, _, rec_cim_g = cim_g(inp.cpu().numpy(), record=True)

    cim_c = XyloCim.from_config(xylo_conf_channel, dt=dt)
    cim_c.reset_state()
    output_c, _, rec_cim_c = cim_c(inp.cpu().numpy(), record=True)






