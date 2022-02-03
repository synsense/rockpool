
def test_spike_clipping():
    from rockpool.nn.modules.torch.lif_torch import StepPWL, PeriodicExponential
    from rockpool.nn.modules.torch import LIFBitshiftTorch
    from rockpool.nn.modules import LIFTorch, LIFSlayer
    import torch

    n_synapses = 5
    n_neurons = 10
    n_batches = 3
    T = 20
    tau_mem = torch.rand(n_neurons)
    tau_syn = 0.05
    threshold = 1.34

    # - Test maximal initialisation
    mod_xylo_step = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        spike_generation_fn=StepPWL,
        threshold=threshold,
        max_spikes_per_dt=15,
    )
    mod_xylo_periodic = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        spike_generation_fn=PeriodicExponential,
        threshold=threshold,
        max_spikes_per_dt=15,
    )
    mod = LIFTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        threshold=threshold,
    )
    mod_bitshift = LIFBitshiftTorch(
        shape=(n_synapses * n_neurons, n_neurons),
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        has_rec=False,
        dt=1e-3,
        noise_std=0.0,
        threshold=threshold,
    )

    # - Generate some data
    input_data = 0.5*torch.ones(n_batches, T, n_synapses * n_neurons)

    # - test that number of spikes does not exceed 15
    out, _, rd = mod(input_data, record=True)
    out_xylo_step, _, rd_xylo_step = mod_xylo_step(input_data, record=True)
    out_xylo_periodic, _, _ = mod_xylo_periodic(input_data, record=True)
    # - test default gradient of xylo
    out_bitshift, _, _ = mod_bitshift(input_data, record=True)

    assert torch.any(out > 15)
    assert torch.all(out_xylo_step <= 15)
    assert torch.all(out_xylo_periodic <= 15)
    assert torch.all(out_bitshift <= 15)

    # - test that membrane potential is not reset entirely when spikes are clipped
    batch = 0
    neuron = 0

    t_spike = torch.where(out[batch, :, neuron] > 15)[0][0]
    vmem = rd['vmem'][batch, :, neuron]
    vmem_xylo_step = rd_xylo_step['vmem'][batch, :, neuron]
    spike_diff = (out[batch, : t_spike+1, neuron]-out_xylo_step[batch, : t_spike+1, neuron])
    assert torch.allclose(vmem[:t_spike+1],  vmem_xylo_step[:t_spike+1] - spike_diff*threshold)


    if torch.cuda.is_available():
        mod_slayer = LIFSlayer(
            shape=(n_synapses * n_neurons, n_neurons),
            tau_mem=tau_mem[0].item(),
            tau_syn=tau_syn,
            has_rec=False,
            dt=1e-3,
            noise_std=0.0,
            threshold=threshold,
        ).cuda()
        mod_slayer_xylo = LIFSlayer(
            shape=(n_synapses * n_neurons, n_neurons),
            tau_mem=tau_mem[0].item(),
            tau_syn=tau_syn,
            has_rec=False,
            dt=1e-3,
            noise_std=0.0,
            threshold=threshold,
            max_spikes_per_dt=15,
        ).cuda()

        out_slayer, _, rd_slayer = mod_slayer(input_data.cuda(), record=True)
        out_slayer_xylo, _, rd_slayer_xylo = mod_slayer_xylo(input_data.cuda(), record=True)

        assert torch.any(out_slayer > 15)
        assert torch.all(out_slayer_xylo <= 15)

        t_spike = torch.where(out_slayer[batch, :, neuron] > 15)[0][0]
        vmem = rd_slayer['vmem'][batch, :, neuron]
        vmem_slayer_xylo = rd_slayer_xylo['vmem'][batch, :, neuron]
        spike_diff = (out_slayer[batch, : t_spike + 1, neuron] - out_slayer_xylo[batch, : t_spike + 1, neuron])
        assert torch.allclose(vmem[:t_spike + 1], vmem_slayer_xylo[:t_spike + 1] - spike_diff * threshold)


